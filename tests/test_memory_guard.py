"""Memory guard: shed load deliberately instead of swapping.

Pressure is injected as synthetic samples — the point is the escalation policy
(cheapest mitigation first, one step at a time, never down to zero cameras), not
whether this machine happens to be short of RAM today.
"""
from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.serving.memory_guard import (
    CRITICAL, OK, WARN, MemoryGuard, MemorySample, build_default_actions, sample_memory,
)


def _s(available_gb, rss=3.0):
    return MemorySample(rss_gb=rss, available_gb=available_gb, percent_used=50.0)


class SampleTests(unittest.TestCase):
    def test_levels_by_available_memory(self):
        self.assertEqual(_s(8.0).level(2.0, 1.0), OK)
        self.assertEqual(_s(1.5).level(2.0, 1.0), WARN)
        self.assertEqual(_s(0.5).level(2.0, 1.0), CRITICAL)

    def test_boundaries_are_inclusive(self):
        self.assertEqual(_s(2.0).level(2.0, 1.0), WARN)
        self.assertEqual(_s(1.0).level(2.0, 1.0), CRITICAL)

    def test_real_sample_is_sane(self):
        s = sample_memory()
        self.assertGreater(s.rss_gb, 0.0)
        self.assertIn("rss_gb", s.to_dict())


class EscalationTests(unittest.TestCase):
    def setUp(self):
        self.done = []
        self.guard = MemoryGuard(
            warn_available_gb=2.0, critical_available_gb=1.0,
            warn_actions=[lambda: self.done.append("w1") or "w1",
                          lambda: self.done.append("w2") or "w2"],
            critical_actions=[lambda: self.done.append("c1") or "c1"])

    def test_healthy_memory_does_nothing(self):
        self.assertEqual(self.guard.check(_s(8.0)), OK)
        self.assertEqual(self.done, [])

    def test_one_step_per_check_not_everything_at_once(self):
        self.guard.check(_s(1.5))
        self.assertEqual(self.done, ["w1"])          # give the cheap fix a chance first
        self.guard.check(_s(1.5))
        self.assertEqual(self.done, ["w1", "w2"])

    def test_actions_are_exhausted_not_repeated(self):
        for _ in range(5):
            self.guard.check(_s(1.5))
        self.assertEqual(self.done, ["w1", "w2"])    # no infinite re-application

    def test_critical_uses_its_own_ladder(self):
        self.guard.check(_s(0.5))
        self.assertEqual(self.done, ["c1"])
        self.assertEqual(self.guard.level, CRITICAL)

    def test_recovery_is_reported_but_load_is_not_grown_back(self):
        self.guard.check(_s(1.5))
        self.guard.check(_s(9.0))
        self.assertEqual(self.guard.level, OK)
        self.assertEqual(self.done, ["w1"])          # never auto-undone (avoids flapping)

    def test_a_failing_mitigation_does_not_crash_the_engine(self):
        def boom():
            raise RuntimeError("nope")
        g = MemoryGuard(warn_actions=[boom], warn_available_gb=2.0)
        self.assertEqual(g.check(_s(1.5)), WARN)
        self.assertEqual(g.mitigations, [])

    def test_status_reports_what_was_given_up(self):
        self.guard.check(_s(1.5))
        st = self.guard.status()
        self.assertEqual(st["level"], WARN)
        self.assertIn("w1", st["mitigations"])
        self.assertIsNotNone(st["memory"])


class _Decoder:
    def __init__(self):
        self.stopped = False
        self.target_fps = 4.0
        self._min_period = 0.25

    def stop(self):
        self.stopped = True


class _Pipe:
    def __init__(self, n=3):
        self.target_fps = 4.0
        self.imgsz = 512
        self._decoders = {f"cam{i}": _Decoder() for i in range(n)}


class _State:
    def __init__(self):
        self._clip_buffer = deque(range(48), maxlen=48)
        self._video_runtime = object()
        self.video_action = True


class DefaultActionTests(unittest.TestCase):
    def setUp(self):
        self.pipe = _Pipe()
        self.states = {"cam0": _State(), "cam1": _State()}
        self.warn, self.crit = build_default_actions(self.pipe, self.states)

    def test_warn_gives_up_quality_first(self):
        self.assertIsNotNone(self.warn[0]())                      # trim buffers
        self.assertEqual(len(self.states["cam0"]._clip_buffer), 16)
        self.warn[1]()                                            # halve fps
        self.assertEqual(self.pipe.target_fps, 2.0)
        for d in self.pipe._decoders.values():
            self.assertEqual(d.target_fps, 2.0)
        self.warn[2]()                                            # smaller frames
        self.assertEqual(self.pipe.imgsz, 320)

    def test_critical_drops_the_heaviest_model_then_a_camera(self):
        what = self.crit[0]()
        self.assertIn("video-action", what)
        self.assertFalse(self.states["cam0"].video_action)
        before = len(self.pipe._decoders)
        self.assertIn("stopped camera", self.crit[1]())
        self.assertEqual(len(self.pipe._decoders), before - 1)

    def test_never_sheds_the_last_camera(self):
        pipe = _Pipe(n=1)
        _, crit = build_default_actions(pipe, {})
        self.assertIsNone(crit[1]())
        self.assertEqual(len(pipe._decoders), 1)     # blind is worse than degraded

    def test_actions_are_idempotent_when_nothing_is_left_to_give(self):
        self.warn[1]()
        self.warn[1]()
        self.assertGreaterEqual(self.pipe.target_fps, 2.0)   # floor respected
        self.warn[2]()
        self.assertIsNone(self.warn[2]())                    # already minimal


if __name__ == "__main__":
    unittest.main()
