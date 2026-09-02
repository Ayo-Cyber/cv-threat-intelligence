"""VideoMAE runs OFF the frame loop (latency audit 1 Sep, D1).

The transformer forward pass (2–6s on pilot hardware) used to run inside
process_frame, in the shared detection loop — one camera's theft check
stalled every camera on the site. The contract now:

- the frame loop's video-action work is O(1): drain finished verdicts,
  snapshot a clip, submit it — NEVER wait on the model;
- one shared worker runs clips oldest-first across cameras, and a newer clip
  from the same camera replaces its pending one (a clip that waited behind a
  slow inference describes a scene that no longer exists);
- weights load at startup, in the phase whose heartbeat already says
  'loading detection models' — not inside the first real alert.
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.contracts import RawEvent
from cvti.video_action_runtime import AsyncVideoActionRunner, VideoActionRuntime


def _event(title: str) -> RawEvent:
    return RawEvent(detector="video_action", active=True, title=title, level="high")


class _SlowClip:
    """A clip whose inference blocks until the test releases it."""

    def __init__(self, title: str):
        self.title = title
        self.release = threading.Event()
        self.started = threading.Event()
        self.ran = False

    def __call__(self) -> list[RawEvent]:
        self.started.set()
        self.release.wait(timeout=10)
        self.ran = True
        return [_event(self.title)]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = AsyncVideoActionRunner().start()

    def tearDown(self):
        # Unblock anything still held so stop() can join the worker.
        self.runner.stop()

    def test_submit_never_blocks_the_caller(self):
        """THE point of D1: the frame loop hands off and returns immediately,
        even while the worker is mid-inference."""
        slow = _SlowClip("first")
        self.runner.submit("cam1", slow)
        self.assertTrue(slow.started.wait(timeout=5))     # worker is busy in it
        t0 = time.monotonic()
        self.runner.submit("cam1", lambda: [_event("second")])
        self.assertEqual(self.runner.drain("cam1"), [])   # nothing finished yet
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.1)                     # frame-loop side is O(1)
        slow.release.set()

    def test_verdicts_arrive_on_a_later_drain(self):
        slow = _SlowClip("verdict")
        self.runner.submit("cam1", slow)
        self.assertTrue(slow.started.wait(timeout=5))
        slow.release.set()
        deadline = time.monotonic() + 5
        events = []
        while not events and time.monotonic() < deadline:
            events = self.runner.drain("cam1")
            time.sleep(0.02)
        self.assertEqual([e.title for e in events], ["verdict"])
        self.assertEqual(self.runner.drain("cam1"), [])   # drain empties

    def test_latest_clip_wins_per_camera(self):
        """While the worker is stuck in camera A's clip, camera B submits
        twice: only B's NEWEST clip may run, and the replacement is counted."""
        blocker = _SlowClip("A")
        self.runner.submit("camA", blocker)
        self.assertTrue(blocker.started.wait(timeout=5))
        stale, fresh = _SlowClip("B-stale"), _SlowClip("B-fresh")
        stale.release.set()
        fresh.release.set()
        self.runner.submit("camB", stale)
        self.runner.submit("camB", fresh)                 # replaces the stale one
        self.assertEqual(self.runner.dropped, 1)
        blocker.release.set()
        deadline = time.monotonic() + 5
        events = []
        while not events and time.monotonic() < deadline:
            events = self.runner.drain("camB")
            time.sleep(0.02)
        self.assertEqual([e.title for e in events], ["B-fresh"])
        self.assertFalse(stale.ran)

    def test_oldest_submission_runs_first_across_cameras(self):
        """A busy camera cannot starve a quiet one: the worker serves
        submissions oldest-first across the whole site."""
        gate = _SlowClip("gate")
        self.runner.submit("cam0", gate)
        self.assertTrue(gate.started.wait(timeout=5))
        order: list[str] = []
        for cam in ("camA", "camB", "camC"):
            self.runner.submit(cam, lambda c=cam: order.append(c) or [])
            time.sleep(0.01)                              # distinct submit times
        gate.release.set()
        deadline = time.monotonic() + 5
        while len(order) < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(order, ["camA", "camB", "camC"])

    def test_a_failing_clip_never_kills_the_worker(self):
        def boom() -> list[RawEvent]:
            raise RuntimeError("cuda fell over")
        self.runner.submit("cam1", boom)
        self.runner.submit("cam2", lambda: [_event("still alive")])
        deadline = time.monotonic() + 5
        events = []
        while not events and time.monotonic() < deadline:
            events = self.runner.drain("cam2")
            time.sleep(0.02)
        self.assertEqual([e.title for e in events], ["still alive"])
        self.assertEqual(self.runner.failed, 1)


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def predict_frames(self, frames, *, top_k=5):
        from cvti.video_action_model import VideoActionPrediction
        self.calls += 1
        return [VideoActionPrediction(label="punching person (boxing)",
                                      confidence=0.9, rank=1)]


def _runtime(model=None, cooldown=2.0) -> VideoActionRuntime:
    import numpy as np
    rt = VideoActionRuntime(model=model or _FakeModel(), backend="videomae",
                            model_name="fake", fps=5.0, cooldown_seconds=cooldown)
    for i in range(20):
        rt.add_frame(np.zeros((32, 32, 3), dtype=np.uint8), frame_index=i)
    return rt


class PrepareAnalysisTests(unittest.TestCase):
    def test_prepare_snapshots_now_and_defers_the_model(self):
        model = _FakeModel()
        rt = _runtime(model)
        clip = rt.prepare_analysis(center_frame_index=10, timestamp=100.0)
        self.assertIsNotNone(clip)
        self.assertEqual(model.calls, 0)                  # nothing heavy yet
        events = clip()
        self.assertEqual(model.calls, 1)
        self.assertTrue(events and events[0].detector == "video_action")

    def test_cooldown_throttles_at_submission_time(self):
        """The stamp moves to prepare(): it must throttle how often clips are
        QUEUED — stamping at completion would let every frame of a 3s
        inference window queue another stale clip behind it."""
        rt = _runtime()
        self.assertIsNotNone(rt.prepare_analysis(center_frame_index=10, timestamp=100.0))
        self.assertIsNone(rt.prepare_analysis(center_frame_index=11, timestamp=101.0))
        self.assertIsNotNone(rt.prepare_analysis(center_frame_index=12, timestamp=102.5))

    def test_sync_analyze_event_is_unchanged_for_the_cli_path(self):
        rt = _runtime()
        events = rt.analyze_event(center_frame_index=10, timestamp=100.0)
        self.assertTrue(events)
        self.assertEqual(rt.analyze_event(center_frame_index=11, timestamp=101.0), [])


class WiringPins(unittest.TestCase):
    """Source-order pins, in the house style of test_scene_mapping_pipeline."""

    def test_engine_frame_loop_uses_the_runner_not_the_model(self):
        import inspect
        from cvti.serving import camera
        src = inspect.getsource(camera.PerCameraState.process)
        self.assertIn("va_runner.drain", src)
        self.assertIn("prepare_analysis", src)
        # Sync analysis survives only as the no-runner fallback.
        self.assertIn("self.va_runner is not None", src)

    def test_run_site_loads_weights_eagerly_and_starts_one_runner(self):
        import inspect
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.run_site)
        self.assertIn("video_action_model.load()", src)
        self.assertIn("AsyncVideoActionRunner().start()", src)
        self.assertIn("va_runner.stop()", src)

    def test_model_load_is_public_and_idempotent(self):
        from cvti.video_action_model import VideoMAEActionModel
        model = VideoMAEActionModel("fake-model")
        with mock.patch.object(model, "_load") as loaded:
            model.load()
        loaded.assert_called_once()


if __name__ == "__main__":
    unittest.main()
