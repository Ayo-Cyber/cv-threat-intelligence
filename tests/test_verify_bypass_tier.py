"""The measured bypass tier: violence alerts fire instantly, theft stays judged.

The bakeoff's per-rule breakdown (runs/eval/kpi/bakeoff_summary.json, manifest
e56b4277) showed the violence detector fired on 0 of 170 normal clips while the
VLM rejected 12 of its 79 true positives — on that rule the judge only
subtracts, and costs ~12s on the one alert where seconds matter most. These
tests hold the tier's three promises: violence bypasses by default, the theft
path is still judged, and one site-file line ("verify_bypass": []) reverts
everything. The eval harness must mirror the engine exactly, or every future
scorecard measures a system nobody ships.
"""
import time
import unittest

import numpy as np

from cvti.contracts import CandidateAlert
from cvti.eval.harness import EvalHarness
from cvti.serving.alert_queue import AlertQueue, QueuedAlert
from cvti.serving.gate_pool import (
    BYPASS_DETECTORS, MEASURED_BYPASS, GatePool, bypass_from_site,
)


def _candidate(detector="violence", rule="violence"):
    return CandidateAlert(
        rule_name=rule, priority="critical", detector=detector,
        title="VIOLENCE: physical altercation", person_id=None,
        object_label=None, timestamp=0.0)


def _queued(candidate):
    return QueuedAlert(
        camera_id="aisle_1", rule_name=candidate.rule_name,
        priority=candidate.priority, title=candidate.title, timestamp=0.0,
        track_id=1, zone=None, object_label=None,
        payload={"candidate": candidate,
                 "frames": [np.zeros((32, 32, 3), dtype=np.uint8)],
                 "scene": {}})


class _RecordingGate:
    """A gate that must NOT be consulted for bypassed detectors."""

    def __init__(self):
        self.calls = []

    def verify(self, frames, candidate, scene, examples=None):
        self.calls.append(candidate)
        from cvti.contracts import VerificationResult
        return VerificationResult(confirmed=False, confidence=0.9,
                                  reason="judged and rejected",
                                  alert_priority="high", timestamp=time.time(),
                                  raw_response="{}")


def _run_one(pool, alert, timeout=3.0):
    pool.queue.add(alert)
    pool.start()
    deadline = time.time() + timeout
    while pool.verified + pool.unverified + pool.errors == 0:
        if time.time() > deadline:
            raise AssertionError("pool never produced a verdict")
        time.sleep(0.01)
    pool._stop.set()


class TheMeasuredTierBypasses(unittest.TestCase):
    def test_violence_is_in_the_default_tier(self):
        self.assertIn("violence", MEASURED_BYPASS)

    def test_violence_auto_confirms_without_a_vlm_call(self):
        gate = _RecordingGate()
        verdicts = []
        pool = GatePool(AlertQueue(), gate_factory=lambda: gate,
                        on_verdict=lambda a, r: verdicts.append(r))
        _run_one(pool, _queued(_candidate("violence")))
        self.assertEqual(gate.calls, [], "the VLM was consulted on a bypassed detector")
        self.assertEqual(len(verdicts), 1)
        self.assertTrue(verdicts[0].confirmed)
        self.assertIn("measured-clean", verdicts[0].reason)
        self.assertEqual(verdicts[0].raw_response, "bypass")

    def test_the_deterministic_tier_keeps_its_own_wording(self):
        gate = _RecordingGate()
        verdicts = []
        pool = GatePool(AlertQueue(), gate_factory=lambda: gate,
                        on_verdict=lambda a, r: verdicts.append(r))
        _run_one(pool, _queued(_candidate("presence", rule="loiter_zone")))
        self.assertEqual(gate.calls, [])
        self.assertIn("deterministic", verdicts[0].reason)

    def test_video_action_is_still_judged(self):
        gate = _RecordingGate()
        pool = GatePool(AlertQueue(), gate_factory=lambda: gate,
                        on_verdict=lambda a, r: None)
        _run_one(pool, _queued(_candidate("video_action", rule="video_theft_candidate")))
        self.assertEqual(len(gate.calls), 1, "the theft path must stay gated")

    def test_an_empty_site_list_restores_full_gating_for_violence(self):
        gate = _RecordingGate()
        pool = GatePool(AlertQueue(), gate_factory=lambda: gate,
                        on_verdict=lambda a, r: None, bypass=set())
        _run_one(pool, _queued(_candidate("violence")))
        self.assertEqual(len(gate.calls), 1, "verify_bypass: [] must re-gate violence")

    def test_the_deterministic_set_is_never_removable(self):
        pool = GatePool(AlertQueue(), gate_factory=_RecordingGate, bypass=set())
        self.assertTrue(BYPASS_DETECTORS <= pool.bypass)


class TheSiteFileControlsTheTier(unittest.TestCase):
    def test_absent_key_accepts_the_measured_default(self):
        self.assertIsNone(bypass_from_site({}))

    def test_empty_list_is_a_veto_not_an_absence(self):
        self.assertEqual(bypass_from_site({"verify_bypass": []}), set())

    def test_names_listed_become_the_tier(self):
        self.assertEqual(bypass_from_site({"verify_bypass": ["violence", "weapons"]}),
                         {"violence", "weapons"})


class TheHarnessMirrorsTheEngine(unittest.TestCase):
    """Scorecards must measure the system as shipped."""

    def _harness(self, **kw):
        return EvalHarness(gate=_RecordingGate(), **kw)

    def test_default_tier_matches_the_engine_exactly(self):
        h = self._harness()
        self.assertEqual(h.bypass, BYPASS_DETECTORS | MEASURED_BYPASS)

    def test_a_bypassed_candidate_confirms_without_a_gate_call(self):
        h = self._harness()
        confirmed = h._confirm(_queued(_candidate("violence")))
        self.assertTrue(confirmed)
        self.assertEqual(h.gate.calls, [])

    def test_a_gated_candidate_still_asks_the_gate(self):
        h = self._harness()
        confirmed = h._confirm(_queued(_candidate("video_action",
                                                  rule="video_theft_candidate")))
        self.assertFalse(confirmed)          # _RecordingGate rejects
        self.assertEqual(len(h.gate.calls), 1)

    def test_an_explicit_empty_tier_measures_the_pre_tier_system(self):
        h = self._harness(bypass=set())
        h._confirm(_queued(_candidate("violence")))
        self.assertEqual(len(h.gate.calls), 1)


if __name__ == "__main__":
    unittest.main()
