"""Gate workers match the server, and every local call knows its budget
(latency audit 1 Sep, V2 + V4b).

Three promises:

- the gate pool never runs more workers than the local server has parallel
  slots (a third worker on OLLAMA_NUM_PARALLEL=2 queued server-side and read
  as gate latency), and a CPU-only box runs exactly one;
- a live verify retries once, not three times, with a timeout derived from
  this machine's observed verify latency — fail-visible UNVERIFIED beats an
  18-minute silent retry storm holding a slot;
- the scanner and watches never hold a slot patiently: no retries, 120s cap.
  Scene mapping alone keeps the patient 360s x3 (startup, 1 Sep lesson).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.contracts import CandidateAlert
from cvti.serving.pipeline import _gate_workers_for
from cvti.verification import ollama
from cvti.verification.gate import VerificationGate


class SlotCountTests(unittest.TestCase):
    # The default now follows the low-memory probe (4 Sep) — these pin the
    # ROOMY machine's contract, so the probe is fixed, not inherited from
    # whatever RAM the test host happens to have (CI runners sit exactly at
    # the 16GB threshold and flipped these).
    def test_defaults_to_the_two_slots_we_spawn_with(self):
        with mock.patch.object(ollama, "_low_memory_box", return_value=False), \
             mock.patch.dict(ollama.os.environ, {}, clear=False):
            ollama.os.environ.pop("OLLAMA_NUM_PARALLEL", None)
            self.assertEqual(ollama.configured_parallel_slots(), 2)

    def test_operator_env_override_wins(self):
        with mock.patch.dict(ollama.os.environ, {"OLLAMA_NUM_PARALLEL": "4"}):
            self.assertEqual(ollama.configured_parallel_slots(), 4)

    def test_garbage_env_falls_back(self):
        with mock.patch.object(ollama, "_low_memory_box", return_value=False), \
             mock.patch.dict(ollama.os.environ, {"OLLAMA_NUM_PARALLEL": "many"}):
            self.assertEqual(ollama.configured_parallel_slots(), 2)


class WorkerSizingTests(unittest.TestCase):
    def test_an_explicit_request_always_wins(self):
        self.assertEqual(_gate_workers_for(5, 2, provider="ollama", device="cpu"), 5)

    def test_local_workers_never_exceed_the_servers_slots(self):
        # 6 cameras used to derive 3 workers; on a 2-slot server the third
        # queued server-side. Now: min(derived, slots).
        self.assertEqual(
            _gate_workers_for(0, 6, provider="ollama", device="cuda", slots=2), 2)

    def test_more_slots_than_cameras_need_stays_camera_derived(self):
        self.assertEqual(
            _gate_workers_for(0, 2, provider="ollama", device="cuda", slots=4), 1)

    def test_a_cpu_only_box_runs_exactly_one_worker(self):
        self.assertEqual(
            _gate_workers_for(0, 6, provider="ollama", device="cpu", slots=2), 1)
        self.assertEqual(
            _gate_workers_for(0, 6, provider="local", device="cpu", slots=2), 1)

    def test_cloud_gates_keep_the_measured_camera_sizing(self):
        # 46.5s -> 28.0s with two workers on 5 cameras; clouds have no slots
        # of ours to respect.
        self.assertEqual(_gate_workers_for(0, 5, provider="anthropic", device="cpu"), 2)
        self.assertEqual(_gate_workers_for(0, 6, provider="openrouter", device="cpu"), 3)


class _Wire:
    """Capture what the gate hands the shared transport helper."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({"confirmed": False, "confidence": 0.9,
                           "reason": "empty scene", "alert_priority": "high"})


def _alert() -> CandidateAlert:
    return CandidateAlert(rule_name="theft_attempt", priority="high",
                          detector="video_action", title="t", person_id=1,
                          object_label=None, timestamp=0.0)


class VerifyBudgetTests(unittest.TestCase):
    def _gate(self) -> VerificationGate:
        return VerificationGate(provider="ollama")

    def test_a_live_verify_retries_once_not_three_times(self):
        wire = _Wire()
        gate = self._gate()
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible", wire):
            gate.verify(np.zeros((64, 64, 3), dtype=np.uint8), _alert())
        self.assertEqual(wire.calls[0]["max_retries"], 1)

    def test_first_verify_keeps_the_full_ceiling_for_the_cold_load(self):
        wire = _Wire()
        gate = self._gate()
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible", wire):
            gate.verify(np.zeros((64, 64, 3), dtype=np.uint8), _alert())
        self.assertEqual(wire.calls[0]["timeout"], gate.TIMEOUT_CEIL_S)

    def test_timeout_tracks_observed_latency_with_a_floor(self):
        wire = _Wire()                       # instant fake verifies -> EMA ~0
        gate = self._gate()
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible", wire):
            gate.verify(np.zeros((64, 64, 3), dtype=np.uint8), _alert())
            gate.verify(np.zeros((64, 64, 3), dtype=np.uint8), _alert())
        self.assertEqual(wire.calls[1]["timeout"], gate.TIMEOUT_FLOOR_S)

    def test_timeout_derivation_is_clamped_both_ways(self):
        gate = self._gate()
        gate._latency_ema = 500.0
        self.assertEqual(gate.transport_timeout(), gate.TIMEOUT_CEIL_S)
        gate._latency_ema = 30.0             # 5x = 150s, inside the clamp
        self.assertEqual(gate.transport_timeout(), 150.0)

    def test_a_failed_verify_does_not_poison_the_latency_average(self):
        gate = self._gate()
        gate._latency_ema = 20.0
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible",
                        side_effect=RuntimeError("ollama down")):
            result = gate.verify(np.zeros((64, 64, 3), dtype=np.uint8), _alert())
        self.assertTrue(result.errored)      # fail-visible, as ever
        self.assertEqual(gate._latency_ema, 20.0)


class SlotCourtesyTests(unittest.TestCase):
    def test_the_scanner_never_holds_a_slot_patiently(self):
        from cvti.serving.custom_rules import CustomRuleScanner
        cam = {"id": "front", "source": "x",
               "custom_threats": [{"name": "hoodie", "description": "a hoodie"}]}
        scanner = CustomRuleScanner([cam], sink=None, model="gemma3:4b")
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible",
                        return_value='{"threats": []}') as called:
            scanner._check(cam, np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(called.call_args.kwargs["max_retries"], 0)
        self.assertEqual(called.call_args.kwargs["timeout"], 120.0)

    def test_scene_mapping_keeps_its_patient_default(self):
        """The 1 Sep lesson stays learned: mapping runs once per camera at
        startup and a pilot-slow machine must not fail it. The shared helper's
        DEFAULTS are the mapper's budget — 3 retries, 360s."""
        import inspect
        from cvti.scene.agent_mapper import call_openai_compatible
        sig = inspect.signature(call_openai_compatible)
        self.assertEqual(sig.parameters["max_retries"].default, 3)
        self.assertEqual(sig.parameters["timeout"].default, 360.0)


if __name__ == "__main__":
    unittest.main()
