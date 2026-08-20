"""Prompt + model versioning (EP-07-T2).

Two drifts this pins shut: an event that cannot say which prompt wording
judged it, and a SENSITIVITY_MEASURED constant that no longer matches the
archived measurement it claims to report.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cvti.contracts import CandidateAlert, VerificationResult
from cvti.verification.gate import VerificationGate

ROOT = Path(__file__).resolve().parents[1]


def _candidate():
    return CandidateAlert(rule_name="shoplifting", priority="high",
                          detector="concealment", title="t", person_id=1,
                          object_label=None, timestamp=0.0)


class PromptVersionStampTest(unittest.TestCase):
    def test_every_verdict_carries_the_prompt_fingerprint(self):
        gate = VerificationGate(provider="mock")
        out = gate.verify(np.zeros((8, 8, 3), np.uint8), _candidate(),
                          {"environment_type": "retail"})
        self.assertTrue(out.prompt_version, "verdict left the gate unversioned")
        self.assertEqual(len(out.prompt_version), 12)

    def test_a_transport_failure_is_versioned_too(self):
        # The UNVERIFIED verdict also records which wording WOULD have judged
        # it — the operator's review happens under that prompt regime.
        gate = VerificationGate(provider="local",
                                base_url="http://127.0.0.1:59999/v1")
        out = gate.verify(np.zeros((8, 8, 3), np.uint8), _candidate(),
                          {"environment_type": "retail"})
        self.assertTrue(out.errored)
        self.assertTrue(out.prompt_version)

    def test_the_version_matches_the_fingerprint_of_the_running_code(self):
        from cvti.eval.prompt_fingerprint import fingerprint
        gate = VerificationGate(provider="mock")
        self.assertEqual(gate.prompt_version, fingerprint()[:12])

    def test_the_field_defaults_empty_for_synthetic_results(self):
        r = VerificationResult(True, 0.9, "r", "high", "t")
        self.assertEqual(r.prompt_version, "")


class EventsCarryTheVersionTest(unittest.TestCase):
    def test_the_sink_persists_prompt_version_with_the_event(self):
        import sqlite3
        from cvti.serving.alert_queue import QueuedAlert
        from cvti.serving.alert_sink import AlertSink
        with tempfile.TemporaryDirectory() as tmp:
            sink = AlertSink(tmp, save_evidence=False, routing_path=None)
            alert = QueuedAlert(camera_id="c", rule_name="theft", priority="high",
                                title="t", timestamp=0.0,
                                payload={"frames": [], "enqueued_at": None})
            result = VerificationResult(True, 0.9, "r", "high", "t",
                                        prompt_version="abc123def456")
            sink.handle(alert, result)
            con = sqlite3.connect(Path(tmp) / "events.db")
            got = con.execute("SELECT prompt_version FROM events").fetchone()[0]
            con.close()
            sink.close()
            self.assertEqual(got, "abc123def456")


@unittest.skipUnless((ROOT / "runs/eval/v2-tightened/metrics.json").exists(),
                     "archived eval runs not on this machine")
class SensitivityIsGeneratedTest(unittest.TestCase):
    def test_the_committed_constants_equal_what_the_archives_produce(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools/make_sensitivity.py"),
                            "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0,
                         f"SENSITIVITY_MEASURED drifted from the archives:\n{r.stdout}")


if __name__ == "__main__":
    unittest.main()
