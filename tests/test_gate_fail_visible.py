"""The gate must not turn "I could not decide" into "this is safe" (EP-01-T4).

`_parse_response` used to return confirmed=False on any exception, which is the
same value it returns when TrueSight looks at a frame and rejects it. A fire
during an Ollama restart was therefore indistinguishable from a fire the model
examined and dismissed — and it was dropped, silently, with nothing on screen.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cvti.contracts import CandidateAlert, VerificationResult
from cvti.serving.alert_queue import QueuedAlert
from cvti.serving.alert_sink import AlertSink
from cvti.verification import gate as gate_mod
from cvti.verification.gate import UNVERIFIED_REASON, VerificationGate


def _alert(rule="baseline_fire_smoke", priority="critical"):
    """What the gate verifies."""
    return CandidateAlert(
        rule_name=rule, priority=priority, detector="fire_smoke",
        title="FIRE: visible flame", person_id=None, object_label=None,
        timestamp=0.0)


def _queued(rule="baseline_fire_smoke", priority="critical"):
    """What the sink persists."""
    return QueuedAlert(
        camera_id="aisle_1", rule_name=rule, priority=priority,
        title="FIRE: visible flame", timestamp=0.0, track_id=1, zone=None,
        object_label=None, payload={"frames": [], "enqueued_at": None})


def _frame():
    return np.zeros((32, 32, 3), dtype=np.uint8)


class ErrorIsNotRejectionTest(unittest.TestCase):
    def test_a_rejection_is_not_flagged_as_an_error(self):
        result = gate_mod._parse_response(
            '{"confirmed": false, "confidence": 0.9, "reason": "empty aisle"}', "high")
        self.assertFalse(result.confirmed)
        self.assertFalse(result.errored)     # the model looked and said no

    def test_malformed_json_is_an_error_not_a_rejection(self):
        result = gate_mod._parse_response("{confirmed: yes, sort of", "high")
        self.assertTrue(result.errored)

    def test_truncated_json_is_an_error(self):
        result = gate_mod._parse_response('{"confirmed": true, "confid', "high")
        self.assertTrue(result.errored)

    def test_empty_response_is_an_error(self):
        self.assertTrue(gate_mod._parse_response("", "high").errored)


class FailVisibleTest(unittest.TestCase):
    """A gate that cannot decide must interrupt a human, not discard the alert."""

    def _gate(self, **kw):
        return VerificationGate(provider="ollama", fail_visible=True, **kw)

    def test_connection_refused_surfaces_the_alert_flagged(self):
        g = self._gate()
        with mock.patch.object(gate_mod, "_call_ollama",
                               side_effect=ConnectionRefusedError("connection refused")):
            result = g.verify(_frame(), _alert())
        self.assertTrue(result.errored)
        self.assertTrue(result.confirmed, "a fire was dropped while the gate was down")
        self.assertIn("UNVERIFIED", result.reason)
        self.assertIn("ConnectionRefusedError", result.error)

    def test_a_timeout_surfaces_the_alert(self):
        g = self._gate()
        with mock.patch.object(gate_mod, "_call_ollama", side_effect=TimeoutError("timed out")):
            result = g.verify(_frame(), _alert())
        self.assertTrue(result.errored)
        self.assertTrue(result.confirmed)

    def test_malformed_response_surfaces_the_alert(self):
        g = self._gate()
        with mock.patch.object(gate_mod, "_call_ollama", return_value="not json at all"):
            result = g.verify(_frame(), _alert())
        self.assertTrue(result.errored)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.reason, UNVERIFIED_REASON)

    def test_fail_silent_is_available_but_not_the_default(self):
        self.assertTrue(VerificationGate(provider="ollama").fail_visible)
        g = VerificationGate(provider="ollama", fail_visible=False)
        with mock.patch.object(gate_mod, "_call_ollama", side_effect=OSError("down")):
            result = g.verify(_frame(), _alert())
        self.assertTrue(result.errored)          # still recorded as an error
        self.assertFalse(result.confirmed)       # but not surfaced

    def test_an_unverified_alert_is_not_downgraded_by_the_confidence_floor(self):
        # It has no confidence to fall below; downgrading would re-hide exactly
        # what fail-visible just surfaced.
        g = VerificationGate(provider="ollama", fail_visible=True, min_confidence=0.9)
        with mock.patch.object(gate_mod, "_call_ollama", return_value="garbage"):
            result = g.verify(_frame(), _alert())
        self.assertTrue(result.confirmed)

    def test_a_genuine_low_confidence_pass_is_still_downgraded(self):
        g = VerificationGate(provider="ollama", min_confidence=0.9)
        with mock.patch.object(gate_mod, "_call_ollama",
                               return_value='{"confirmed": true, "confidence": 0.4, "reason": "maybe"}'):
            result = g.verify(_frame(), _alert())
        self.assertFalse(result.confirmed)
        self.assertFalse(result.errored)

    def test_an_unknown_provider_stays_loud(self):
        # A config bug must not become an endless stream of unverified alerts.
        g = VerificationGate(provider="nonsense-provider")
        with self.assertRaises(gate_mod.UnsupportedProvider):
            g.verify(_frame(), _alert())


class UnverifiedReachesTheOperatorTest(unittest.TestCase):
    def test_unverified_alerts_are_persisted_and_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = AlertSink(tmp, save_evidence=False, routing_path=None)
            sink.handle(_queued(), VerificationResult(
                confirmed=True, confidence=0.0, reason=UNVERIFIED_REASON,
                alert_priority="critical", timestamp="t", error="transport: OSError: down"))
            row = sink._db.execute(
                "SELECT reason, unverified, gate_error FROM events").fetchone()
            sink.close()
            self.assertIn("UNVERIFIED", row[0])
            self.assertEqual(row[1], 1)
            self.assertIn("OSError", row[2])

    def test_a_normal_confirmation_is_not_flagged_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = AlertSink(tmp, save_evidence=False, routing_path=None)
            sink.handle(_queued(), VerificationResult(
                confirmed=True, confidence=0.88, reason="visible flame on the shelf",
                alert_priority="critical", timestamp="t"))
            row = sink._db.execute("SELECT unverified, gate_error FROM events").fetchone()
            sink.close()
            self.assertEqual(row[0], 0)
            self.assertIsNone(row[1])

    def test_a_database_from_before_this_change_still_opens(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "events.db"
            con = sqlite3.connect(db)
            con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "ts REAL, iso TEXT, camera_id TEXT, rule TEXT, priority TEXT, "
                        "confidence REAL, reason TEXT, track_id INTEGER, zone TEXT, "
                        "object_label TEXT, evidence_dir TEXT, review TEXT, reviewed_at TEXT)")
            con.commit()
            con.close()
            sink = AlertSink(tmp, save_evidence=False, routing_path=None)   # migrates
            cols = {r[1] for r in sink._db.execute("PRAGMA table_info(events)")}
            sink.close()
            self.assertIn("unverified", cols)
            self.assertIn("gate_error", cols)


if __name__ == "__main__":
    unittest.main()
