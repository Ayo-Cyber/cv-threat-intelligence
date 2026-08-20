"""Two-tier critical alerting (EP-06-T4).

Detection is ~163ms per batch; the ~20s latency is entirely verification. For
theft that trade is right. For a weapon or a fire it is not — so criticals are
shown provisionally the moment the detector fires, and the verdict updates the
same row in place: confirmation, or an explicit retraction. Never silence.
"""
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from cvti.contracts import VerificationResult
from cvti.serving.alert_queue import AlertQueue, QueuedAlert
from cvti.serving.alert_sink import AlertSink


class _Notifier:
    def __init__(self):
        self.sent = []

    def notify(self, event):
        self.sent.append(event)


def _alert(priority="critical", rule="weapon_sighting", enqueued_at=None):
    return QueuedAlert(camera_id="till", rule_name=rule, priority=priority,
                       title="WEAPON", timestamp=time.time(),
                       payload={"frames": [], "candidate": None,
                                "enqueued_at": enqueued_at if enqueued_at is not None
                                else time.time()})


class CriticalsJumpTheQueueTest(unittest.TestCase):
    def test_a_late_critical_drains_before_earlier_lower_tiers(self):
        # The plan says the architecture already contains the answer — pin it,
        # so a refactor of the queue cannot silently lose it.
        q = AlertQueue()
        for i, prio in enumerate(["medium", "high", "low"]):
            q.add(QueuedAlert(camera_id=f"c{i}", rule_name=f"r{i}", priority=prio,
                              title="t", timestamp=float(i)))
        q.add(QueuedAlert(camera_id="c9", rule_name="weapon", priority="critical",
                          title="t", timestamp=99.0))
        order = [a.priority for a in q.drain(max_per_drain=10)]
        self.assertEqual(order[0], "critical",
                         "a critical queued last must still verify first")


class ProvisionalTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.notifier = _Notifier()
        self.sink = AlertSink(self._tmp.name, save_evidence=False,
                              routing_path=None, notifier=self.notifier)

    def tearDown(self):
        self.sink.close()
        self._tmp.cleanup()

    def _row(self, event_id):
        con = sqlite3.connect(Path(self._tmp.name) / "events.db")
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        con.close()
        return row

    def test_provisional_is_persisted_and_notified_in_under_a_second(self):
        started = time.monotonic()
        event_id = self.sink.provisional(_alert())
        elapsed = time.monotonic() - started
        self.assertIsNotNone(event_id)
        self.assertLess(elapsed, 1.0, f"provisional took {elapsed:.2f}s")
        row = self._row(event_id)
        self.assertEqual(row["provisional"], 1)
        self.assertEqual(row["unverified"], 1)       # nothing has checked it yet
        self.assertEqual(len(self.notifier.sent), 1)
        self.assertIn("PROVISIONAL", self.notifier.sent[0]["reason"])
        self.assertIn("unconfirmed", self.notifier.sent[0]["reason"])

    def test_confirmation_updates_the_same_row_in_place(self):
        alert = _alert()
        event_id = self.sink.provisional(alert)
        alert.payload["provisional_event_id"] = event_id
        self.sink.handle(alert, VerificationResult(
            True, 0.91, "a person holding a knife at the till", "critical", "t"))
        row = self._row(event_id)
        self.assertEqual(row["provisional"], 0)
        self.assertEqual(row["retracted"], 0)
        self.assertEqual(row["unverified"], 0)
        self.assertAlmostEqual(row["confidence"], 0.91, places=2)
        self.assertIn("CONFIRMED", self.notifier.sent[-1]["reason"])
        # one row, not two: the update was in place
        con = sqlite3.connect(Path(self._tmp.name) / "events.db")
        self.assertEqual(con.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)
        con.close()

    def test_rejection_is_an_explicit_visible_retraction(self):
        alert = _alert()
        event_id = self.sink.provisional(alert)
        alert.payload["provisional_event_id"] = event_id
        self.sink.handle(alert, VerificationResult(
            False, 0.88, "an umbrella, not a weapon", "critical", "t"))
        row = self._row(event_id)
        self.assertEqual(row["retracted"], 1)
        self.assertIn("RETRACTED", row["reason"])
        self.assertIn("umbrella", row["reason"])     # the why survives
        self.assertIn("RETRACTED", self.notifier.sent[-1]["reason"])
        # The row is kept: a quiet deletion would leave "why did my phone buzz?"
        # unanswerable.

    def test_a_retraction_is_not_an_operator_label(self):
        # TrueSight's own rejection must NOT feed the training data as if a
        # human had reviewed it.
        alert = _alert()
        event_id = self.sink.provisional(alert)
        alert.payload["provisional_event_id"] = event_id
        self.sink.handle(alert, VerificationResult(False, 0.9, "no", "critical", "t"))
        self.assertIsNone(self._row(event_id)["review"])

    def test_an_unverified_verdict_keeps_the_alert_flagged_not_retracted(self):
        alert = _alert()
        event_id = self.sink.provisional(alert)
        alert.payload["provisional_event_id"] = event_id
        self.sink.handle(alert, VerificationResult(
            True, 0.0, "UNVERIFIED", "critical", "t", error="transport: down"))
        row = self._row(event_id)
        self.assertEqual(row["retracted"], 0)
        self.assertEqual(row["unverified"], 1)
        self.assertEqual(row["provisional"], 0)

    def test_retracted_alerts_leave_the_needs_attention_queue(self):
        from cvti import triage
        alert = _alert()
        event_id = self.sink.provisional(alert)
        con = sqlite3.connect(Path(self._tmp.name) / "events.db")
        con.row_factory = sqlite3.Row
        self.assertEqual(triage.needs_attention(con, min_priority="low")["now"]["id"],
                         event_id, "a live provisional must be in the queue")
        alert.payload["provisional_event_id"] = event_id
        self.sink.handle(alert, VerificationResult(False, 0.9, "no", "critical", "t"))
        out = triage.needs_attention(con, min_priority="low")
        con.close()
        self.assertIsNone(out["now"], "a retracted alert stayed in the queue")
        self.assertEqual(out["waiting"], 0)

    def test_latency_is_measured_per_tier_and_for_the_fast_path(self):
        alert = _alert(enqueued_at=time.time() - 0.4)
        event_id = self.sink.provisional(alert)
        alert.payload["provisional_event_id"] = event_id
        self.sink.handle(alert, VerificationResult(True, 0.9, "yes", "critical", "t"))
        stats = self.sink.latency_stats()
        self.assertGreaterEqual(stats["provisional"]["n"], 1)
        self.assertLess(stats["provisional"]["median_s"], 1.0)
        self.assertGreaterEqual(stats["critical"]["n"], 1)


class FastPathWiringTest(unittest.TestCase):
    """Only criticals take the fast path — the precision story holds elsewhere."""

    def test_pipeline_fast_path_gates_on_priority(self):
        import inspect

        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.run_site)
        self.assertIn('alert.priority != "critical"', src)
        self.assertIn("provisional_event_id", src)


if __name__ == "__main__":
    unittest.main()
