"""Alert states and ownership (EP-06-T1).

The definition of done for this epic: two guards on a shift never both respond
to the same alert, neither ignores one, and the incoming shift knows what
happened. These tests pin the first two.
"""
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from _backend_helper import OWNER_PASSWORD, signed_in

from cvti import triage
from cvti.serving.alert_sink import AlertSink
from cvti.triage import ACKNOWLEDGED, NEW, RESOLVED, TriageError


def _db(tmp):
    """An events table shaped like production, via the real sink schema."""
    AlertSink(tmp, save_evidence=False, routing_path=None).close()
    con = sqlite3.connect(Path(tmp) / "events.db")
    con.row_factory = sqlite3.Row
    return con


def _insert(con, *, priority="high", rule="theft", camera="cam1",
            ts=None, state=None, review=None) -> int:
    cur = con.execute(
        "INSERT INTO events (ts, iso, camera_id, rule, priority, state, review) "
        "VALUES (?,?,?,?,?,?,?)",
        (ts if ts is not None else time.time(), "iso", camera, rule, priority,
         state, review))
    con.commit()
    return cur.lastrowid


class StateMachineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.con = _db(self._tmp.name)

    def tearDown(self):
        self.con.close()
        self._tmp.cleanup()

    def test_a_new_alert_can_be_claimed(self):
        event = _insert(self.con)
        result = triage.acknowledge(self.con, event, "ayo")
        self.assertEqual(result["state"], ACKNOWLEDGED)
        self.assertEqual(result["owner"], "ayo")

    def test_two_guards_cannot_both_claim_the_same_alert(self):
        # The epic's definition of done, as a test.
        event = _insert(self.con)
        triage.acknowledge(self.con, event, "ayo")
        with self.assertRaises(TriageError) as ctx:
            triage.acknowledge(self.con, event, "sam")
        self.assertIn("ayo", str(ctx.exception), "the second guard must learn WHO has it")

    def test_reclaiming_your_own_alert_is_a_no_op_not_an_error(self):
        event = _insert(self.con)
        triage.acknowledge(self.con, event, "ayo")
        result = triage.acknowledge(self.con, event, "ayo")
        self.assertTrue(result["already_yours"])

    def test_resolution_captures_outcome_and_note(self):
        event = _insert(self.con)
        triage.acknowledge(self.con, event, "ayo")
        triage.resolve(self.con, event, "ayo", "real", "police called, one detained")
        row = self.con.execute("SELECT * FROM events WHERE id=?", (event,)).fetchone()
        self.assertEqual(row["state"], RESOLVED)
        self.assertEqual(row["outcome"], "real")
        self.assertEqual(row["note"], "police called, one detained")
        self.assertIsNotNone(row["resolved_at"])

    def test_resolving_an_unclaimed_alert_claims_it_in_the_same_breath(self):
        event = _insert(self.con)
        result = triage.resolve(self.con, event, "ayo", "false_alarm")
        self.assertEqual(result["transitions"], [ACKNOWLEDGED, RESOLVED])
        row = self.con.execute("SELECT owner FROM events WHERE id=?", (event,)).fetchone()
        self.assertEqual(row["owner"], "ayo")

    def test_a_resolved_alert_is_finished(self):
        event = _insert(self.con)
        triage.resolve(self.con, event, "ayo", "real")
        with self.assertRaises(TriageError):
            triage.acknowledge(self.con, event, "sam")
        with self.assertRaises(TriageError):
            triage.resolve(self.con, event, "sam", "false_alarm")

    def test_an_unknown_outcome_is_refused(self):
        event = _insert(self.con)
        with self.assertRaises(TriageError):
            triage.resolve(self.con, event, "ayo", "maybe")

    def test_outcomes_project_onto_the_legacy_review_column(self):
        # cvti/feedback reads `review` — resolution outcomes must feed it.
        for outcome, review in (("real", "true"), ("false_alarm", "false")):
            event = _insert(self.con)
            triage.resolve(self.con, event, "ayo", outcome)
            row = self.con.execute("SELECT review FROM events WHERE id=?",
                                   (event,)).fetchone()
            self.assertEqual(row["review"], review)

    def test_rows_from_before_the_state_column_still_work(self):
        # A pre-EP-06 database: state is NULL, review carries the truth.
        old_resolved = _insert(self.con, state=None, review="true")
        old_acked = _insert(self.con, state=None, review="ack")
        old_new = _insert(self.con, state=None, review=None)
        self.assertEqual(triage.state_of(self.con.execute(
            "SELECT * FROM events WHERE id=?", (old_resolved,)).fetchone()), RESOLVED)
        self.assertEqual(triage.state_of(self.con.execute(
            "SELECT * FROM events WHERE id=?", (old_acked,)).fetchone()), ACKNOWLEDGED)
        self.assertEqual(triage.state_of(self.con.execute(
            "SELECT * FROM events WHERE id=?", (old_new,)).fetchone()), NEW)
        with self.assertRaises(TriageError):
            triage.resolve(self.con, old_resolved, "ayo", "real")


class NeedsAttentionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.con = _db(self._tmp.name)

    def tearDown(self):
        self.con.close()
        self._tmp.cleanup()

    def test_one_item_first_highest_priority_oldest_first(self):
        _insert(self.con, priority="medium", ts=100)
        crit_old = _insert(self.con, priority="critical", ts=200)
        _insert(self.con, priority="critical", ts=300)
        out = triage.needs_attention(self.con)
        self.assertEqual(out["now"]["id"], crit_old,
                         "the oldest critical must come first")
        self.assertEqual(out["waiting"], 3)
        self.assertEqual(len(out["then"]), 2)

    def test_claimed_alerts_leave_the_queue_and_appear_as_held(self):
        event = _insert(self.con, priority="high")
        _insert(self.con, priority="high")
        triage.acknowledge(self.con, event, "sam")
        out = triage.needs_attention(self.con)
        self.assertNotEqual(out["now"]["id"], event)
        self.assertEqual(out["waiting"], 1)
        self.assertEqual(out["held"][0]["owner"], "sam",
                         "the other guard must see who is on it")

    def test_low_priority_stays_below_the_threshold(self):
        _insert(self.con, priority="low")
        out = triage.needs_attention(self.con, min_priority="medium")
        self.assertIsNone(out["now"])
        self.assertEqual(out["waiting"], 0)

    def test_an_empty_queue_is_an_answer_not_an_error(self):
        out = triage.needs_attention(self.con)
        self.assertIsNone(out["now"])
        self.assertEqual(out["held"], [])


class BackendIntegrationTest(unittest.TestCase):
    """Through ConsoleBackend: permissions, audit, and the legacy entry point."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "site.json").write_text('{"cameras": []}')
        AlertSink(root, save_evidence=False, routing_path=None).close()   # schema
        self.be = signed_in("owner", site_path=str(root / "site.json"),
                            db_path=str(root / "events.db"), enable_demo=False)
        con = sqlite3.connect(self.be.db_path)
        self.event = _insert(con)
        con.close()

    def tearDown(self):
        self._tmp.cleanup()

    def test_acknowledge_records_the_signed_in_user_and_audits(self):
        result = self.be.acknowledge_alert(self.event)
        self.assertEqual(result["owner"], "owner")
        entry = self.be.audit_entries()[0]
        self.assertEqual(entry["action"], "alert_resolution")
        self.assertEqual(entry["detail"]["transition"], "acknowledged")

    def test_resolve_audits_the_outcome(self):
        self.be.resolve_alert(self.event, "real", "confirmed on review")
        entry = self.be.audit_entries()[0]
        self.assertEqual(entry["detail"]["outcome"], "real")

    def test_a_second_user_is_told_who_owns_it(self):
        self.be.acknowledge_alert(self.event)
        self.be.add_user("sam", OWNER_PASSWORD, role="operator")
        self.be.sign_out()
        self.be.sign_in("sam", OWNER_PASSWORD)
        result = self.be.acknowledge_alert(self.event)
        self.assertFalse(result["ok"])
        self.assertIn("owner", result["error"])

    def test_the_legacy_review_entry_point_flows_through_the_state_machine(self):
        self.be.set_review(self.event, "true")
        con = sqlite3.connect(self.be.db_path)
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM events WHERE id=?", (self.event,)).fetchone()
        con.close()
        self.assertEqual(row["state"], "resolved")
        self.assertEqual(row["outcome"], "real")
        self.assertEqual(row["owner"], "owner")
        self.assertEqual(row["review"], "true")     # feedback loop unchanged

    def test_needs_attention_returns_the_now_shape(self):
        out = self.be.needs_attention()
        self.assertEqual(out["now"]["id"], self.event)
        self.assertIn("frames", out["now"])
        self.assertEqual(out["waiting"], 1)


if __name__ == "__main__":
    unittest.main()
