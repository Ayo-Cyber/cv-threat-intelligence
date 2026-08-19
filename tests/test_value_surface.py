"""The Value screen: suppression counts reframed as what a buyer decides on.

The product's central claim — "raw detectors would have shown you 201 alerts,
we showed you 28" — has to survive an engine restart, so the counts live in the
DB rather than in the gate pool's process memory.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cvti.app.console_backend import ConsoleBackend
from cvti.serving.alert_sink import AlertSink


class SuppressionLedgerTest(unittest.TestCase):
    def _sink(self, tmp):
        return AlertSink(tmp, save_evidence=False, routing_path=None)

    def test_deltas_accumulate_within_a_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = self._sink(tmp)
            sink.record_suppression(shown=2, rejected=5, day="2026-08-19")
            sink.record_suppression(shown=1, rejected=3, deduped=4, day="2026-08-19")
            row = sink._db.execute(
                "SELECT shown, rejected, deduped FROM suppression_daily WHERE day=?",
                ("2026-08-19",)).fetchone()
            self.assertEqual(tuple(row), (3, 8, 4))
            sink.close()

    def test_days_are_kept_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = self._sink(tmp)
            sink.record_suppression(shown=1, day="2026-08-18")
            sink.record_suppression(shown=7, day="2026-08-19")
            days = dict(sink._db.execute("SELECT day, shown FROM suppression_daily"))
            self.assertEqual(days, {"2026-08-18": 1, "2026-08-19": 7})
            sink.close()

    def test_empty_delta_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = self._sink(tmp)
            sink.record_suppression()
            self.assertEqual(
                sink._db.execute("SELECT COUNT(*) FROM suppression_daily").fetchone()[0], 0)
            sink.close()


class ValueSummaryTest(unittest.TestCase):
    def _backend_with(self, tmp, shown, rejected, deduped, errors=0):
        db = Path(tmp) / "events.db"
        sink = AlertSink(tmp, save_evidence=False, routing_path=None)
        sink.record_suppression(shown=shown, rejected=rejected, deduped=deduped, errors=errors)
        sink.close()
        site = Path(tmp) / "site.json"
        site.write_text('{"cameras": []}')
        self.assertTrue(db.exists())
        return ConsoleBackend(site_path=str(site), db_path=str(db), enable_demo=False)

    def test_raw_is_shown_plus_everything_prevented(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = self._backend_with(tmp, shown=28, rejected=150, deduped=23).value_summary(30)
            self.assertEqual(v["raw_alerts"], 201)
            self.assertEqual(v["shown"], 28)
            # Rejected and deduped stay apart: only the first are false alarms.
            self.assertEqual(v["false_alarms_prevented"], 150)
            self.assertEqual(v["duplicates_collapsed"], 23)
            self.assertEqual(v["noise_removed"], 173)
            self.assertAlmostEqual(v["suppression_pct"], 173 / 201, places=4)

    def test_attention_hours_follow_the_sites_own_review_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            be = self._backend_with(tmp, shown=10, rejected=110, deduped=10)
            be.set_value_inputs(review_minutes=3.0)
            self.assertEqual(be.value_summary(30)["attention_hours_saved"], 6.0)  # 120 * 3 / 60

    def test_money_is_hidden_until_the_site_supplies_a_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            be = self._backend_with(tmp, shown=5, rejected=55, deduped=0)
            self.assertEqual(be.value_summary(30)["money"], {})
            be.set_value_inputs(guard_hourly_cost=20.0, review_minutes=2.0)
            money = be.value_summary(30)["money"]
            # 55 prevented * 2 min = 1.8333 h; * £20 = £36.67
            self.assertAlmostEqual(money["attention_saved"], 36.67, places=1)
            self.assertNotIn("incidents_value", money)   # incident_value still unset

    def test_gate_errors_are_reported_not_absorbed(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = self._backend_with(tmp, shown=1, rejected=1, deduped=0, errors=4).value_summary(30)
            self.assertEqual(v["gate_errors"], 4)

    def test_no_history_is_stated_rather_than_shown_as_zero_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site.json"
            site.write_text('{"cameras": []}')
            be = ConsoleBackend(site_path=str(site), db_path=str(Path(tmp) / "events.db"),
                                enable_demo=False)
            v = be.value_summary(30)
            self.assertFalse(v["has_data"])
            self.assertIsNone(v["suppression_pct"])

    def test_incidents_come_from_confirmed_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            be = self._backend_with(tmp, shown=3, rejected=9, deduped=0)
            con = sqlite3.connect(be.db_path)
            import time as _t
            for i, review in enumerate((None, "true", "false")):
                con.execute("INSERT INTO events (ts, iso, camera_id, rule, review) "
                            "VALUES (?,?,?,?,?)", (_t.time(), "now", "cam1", "theft", review))
            con.commit()
            con.close()
            v = be.value_summary(30)
            self.assertEqual(v["incidents"], 3)
            self.assertEqual(v["operator_labels"], {"true": 1, "false": 1})


if __name__ == "__main__":
    unittest.main()
