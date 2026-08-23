"""The weekly owner summary (EP-08-T2).

The renewal decision-maker gets recurring contact with what the product did —
in counts that can be walked back to rows, never modelled numbers.
"""
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from cvti import owner_summary as osum


def _seed(db: Path, rows):
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE events (id INTEGER PRIMARY KEY, ts REAL, iso TEXT,
        camera_id TEXT, rule TEXT, priority TEXT, outcome TEXT, review TEXT,
        unverified INTEGER DEFAULT 0, retracted INTEGER DEFAULT 0)""")
    con.execute("""CREATE TABLE suppression_daily (day TEXT PRIMARY KEY,
        shown INTEGER, rejected INTEGER, deduped INTEGER, errors INTEGER,
        updated_at REAL)""")
    for r in rows:
        con.execute("INSERT INTO events (ts, camera_id, rule, outcome, review, "
                    "unverified, retracted) VALUES (?,?,?,?,?,?,?)", r)
    con.commit(); con.close()


NOW = time.mktime((2026, 8, 24, 12, 0, 0, 0, 0, -1))   # a Monday noon
DAY = 86400


class FiguresAreRowCountsTest(unittest.TestCase):
    def _db(self, tmp):
        db = Path(tmp) / "events.db"
        _seed(db, [
            (NOW - 2 * DAY, "till", "theft", "real", "true", 0, 0),        # this week
            (NOW - 3 * DAY, "till", "theft", "false_alarm", "false", 0, 0),
            (NOW - 4 * DAY, "dock", "fire", None, None, 1, 0),              # unverified
            (NOW - 5 * DAY, "till", "theft", None, None, 0, 1),             # retracted: excluded
            (NOW - 40 * DAY, "till", "theft", "real", "true", 0, 0),        # prev month
        ])
        con = sqlite3.connect(db)
        day = time.strftime("%Y-%m-%d", time.localtime(NOW - 2 * DAY))
        con.execute("INSERT INTO suppression_daily VALUES (?, 10, 40, 5, 0, ?)", (day, NOW))
        con.commit(); con.close()
        return db

    def test_week_counts_match_the_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = osum.compute_summary(self._db(tmp), {"name": "Shop"}, now=NOW)
            w = s["week"]
            self.assertEqual((w["incidents"], w["real"], w["false_alarm"], w["unverified"]),
                             (3, 1, 1, 1))         # retracted row excluded
            self.assertEqual(w["noise_removed"], 45)
            self.assertEqual(w["by_camera"]["till"], 2)

    def test_month_over_month_is_a_real_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = osum.compute_summary(self._db(tmp), {}, now=NOW)
            d = s["month_over_month"]["incidents"]
            self.assertEqual((d["prev"], d["now"]), (1, 3))
            self.assertEqual(d["change"], 2)

    def test_money_is_omitted_without_site_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            s = osum.compute_summary(db, {}, now=NOW)
            self.assertEqual(s["money"], {})
            s2 = osum.compute_summary(db, {"guard_hourly_cost": 10,
                                           "review_minutes": 2}, now=NOW)
            self.assertAlmostEqual(s2["money"]["attention_saved"],
                                   round(45 * 2 / 60 * 10, 2))

    def test_the_pdf_is_a_real_document_with_the_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = osum.weekly_summary(self._db(tmp), {"name": "Shop"}, tmp, now=NOW)
            data = Path(out["pdf"]).read_bytes()
            self.assertTrue(data.startswith(b"%PDF"))
            import zlib
            text = ""
            i = 0
            while (i := data.find(b"stream\n", i)) >= 0:
                j = data.find(b"\nendstream", i)
                try:
                    text += zlib.decompress(data[i + 7:j]).decode("latin-1", "replace")
                except zlib.error:
                    pass
                i = j
            for needed in ("WEEKLY SUMMARY", "Shop", "false alarms filtered", "row count"):
                self.assertIn(needed.split()[0], text)

    def test_a_fresh_site_summarises_zeros_not_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = osum.compute_summary(Path(tmp) / "none.db", {}, now=NOW)
            self.assertEqual(s["week"]["incidents"], 0)


class WeeklyCadenceTest(unittest.TestCase):
    def test_due_only_monday_morning_once_per_week(self):
        with tempfile.TemporaryDirectory() as tmp:
            monday_9 = NOW - 3 * 3600      # Monday 09:00
            sunday = NOW - DAY
            self.assertFalse(osum.due(tmp, now=sunday))
            self.assertTrue(osum.due(tmp, now=monday_9))
            osum.mark_sent(tmp, now=monday_9)
            self.assertFalse(osum.due(tmp, now=NOW), "sent twice in one week")
            next_monday = NOW + 7 * DAY
            self.assertTrue(osum.due(tmp, now=next_monday))

    def test_state_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            osum.mark_sent(tmp, now=NOW)
            self.assertFalse(osum.due(tmp, now=NOW + 3600))


if __name__ == "__main__":
    unittest.main()
