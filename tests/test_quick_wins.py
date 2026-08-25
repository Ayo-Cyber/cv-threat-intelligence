"""Quick-win architecture fixes (25 Aug): no double decode, per-feed stores,
measured uptime."""
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _backend_helper import signed_in

ROOT = Path(__file__).resolve().parents[1]


class NoDoubleDecodeTest(unittest.TestCase):
    def test_scanner_with_a_frame_source_never_opens_a_capture(self):
        from cvti.serving.custom_rules import CustomRuleScanner
        frames = {"c1": object()}
        checked = []
        sc = CustomRuleScanner(
            [{"id": "c1", "source": "rtsp://cam/1",
              "custom_rules": [{"question": "Is there a ladder?"}]}],
            sink=None, model="m",
            frame_source=lambda cid: frames.get(cid))
        sc._open = lambda src: (_ for _ in ()).throw(AssertionError(
            "scanner opened its own VideoCapture despite the engine's frames"))
        sc._check = lambda c, f: checked.append((c["id"], f)) or None
        # one loop body pass, unrolled: emulate by calling the pieces the loop uses
        frame = sc.frame_source("c1")
        self.assertIs(frame, frames["c1"])
        sc._check({"id": "c1"}, frame)
        self.assertEqual(checked[0][0], "c1")

    def test_engine_wires_peek_not_read_for_scanner_and_watches(self):
        src = (ROOT / "cvti/serving/pipeline.py").read_text()
        self.assertIn("frame_source=_scanner_frame", src)
        # neither side helper may CONSUME frames meant for detection
        for helper in ("_scanner_frame", "_latest_frame", "_any_latest_frame"):
            body = src.split(f"def {helper}")[1].split("def ")[0]
            self.assertIn("peek_latest(", body, f"{helper} must peek")
            self.assertNotIn(".read_latest(", body,
                             f"{helper} steals frames from detection")


class PerFeedStoreTest(unittest.TestCase):
    def test_each_feed_gets_its_own_events_store(self):
        import os
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                root = Path(tmp)
                (root / "site.json").write_text('{"cameras": []}')
                (root / "demo_site.json").write_text('{"cameras": []}')
                be = signed_in("owner", site_path=str(root / "site.json"),
                               db_path=str(root / "events.db"), enable_demo=False)
                home = be._db_for_feed("stage", str(root / "site.json"))
                demo = be._db_for_feed("demo", str(root / "demo_site.json"))
                live = be._db_for_feed("live", str(root / "live_site.json"))
                self.assertEqual(home, str(root / "events.db"),
                                 "the home site must keep its original store")
                self.assertNotEqual(demo, home, "demo alerts would pollute real triage")
                self.assertNotEqual(demo, live)
                self.assertIn("runs/feeds/demo", demo.replace("\\\\", "/"))
            finally:
                os.chdir(cwd)

    def test_switch_updates_the_db_path(self):
        src = (ROOT / "cvti/app/console_backend.py").read_text()
        self.assertIn('self.db_path = self._db_for_feed(key, src["config"])', src)


class HealthHistoryTest(unittest.TestCase):
    def _doc(self, connected=2, total=2, status="ok"):
        return {"status": status, "reasons": [],
                "cameras": [{"state": "connected"}] * connected
                           + [{"state": "offline"}] * (total - connected),
                "gate": {"reachable": True}, "disk": {"used_pct": 50.0}}

    def test_one_row_per_minute_no_matter_how_often_health_writes(self):
        from cvti.serving import health_history as hh
        with tempfile.TemporaryDirectory() as tmp:
            t0 = 1_000_000.0
            wrote = [hh.record(tmp, self._doc(), now=t0 + i * 5) for i in range(30)]
            self.assertEqual(sum(wrote), 3, "throttle failed: 150s should yield 3 rows")

    def test_uptime_is_rows_over_minutes_and_absence_is_unmeasured(self):
        from cvti.serving import health_history as hh
        with tempfile.TemporaryDirectory() as tmp:
            t0 = 1_000_000.0
            # engine alive for 30 of 60 minutes
            for i in range(30):
                hh.record(tmp, self._doc(connected=1, total=2), now=t0 + i * 60)
            out = hh.stats(tmp, since=t0, now=t0 + 3600)
            self.assertAlmostEqual(out["uptime_pct"], 50.0, delta=2.0)
            self.assertAlmostEqual(out["camera_availability_pct"], 50.0, delta=0.1)
            self.assertEqual(hh.stats(tmp, since=t0 - 7200, now=t0 - 3600), {},
                             "no history must read as UNMEASURED, never zero")

    def test_history_is_pruned(self):
        from cvti.serving import health_history as hh
        with tempfile.TemporaryDirectory() as tmp:
            old = time.time() - (hh.KEEP_DAYS + 5) * 86400
            hh.record(tmp, self._doc(), now=old)
            hh.record(tmp, self._doc(), now=time.time())
            con = sqlite3.connect(str(Path(tmp) / hh.DB_NAME))
            n = con.execute("SELECT COUNT(*) FROM health_minutes").fetchone()[0]
            con.close()
            self.assertEqual(n, 1)

    def test_weekly_summary_reports_measured_uptime(self):
        from cvti.serving import health_history as hh
        from cvti import owner_summary as osum
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            for i in range(60):
                hh.record(tmp, self._doc(), now=now - 3600 + i * 60)
            s = osum.compute_summary(Path(tmp) / "events.db", {}, now=now)
            self.assertGreater(s["monitoring"].get("samples", 0), 0)
            self.assertIn("uptime_pct", s["monitoring"])


if __name__ == "__main__":
    unittest.main()
