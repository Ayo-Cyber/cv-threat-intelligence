from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.app.console_backend import ConsoleBackend

from _backend_helper import signed_in


def _seed_db(path: str):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, iso TEXT,
        camera_id TEXT, rule TEXT, priority TEXT, confidence REAL, reason TEXT, track_id INTEGER,
        zone TEXT, object_label TEXT, evidence_dir TEXT, review TEXT, reviewed_at TEXT)""")
    con.execute("INSERT INTO events (ts,iso,camera_id,rule,priority,confidence,reason,evidence_dir) "
                "VALUES (1.0,'2026-07-21T08:00','aisle_1','shoplifting','high',0.88,'concealment','')")
    con.commit()
    con.close()


class ConsoleBackendTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.site = str(Path(self._tmp.name) / "site.json")
        self.db = str(Path(self._tmp.name) / "events.db")
        self.be = signed_in(site_path=self.site, db_path=self.db, enable_demo=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cameras_roundtrip(self):
        self.assertEqual(self.be.list_cameras(), [])
        self.be.add_camera({"id": "front", "source": "rtsp://a/1", "concealment": True})
        self.assertEqual(self.be.counts()["cameras"], 1)
        self.be.remove_camera("front")
        self.assertEqual(self.be.list_cameras(), [])

    def test_events_and_review(self):
        _seed_db(self.db)
        evs = self.be.list_events()
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]["review"], "new")          # NULL surfaces as 'new'
        self.assertEqual(evs[0]["rule"], "shoplifting")
        # review it
        self.be.set_review(evs[0]["id"], "true")
        self.assertEqual(self.be.list_events()[0]["review"], "true")
        self.assertEqual(self.be.counts()["pending_alerts"], 0)   # labeled -> not pending

    def test_events_missing_db_is_empty(self):
        self.assertEqual(self.be.list_events(), [])              # no db yet -> no crash
        self.assertEqual(self.be.counts()["pending_alerts"], 0)

    def test_bad_review_rejected(self):
        _seed_db(self.db)
        with self.assertRaises(ValueError):
            self.be.set_review(1, "maybe")

    def test_site_meta_and_setup_state(self):
        # fresh site -> defaults, not configured
        s0 = self.be.setup_state()
        self.assertFalse(s0["configured"])
        self.assertEqual(s0["cameras"], 0)
        self.assertIn("gate", s0)                       # gate status always present
        # name it + add a camera + finish setup
        self.be.set_site(name="Front Store", notify="whatsapp")
        self.be.add_camera({"id": "entry", "source": "rtsp://a/1"})
        self.be.mark_configured()
        s1 = self.be.setup_state()
        self.assertTrue(s1["configured"])
        self.assertEqual(s1["site_name"], "Front Store")
        self.assertEqual(s1["notify"], "whatsapp")
        self.assertEqual(s1["cameras"], 1)
        saved = json.loads(Path(self.site).read_text())
        self.assertNotIn("scene_context_policy", saved,
                         "the wizard must never stamp a strict policy (1 Sep)")

    def test_detect_subnet_shape(self):
        r = self.be.detect_subnet()                      # network-dependent: None or a /24
        self.assertIn("cidr", r)
        if r["cidr"] is not None:
            self.assertTrue(r["cidr"].endswith("/24"))

    def test_gate_status_shape(self):
        g = self.be.gate_status()                       # hits real localhost; shape must hold
        for k in ("ollama", "model_present", "model", "mode"):
            self.assertIn(k, g)


if __name__ == "__main__":
    unittest.main()
