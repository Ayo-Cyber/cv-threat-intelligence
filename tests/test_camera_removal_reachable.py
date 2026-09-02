"""Removing a camera must be reachable from the Cameras screen (2 Sep UI pass).

The backend has had removeCamera since onboarding shipped, but the only UI
that called it was the first-run wizard — on the everyday Cameras screen a
typo'd RTSP address was permanent without hand-editing site.json. These pins
keep the affordance wired, and pin the promise its confirm() makes: removal
never touches recorded alerts or evidence.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _backend_helper import signed_in
from cvti.app.console_backend import ConsoleBackend


class RemovalReachableFromCamerasScreen(unittest.TestCase):
    def setUp(self):
        self.html = (ROOT / "cvti" / "app" / "web" / "index.html").read_text()

    def test_camera_rows_carry_a_remove_button(self):
        # The chip lives in renderCameras' row template, not just the wizard.
        rows_start = self.html.index("function renderCameras()")
        rows = self.html[rows_start:rows_start + 4000]
        self.assertIn("uiRemoveCamera", rows)

    def test_remove_asks_before_acting_and_calls_the_backend(self):
        fn_start = self.html.index("function uiRemoveCamera")
        fn = self.html[fn_start:fn_start + 600]
        self.assertIn("confirm(", fn)                    # destructive: always confirmed
        self.assertIn('call("removeCamera"', fn)


class RemovalKeepsEvidence(unittest.TestCase):
    def test_removing_a_camera_leaves_its_events_alone(self):
        """The confirm() promises 'alerts and evidence are kept' — hold it to
        that: remove a camera and its recorded events must still be there."""
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            site = str(Path(tmp) / "site.json")
            db = str(Path(tmp) / "events.db")
            con = sqlite3.connect(db)
            con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts REAL, "
                        "camera_id TEXT, rule TEXT, priority TEXT, review TEXT)")
            con.execute("INSERT INTO events (ts, camera_id, rule, priority) "
                        "VALUES (1, 'front', 'theft_attempt', 'high')")
            con.commit()
            con.close()
            backend = signed_in(site_path=site, db_path=db, enable_demo=False)
            backend.add_camera({"id": "front", "source": "rtsp://a/1"})
            remaining = backend.remove_camera("front")
            self.assertEqual(remaining, [])
            con = sqlite3.connect(db)
            kept = con.execute("SELECT COUNT(*) FROM events "
                               "WHERE camera_id='front'").fetchone()[0]
            con.close()
            self.assertEqual(kept, 1)


if __name__ == "__main__":
    unittest.main()
