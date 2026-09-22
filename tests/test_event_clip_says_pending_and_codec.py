"""The clip endpoint says whether evidence is still coming, and what codec it is.

Two silences behind "there's no replay" (pilot, Windows, 22 Sep):

* A critical alert's row exists ~20s before its evidence does (two-tier
  alerting), and the push loop hands the UI that early row. The card must
  hear "pending", not "nothing recorded".
* A clip OpenCV wrote as mp4v is a black box in Chromium with no error
  anyone can read. With the codec in the reply the card can fall back to
  the frames and say why.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from cvti.api.app import create_app
from cvti.serving.alert_sink import _SCHEMA


class EventClipReply(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        site = self.tmp / "site.json"
        site.write_text('{"name": "clip-test", "notify": "console", "cameras": []}')
        self.db = self.tmp / "events.db"
        con = sqlite3.connect(self.db)
        con.executescript(_SCHEMA)
        con.commit()
        con.close()
        self.client = TestClient(create_app(db_path=str(self.db), site_path=str(site)))
        self.client.post("/api/v1/auth/first-owner",
                         json={"username": "ayo", "password": "pw-123456"})
        r = self.client.post("/api/v1/auth/session",
                             json={"username": "ayo", "password": "pw-123456"})
        self.owner = {"Authorization": f"Bearer {r.json()['token']}"}

    def _row(self, ts: float, evidence_dir: str | None) -> str:
        con = sqlite3.connect(self.db)
        cur = con.execute(
            "INSERT INTO events (ts, iso, camera_id, rule, priority, confidence, reason, "
            "evidence_dir) VALUES (?,?,?,?,?,?,?,?)",
            (ts, "x", "gate", "loitering_gate", "medium", 0.9, "r", evidence_dir))
        con.commit()
        con.close()
        return f"evt_{cur.lastrowid}"

    def _clip(self, eid: str) -> dict:
        r = self.client.get(f"/api/v1/events/{eid}/clip", headers=self.owner)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_a_fresh_row_without_evidence_is_pending(self):
        got = self._clip(self._row(time.time() - 3, None))
        self.assertIsNone(got["uri"])
        self.assertEqual(got["frames"], [])
        self.assertTrue(got["pending"])

    def test_an_old_row_without_evidence_is_not_pending(self):
        got = self._clip(self._row(time.time() - 600, None))
        self.assertFalse(got["pending"])

    def test_a_clip_reports_its_codec_and_is_not_pending(self):
        ev = self.tmp / "ev"
        ev.mkdir()
        (ev / "clip.mp4").write_bytes(b"\x00" * 32)
        (ev / "frame_00.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        with mock.patch("cvti.serving.alert_sink._clip_codec", return_value="mp4v"):
            got = self._clip(self._row(time.time() - 3, str(ev)))
        self.assertTrue(got["uri"].startswith("data:video/mp4;base64,"))
        self.assertEqual(got["codec"], "mp4v")
        self.assertEqual(len(got["frames"]), 1)
        self.assertNotIn("pending", got)


if __name__ == "__main__":
    unittest.main()
