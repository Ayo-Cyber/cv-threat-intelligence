"""The engine API is the contract Demi's Electron app builds against (6 Sep).

Pins the read-only surface of contract v0.2: auth gates every route, the data
endpoints return the agreed shapes from a real events.db + gate_health.json,
credentials never appear in a camera source, the live WebSocket hydrates on
connect and pushes new alerts, and the mock server answers the same shapes so
the frontend can build with no engine.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from cvti.api.app import create_app
from cvti.api.mock import create_mock_app

PREFIX = "/api/v1"


def _make_site(tmp: Path) -> tuple[str, str]:
    """A site config + an events.db with one account and two events."""
    db = tmp / "events.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, iso TEXT, camera_id TEXT, rule TEXT, priority TEXT,
        confidence REAL, reason TEXT, track_id INTEGER, zone TEXT,
        object_label TEXT, evidence_dir TEXT, review TEXT, reviewed_at TEXT)""")
    con.execute("INSERT INTO events (ts,iso,camera_id,rule,priority,confidence,reason,evidence_dir) "
                "VALUES (1,'2026-09-06T10:00:00Z','Dublin Street','custom:hoodie','high',0.95,'a hoodie','/e/1')")
    con.execute("INSERT INTO events (ts,iso,camera_id,rule,priority,confidence,reason,review) "
                "VALUES (2,'2026-09-06T10:01:00Z','Loading Bay','video_theft','critical',0.9,'theft','false')")
    con.commit(); con.close()

    from cvti.security.accounts import AccountStore
    from cvti.security import permissions as perms
    store = AccountStore(tmp / "auth.db")
    store.create_user("ayo", "Argus-Fresh-2026", role=perms.OWNER)

    site = tmp / "site.json"
    site.write_text(json.dumps({"cameras": [
        {"id": "Dublin Street", "source": "rtsp://user:secret@10.0.0.9/stream1", "config": "x"},
    ]}))
    (tmp / "gate_health.json").write_text(json.dumps({
        "status": "ok", "generated_at": 9e12, "version": "test",
        "engine": {"phase": "monitoring"},
        "cameras": [{"camera_id": "Dublin Street", "state": "connected",
                     "last_frame_age_s": 0.2, "ingest": {"width": 1280, "sampling_fps": 12}}],
    }))
    return str(db), str(site)


class RealApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        db, site = _make_site(tmp)
        self.client = TestClient(create_app(db_path=db, site_path=site))
        self.addCleanup(self._tmp.cleanup)

    def _token(self) -> str:
        r = self.client.post(f"{PREFIX}/auth/session",
                             json={"username": "ayo", "password": "Argus-Fresh-2026"})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["token"]

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self._token()}"}

    def test_unauthenticated_is_refused(self):
        self.assertEqual(self.client.get(f"{PREFIX}/events").status_code, 401)
        self.assertEqual(self.client.get(f"{PREFIX}/system/health").status_code, 401)

    def test_index_is_self_describing_and_needs_no_auth(self):
        for path in ("/", PREFIX):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, path)
            body = r.json()
            self.assertEqual(body["name"], "Argus Engine API")
            self.assertEqual(body["docs"], "/docs")
            self.assertFalse(body["mock"])
            self.assertIn(f"{PREFIX}/system/health", body["endpoints"])

    def test_sign_in_bad_password(self):
        r = self.client.post(f"{PREFIX}/auth/session",
                             json={"username": "ayo", "password": "wrong"})
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["error"]["code"], "unauthorized")

    def test_sign_in_returns_token_and_role(self):
        r = self.client.post(f"{PREFIX}/auth/session",
                             json={"username": "ayo", "password": "Argus-Fresh-2026"})
        body = r.json()
        self.assertIn("token", body)
        self.assertEqual(body["user"]["role"], "owner")
        self.assertIn("view_alerts", body["user"]["permissions"])

    def test_health_shape(self):
        r = self.client.get(f"{PREFIX}/system/health", headers=self._auth())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        self.assertEqual(r.json()["engine"]["phase"], "monitoring")

    def test_cameras_redact_credentials(self):
        r = self.client.get(f"{PREFIX}/cameras", headers=self._auth())
        cams = r.json()
        self.assertEqual(len(cams), 1)
        self.assertNotIn("secret", cams[0]["source"])
        self.assertIn("rtsp://***@10.0.0.9/stream1", cams[0]["source"])
        self.assertEqual(cams[0]["state"], "connected")   # merged from health

    def test_events_list_and_shape(self):
        r = self.client.get(f"{PREFIX}/events", headers=self._auth())
        body = r.json()
        self.assertEqual(len(body["events"]), 2)
        newest = body["events"][0]
        self.assertTrue(newest["id"].startswith("evt_"))
        self.assertIn(newest["verdict"], ("confirmed", "rejected"))
        self.assertIn("thumb", newest["evidence"])

    def test_events_cursor_paginates(self):
        first = self.client.get(f"{PREFIX}/events?limit=1", headers=self._auth()).json()
        self.assertEqual(len(first["events"]), 1)
        self.assertIsNotNone(first["next_cursor"])
        nxt = self.client.get(f"{PREFIX}/events?limit=1&cursor={first['next_cursor']}",
                              headers=self._auth()).json()
        self.assertEqual(len(nxt["events"]), 1)
        self.assertNotEqual(first["events"][0]["id"], nxt["events"][0]["id"])

    def test_event_detail_and_404(self):
        got = self.client.get(f"{PREFIX}/events/evt_1", headers=self._auth())
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.json()["camera_id"], "Dublin Street")
        miss = self.client.get(f"{PREFIX}/events/evt_999", headers=self._auth())
        self.assertEqual(miss.status_code, 404)

    def test_triage_counts(self):
        t = self.client.get(f"{PREFIX}/triage", headers=self._auth()).json()
        self.assertEqual(t["total"], 2)
        self.assertEqual(t["to_review"], 1)          # one has review=NULL

    def test_websocket_hydrates_and_pushes(self):
        token = self._token()
        with self.client.websocket_connect(f"{PREFIX}/stream?token={token}") as ws:
            first = ws.receive_json()
            self.assertEqual(first["type"], "health")
            second = ws.receive_json()
            self.assertEqual(second["type"], "triage")

    def test_websocket_rejects_bad_token(self):
        with self.assertRaises(Exception):
            with self.client.websocket_connect(f"{PREFIX}/stream?token=nope") as ws:
                ws.receive_json()


class MockApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_mock_app())

    def test_any_login_works_and_shapes_match(self):
        tok = self.client.post(f"{PREFIX}/auth/session",
                               json={"username": "demo", "password": "x"}).json()["token"]
        h = {"Authorization": f"Bearer {tok}"}
        self.assertEqual(self.client.get(f"{PREFIX}/system/health", headers=h).json()["gate"]["model"],
                         "gemma3:4b")
        cams = self.client.get(f"{PREFIX}/cameras", headers=h).json()
        self.assertTrue(any(c["view_only"] for c in cams))
        for c in cams:
            self.assertNotIn("secret", c["source"])
        evs = self.client.get(f"{PREFIX}/events", headers=h).json()["events"]
        self.assertTrue(evs and evs[0]["id"].startswith("evt_"))

    def test_mock_requires_a_token_too(self):
        self.assertEqual(self.client.get(f"{PREFIX}/events").status_code, 401)

    def test_mock_index_says_it_is_mock(self):
        body = self.client.get("/").json()
        self.assertTrue(body["mock"])
        self.assertEqual(body["docs"], "/docs")


if __name__ == "__main__":
    unittest.main()
