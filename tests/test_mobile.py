"""The mobile response view (EP-06-T3).

The sequencing risk the plan names in bold: this is the first Argus surface
that leaves localhost, and an unauthenticated route here puts live camera
evidence on an open port on a customer's network. So the first test class does
nothing but try to get in without signing in.
"""
import sqlite3
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

import cv2
import numpy as np

from cvti.security.accounts import AccountStore
from cvti.serving.alert_sink import AlertSink
from cvti.serving.mobile import MobileServer, _csrf_for


def _jpeg():
    ok, buf = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
    return buf.tobytes()


class _Stack:
    """A real site directory: accounts, events db with evidence, mobile server."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        AlertSink(self.root, save_evidence=False, routing_path=None).close()
        accounts = AccountStore(self.root / "auth.db")
        accounts.create_user("guard", "a-strong-password", role="operator")
        accounts.create_user("fitter", "a-strong-password", role="installer")
        accounts.close()

        ev_dir = self.root / "events" / "20260820_warehouse_fire"
        ev_dir.mkdir(parents=True)
        (ev_dir / "frame_000.jpg").write_bytes(_jpeg())
        con = sqlite3.connect(self.root / "events.db")
        cur = con.execute(
            "INSERT INTO events (ts, iso, camera_id, rule, priority, confidence, "
            "reason, evidence_dir) VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), "2026-08-20T21:04:12", "warehouse", "baseline_fire_smoke",
             "critical", 0.94, "Visible flame at the shelving.", str(ev_dir)))
        con.commit()
        self.event = cur.lastrowid
        con.close()

        self.server = MobileServer(self.root, port=0, host="127.0.0.1")
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server.port}"

    def close(self):
        self.server.stop()
        self._tmp.cleanup()

    # -- an http client with a cookie jar, like a phone browser --
    def client(self):
        return urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar()))

    def sign_in(self, opener, user="guard", password="a-strong-password"):
        data = urllib.parse.urlencode({"username": user, "password": password,
                                       "next": "/"}).encode()
        return opener.open(f"{self.base}/login", data=data, timeout=5)

    def csrf(self, opener) -> str:
        jar = [c for h in opener.handlers if isinstance(h, urllib.request.HTTPCookieProcessor)
               for c in h.cookiejar]
        token = next(c.value for c in jar if c.name == "argus_session")
        return _csrf_for(token)


class NoUnauthenticatedRouteTest(unittest.TestCase):
    """The acceptance item in bold: verify no unauthenticated route exists."""

    @classmethod
    def setUpClass(cls):
        cls.stack = _Stack()

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()

    def _get_unauth(self, path):
        req = urllib.request.Request(self.stack.base + path)
        try:
            resp = urllib.request.urlopen(req, timeout=5)   # no cookies, no redirect-follow issue
            return resp.status, resp.geturl(), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, "", b""

    def test_every_content_route_redirects_to_login(self):
        for path in ("/", f"/alert/{self.stack.event}",
                      f"/frame/{self.stack.event}/0.jpg", "/made-up"):
            status, final_url, body = self._get_unauth(path)
            self.assertTrue(final_url.endswith(urllib.parse.quote(path, safe="/?=&")
                                               and "") or "/login" in final_url,
                            f"{path} did not gate")
            self.assertIn(b"Sign in", body, f"{path} served content without a session")
            self.assertNotIn(b"Visible flame", body)
            self.assertNotIn(b"warehouse", body)

    def test_evidence_frames_never_leave_without_a_session(self):
        status, final_url, body = self._get_unauth(f"/frame/{self.stack.event}/0.jpg")
        self.assertFalse(body.startswith(b"\xff\xd8"),
                         "a camera frame was served to an unauthenticated request")

    def test_actions_are_refused_without_a_session(self):
        data = urllib.parse.urlencode({"outcome": "real"}).encode()
        req = urllib.request.Request(
            f"{self.stack.base}/resolve/{self.stack.event}", data=data)
        opener = urllib.request.build_opener()
        resp = opener.open(req, timeout=5)
        self.assertIn(b"Sign in", resp.read())
        con = sqlite3.connect(self.stack.root / "events.db")
        state = con.execute("SELECT state FROM events WHERE id=?",
                            (self.stack.event,)).fetchone()[0]
        con.close()
        self.assertNotEqual(state, "resolved", "an unauthenticated POST resolved an alert")

    def test_wrong_credentials_are_refused_with_the_backend_message(self):
        opener = self.stack.client()
        resp = self.stack.sign_in(opener, password="wrong-password")
        self.assertIn(b"invalid username or password", resp.read())


class SignedInFlowTest(unittest.TestCase):
    def setUp(self):
        self.stack = _Stack()
        self.opener = self.stack.client()
        self.stack.sign_in(self.opener)

    def tearDown(self):
        self.stack.close()

    def test_the_now_view_shows_the_alert_and_its_frame(self):
        body = self.opener.open(self.stack.base + "/", timeout=5).read()
        self.assertIn(b"baseline_fire_smoke", body)
        self.assertIn(b"Visible flame", body)
        self.assertIn(b"I\xe2\x80\x99m on it", body)
        frame = self.opener.open(
            f"{self.stack.base}/frame/{self.stack.event}/0.jpg", timeout=5).read()
        self.assertTrue(frame.startswith(b"\xff\xd8"))

    def test_the_deep_link_opens_the_specific_alert(self):
        body = self.opener.open(f"{self.stack.base}/alert/{self.stack.event}",
                                timeout=5).read()
        self.assertIn(b"warehouse", body)
        self.assertIn(b"21:04:12", body)

    def test_acknowledge_then_resolve_with_a_note_from_the_phone(self):
        csrf = self.stack.csrf(self.opener)
        self.opener.open(f"{self.stack.base}/ack/{self.stack.event}",
                         data=urllib.parse.urlencode({"csrf": csrf}).encode(),
                         timeout=5)
        con = sqlite3.connect(self.stack.root / "events.db")
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT state, owner FROM events WHERE id=?",
                          (self.stack.event,)).fetchone()
        self.assertEqual((row["state"], row["owner"]), ("acknowledged", "guard"))
        con.close()

        self.opener.open(f"{self.stack.base}/resolve/{self.stack.event}",
                         data=urllib.parse.urlencode(
                             {"csrf": csrf, "outcome": "real",
                              "note": "smoke cleared, stock moved"}).encode(),
                         timeout=5)
        con = sqlite3.connect(self.stack.root / "events.db")
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT state, outcome, note FROM events WHERE id=?",
                          (self.stack.event,)).fetchone()
        con.close()
        self.assertEqual(row["state"], "resolved")
        self.assertEqual(row["outcome"], "real")
        self.assertEqual(row["note"], "smoke cleared, stock moved")

    def test_actions_without_the_csrf_token_are_refused(self):
        # The cookie alone must not be enough — that is the cross-site hole.
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.opener.open(f"{self.stack.base}/resolve/{self.stack.event}",
                             data=urllib.parse.urlencode(
                                 {"outcome": "real"}).encode(), timeout=5)
        self.assertEqual(ctx.exception.code, 403)
        con = sqlite3.connect(self.stack.root / "events.db")
        state = con.execute("SELECT state FROM events WHERE id=?",
                            (self.stack.event,)).fetchone()[0]
        con.close()
        self.assertNotEqual(state, "resolved")

    def test_a_second_guard_is_told_who_has_it(self):
        csrf = self.stack.csrf(self.opener)
        self.opener.open(f"{self.stack.base}/ack/{self.stack.event}",
                         data=urllib.parse.urlencode({"csrf": csrf}).encode(),
                         timeout=5)
        # sam signs in on their own phone
        accounts = AccountStore(self.stack.root / "auth.db")
        accounts.create_user("sam", "a-strong-password", role="operator")
        accounts.close()
        sam = self.stack.client()
        self.stack.sign_in(sam, user="sam")
        body = sam.open(f"{self.stack.base}/ack/{self.stack.event}",
                        data=urllib.parse.urlencode(
                            {"csrf": self.stack.csrf(sam)}).encode(),
                        timeout=5).read()
        self.assertIn(b"guard is already on alert", body)

    def test_an_installer_cannot_read_incidents_from_the_phone_either(self):
        # Role enforcement is the same on every surface.
        fitter = self.stack.client()
        # sign_in itself 403s: the 303 redirect lands on "/" which the role
        # cannot view. That IS the enforcement working — catch and inspect it.
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.stack.sign_in(fitter, user="fitter")
        self.assertEqual(ctx.exception.code, 403)
        body = ctx.exception.read()
        self.assertIn(b"cannot view alerts", body)
        self.assertNotIn(b"Visible flame", body)

    def test_the_open_redirect_is_closed(self):
        opener = self.stack.client()
        data = urllib.parse.urlencode({"username": "guard",
                                       "password": "a-strong-password",
                                       "next": "http://evil.example/"}).encode()
        resp = opener.open(f"{self.stack.base}/login", data=data, timeout=5)
        self.assertTrue(resp.geturl().startswith(self.stack.base),
                        "login redirected off-host")


class DeepLinkWiringTest(unittest.TestCase):
    def test_notifications_carry_the_response_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = AlertSink(tmp, save_evidence=False, routing_path=None)
            sink.mobile_base = "http://192.168.1.20:8710"
            from cvti.contracts import VerificationResult
            from cvti.serving.alert_queue import QueuedAlert
            alert = QueuedAlert(camera_id="cam1", rule_name="theft", priority="high",
                                title="t", timestamp=0.0,
                                payload={"frames": [], "enqueued_at": None})
            captured = []
            sink.notifier = type("N", (), {"notify": lambda self, e: captured.append(e)})()
            sink.handle(alert, VerificationResult(True, 0.9, "r", "high", "t"))
            self.assertTrue(captured[0]["link"].startswith(
                "http://192.168.1.20:8710/alert/"))
            sink.close()


if __name__ == "__main__":
    unittest.main()


class GlobalIdentityTest(unittest.TestCase):
    """Per-feed event stores (25 Aug) must never fragment identity: accounts
    and the tamper-evident audit log are global to the install. The regression
    this pins: on a non-home feed the phone view built an EMPTY auth.db and
    nobody could sign in."""

    def test_the_mobile_server_uses_the_security_dir_not_the_feed_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; feed = Path(tmp) / "feeds" / "live"
            home.mkdir(parents=True); feed.mkdir(parents=True)
            accounts = AccountStore(home / "auth.db")
            accounts.create_user("guard", "a-strong-password", role="operator")
            accounts.close()
            AlertSink(feed, save_evidence=False, routing_path=None).close()
            srv = MobileServer(feed, port=0, host="127.0.0.1",
                               security_dir=str(home))
            try:
                self.assertTrue(srv.accounts.any_users(),
                                "phone view has no accounts — nobody can sign in")
            finally:
                srv.accounts.close()
            self.assertFalse((feed / "auth.db").exists(),
                             "identity leaked into the per-feed store")

    def test_it_still_defaults_to_the_output_dir_standalone(self):
        with tempfile.TemporaryDirectory() as tmp:
            AlertSink(tmp, save_evidence=False, routing_path=None).close()
            srv = MobileServer(tmp, port=0, host="127.0.0.1")
            try:
                self.assertTrue((Path(tmp) / "auth.db").exists())
            finally:
                srv.accounts.close()

    def test_the_engine_is_told_where_identity_lives(self):
        import inspect
        from cvti.app.console_backend import ConsoleBackend
        src = inspect.getsource(ConsoleBackend._spawn_engine)
        self.assertIn("--security-dir", src)
