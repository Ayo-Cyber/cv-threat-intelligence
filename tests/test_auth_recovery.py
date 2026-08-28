"""'Invalid credentials' must not be a dead end.

Field report, 28 Aug: a machine with a forgotten account showed the sign-in
page and rejected everything, and nothing anywhere said a recovery path
existed. There is no dev login BY DESIGN — so the sign-in screen has to say
what recovery is: an OS-level file deletion, which keeps the trust boundary at
'has access to the machine's files', not 'is standing at the console'.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from cvti.app.console_backend import ConsoleBackend


def _backend(tmp):
    site = Path(tmp) / "site"
    site.mkdir(exist_ok=True)
    return ConsoleBackend(site_path=str(site / "site.json"),
                          db_path=str(site / "events.db"), enable_demo=False)


class AuthRecoveryTest(unittest.TestCase):

    def test_recovery_names_the_real_store_on_this_machine(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            info = cb.auth_recovery()
            self.assertEqual(Path(info["auth_db"]).name, "auth.db")
            self.assertTrue(str(info["auth_db"]).startswith(tmp),
                            "recovery points somewhere other than this site's store")

    def test_recovery_needs_no_session(self):
        """The whole point is that the caller CANNOT sign in."""
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            self.assertIsNone(cb.current_user)
            self.assertIn("auth_db", cb.auth_recovery())

    def test_viewing_recovery_is_audit_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            cb.auth_recovery()
            con = sqlite3.connect(Path(tmp) / "site" / "audit.db")
            try:
                n = con.execute("SELECT COUNT(*) FROM audit WHERE action='auth_recovery_viewed'"
                                ).fetchone()[0]
            finally:
                con.close()
            self.assertEqual(n, 1, "a recovery view left no audit trace")

    def test_deleting_the_store_returns_the_site_to_first_run(self):
        """The documented recovery, executed: accounts reset, evidence kept."""
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            cb.create_first_owner("ayo", "correct-horse-9")
            self.assertTrue(cb.auth_state()["configured"])
            store = Path(cb.auth_recovery()["auth_db"])
            del cb
            store.unlink()
            cb2 = _backend(tmp)
            st = cb2.auth_state()
            self.assertFalse(st["configured"], "still shows sign-in after the reset")
            # the audit trail survives the reset — including the recovery view
            con = sqlite3.connect(Path(tmp) / "site" / "audit.db")
            try:
                n = con.execute("SELECT COUNT(*) FROM audit").fetchone()[0]
            finally:
                con.close()
            self.assertGreater(n, 0, "audit history was lost in the reset")

    def test_the_signin_screen_offers_the_path(self):
        html = Path("cvti/app/web/index.html").read_text()
        self.assertIn("showAuthRecovery", html)
        self.assertIn('call("authRecovery"', html)
        self.assertIn("Can\\'t sign in?", html)


if __name__ == "__main__":
    unittest.main()


class FirstPaintIsAuthTest(unittest.TestCase):
    """The console must never be the first thing an unauthenticated eye sees.

    The auth gate used to start hidden and only appear after the bridge
    connected and authState answered — so every launch flashed the whole
    console first (28 Aug: 'the front and first page should be sign in or
    create an account not this thing we are seeing')."""

    def test_the_gate_is_open_in_the_static_html(self):
        html = Path("cvti/app/web/index.html").read_text()
        self.assertIn('<div id="authGate" class="on">', html,
                      "first paint shows the console, not the auth gate")

    def test_a_missing_bridge_reports_into_the_gate(self):
        html = Path("cvti/app/web/index.html").read_text()
        self.assertNotIn('document.getElementById("screen").innerHTML=\'<div class="pad mut">Backend bridge', html)
