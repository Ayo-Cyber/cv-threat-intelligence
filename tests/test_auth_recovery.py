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


class BothDoorsOnTheFirstScreenTest(unittest.TestCase):
    """'It has to be both' (28 Aug): Sign in AND Create account are always
    visible tabs. On a configured site the create tab EXPLAINS rather than
    mints — an unauthenticated create on a security console would let anyone
    at the login screen join the site."""

    def setUp(self):
        self.html = Path("cvti/app/web/index.html").read_text()

    def test_both_tabs_exist(self):
        self.assertIn('tab("signin","Sign in")', self.html)
        self.assertIn('tab("create","Create account")', self.html)

    def test_a_configured_site_overrides_loudly_never_quietly(self):
        """Policy changed 28 Aug at the owner's reaffirmed decision: the create
        tab REPLACES accounts (pilot-phase). The pinned invariants are the
        controls, not the prohibition: the tab lists who will be replaced and
        goes through the confirmed, audited override — never doFirstRun."""
        self.assertIn('call("authAccounts"', self.html)
        self.assertIn("replaces every account above", self.html)
        create_branch = self.html.split("if(state.authConfigured){", 2)[-1][:2000]
        self.assertIn("doOwnerOverride", create_branch)
        self.assertNotIn("doFirstRun", create_branch,
                         "a configured site reached the unaudited first-run create")

    def test_the_old_single_track_screens_are_gone(self):
        self.assertNotIn("function renderSignIn(", self.html)
        self.assertNotIn("function renderFirstRun(", self.html)


class OwnerOverrideTest(unittest.TestCase):
    """Pilot-phase decision, reaffirmed 28 Aug: the create tab on a configured
    site REPLACES existing accounts. Compensating controls, each pinned here:
    the screen can list who will be replaced, the replacement is audit-logged
    with the old names, sessions die with the old accounts, and evidence and
    audit survive."""

    def test_accounts_are_listable_before_auth(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            cb.create_first_owner("martins", "pilot-pass-1")
            cb.sign_out()
            cb2 = _backend(tmp)                      # nobody signed in
            names = [u["username"] for u in cb2.auth_accounts()["users"]]
            self.assertEqual(names, ["martins"])

    def test_override_replaces_and_signs_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            cb.create_first_owner("old-owner", "forgotten-pw-1")
            cb.sign_out()
            r = cb.create_owner_override("ayo", "fresh-pass-99")
            self.assertTrue(r["ok"])
            self.assertEqual(r["username"], "ayo")
            names = [u["username"] for u in cb.auth_accounts()["users"]]
            self.assertEqual(names, ["ayo"], "the old account survived the override")

    def test_the_replacement_is_loud_in_the_audit_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            cb.create_first_owner("old-owner", "forgotten-pw-1")
            cb.create_owner_override("ayo", "fresh-pass-99")
            con = sqlite3.connect(Path(tmp) / "site" / "audit.db")
            try:
                row = con.execute("SELECT detail FROM audit WHERE action="
                                  "'accounts_override_via_login'").fetchone()
            finally:
                con.close()
            self.assertIsNotNone(row, "an account takeover left no audit trace")
            self.assertIn("old-owner", row[0], "the audit row does not name who was replaced")

    def test_old_sessions_die_with_the_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            cb.create_first_owner("old-owner", "forgotten-pw-1")   # signs in
            self.assertIsNotNone(cb.current_user)
            cb.create_owner_override("ayo", "fresh-pass-99")
            self.assertEqual(cb.current_user.username, "ayo",
                             "the old session outlived its account")

    def test_the_ui_confirms_before_replacing(self):
        html = Path("cvti/app/web/index.html").read_text()
        self.assertIn("doOwnerOverride", html)
        self.assertIn("confirm(", html.split("function doOwnerOverride")[1][:400],
                      "the override fires without a confirmation")
