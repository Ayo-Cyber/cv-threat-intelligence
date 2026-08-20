"""Three roles, enforced server-side (EP-03-T2).

The interface assumed installer, operator and owner were one person. These
tests deliberately call the backend directly, bypassing the UI entirely —
because hiding a button changes what is easy, not what is possible, and what a
security review asks about is what is *possible*.
"""
import tempfile
import unittest
from pathlib import Path

from _backend_helper import OWNER_PASSWORD, signed_in

from cvti.security.permissions import (
    CONFIGURE_DETECTORS,
    INSTALLER,
    OPERATOR,
    OWNER,
    PermissionDenied,
    allows,
    landing_for,
    permissions_for,
    require,
)


class PermissionTableTest(unittest.TestCase):
    def test_an_operator_cannot_configure_detectors(self):
        # The reason the roles are separate at all: a detector must not be
        # switched off during a shift.
        self.assertFalse(allows(OPERATOR, CONFIGURE_DETECTORS))
        self.assertTrue(allows(OWNER, CONFIGURE_DETECTORS))
        self.assertTrue(allows(INSTALLER, CONFIGURE_DETECTORS))

    def test_an_installer_cannot_read_recorded_incidents(self):
        # They commission the site and leave.
        self.assertFalse(allows(INSTALLER, "view_alerts"))
        self.assertTrue(allows(INSTALLER, "view_live"))

    def test_only_the_owner_manages_users_and_reads_the_audit_trail(self):
        for role in (OPERATOR, INSTALLER):
            self.assertFalse(allows(role, "manage_users"))
            self.assertFalse(allows(role, "view_audit"))
        self.assertTrue(allows(OWNER, "manage_users"))
        self.assertTrue(allows(OWNER, "view_audit"))

    def test_an_unknown_role_holds_nothing(self):
        self.assertEqual(permissions_for("superadmin"), frozenset())

    def test_anonymous_is_refused(self):
        with self.assertRaises(PermissionDenied):
            require(None, "view_alerts")

    def test_each_role_lands_where_it_came_for(self):
        self.assertEqual(landing_for(INSTALLER), "watch")
        self.assertEqual(landing_for(OPERATOR), "triage")


class BackendEnforcementTest(unittest.TestCase):
    """Calling the backend directly — the UI is not in the way."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.site = root / "site.json"
        self.site.write_text('{"cameras": [{"id": "cam1", "source": "rtsp://x/y"}]}')
        self.db = root / "events.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _backend(self, role):
        return signed_in(role, site_path=str(self.site), db_path=str(self.db),
                         enable_demo=False)

    def test_an_operator_is_refused_detector_configuration(self):
        be = self._backend(OPERATOR)
        with self.assertRaises(PermissionDenied):
            be.set_camera_rules("cam1", {"weapons": False})

    def test_an_operator_is_refused_camera_changes(self):
        be = self._backend(OPERATOR)
        with self.assertRaises(PermissionDenied):
            be.remove_camera("cam1")
        with self.assertRaises(PermissionDenied):
            be.add_zone("cam1", "till", [[0, 0], [1, 0], [1, 1]])

    def test_an_operator_is_refused_engine_control(self):
        # Stopping monitoring is the "silently disable detection" case.
        be = self._backend(OPERATOR)
        with self.assertRaises(PermissionDenied):
            be.stop_monitoring()

    def test_an_operator_is_refused_the_audit_trail_and_user_admin(self):
        be = self._backend(OPERATOR)
        for call in (be.audit_entries, be.list_users, be.audit_verify):
            with self.assertRaises(PermissionDenied):
                call()

    def test_an_operator_can_still_do_their_job(self):
        be = self._backend(OPERATOR)
        be.list_events(limit=1)          # must not raise
        self.assertEqual(be.auth_state()["role"], OPERATOR)

    def test_an_installer_is_refused_recorded_evidence(self):
        be = self._backend(INSTALLER)
        with self.assertRaises(PermissionDenied):
            be.list_events(limit=1)

    def test_an_installer_can_configure(self):
        be = self._backend(INSTALLER)
        be.set_camera_rules("cam1", {"weapons": True})     # must not raise

    def test_the_owner_can_do_all_of_it(self):
        be = self._backend(OWNER)
        be.set_camera_rules("cam1", {"weapons": True})
        be.list_events(limit=1)
        be.audit_entries()
        be.list_users()

    def test_signing_out_removes_every_permission(self):
        be = self._backend(OWNER)
        be.sign_out()
        with self.assertRaises(PermissionDenied):
            be.list_events(limit=1)
        self.assertFalse(be.auth_state()["signed_in"])


class RoleChangesAreAuditedTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        site = root / "site.json"
        site.write_text('{"cameras": []}')
        self.be = signed_in(OWNER, site_path=str(site),
                            db_path=str(root / "events.db"), enable_demo=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_creating_a_user_is_recorded(self):
        self.be.add_user("sam", OWNER_PASSWORD, role=OPERATOR)
        entry = self.be.audit_entries()[0]
        self.assertEqual(entry["action"], "role_change")
        self.assertEqual(entry["target"], "user:sam")
        self.assertEqual(entry["detail"]["role"], OPERATOR)

    def test_changing_a_role_records_both_sides(self):
        self.be.add_user("sam", OWNER_PASSWORD, role=OPERATOR)
        self.be.set_user_role("sam", INSTALLER)
        entry = self.be.audit_entries()[0]
        self.assertEqual(entry["detail"]["from"], OPERATOR)
        self.assertEqual(entry["detail"]["to"], INSTALLER)

    def test_logins_are_recorded_with_their_outcome(self):
        self.be.sign_out()
        self.be.sign_in("owner", "wrong-password")
        self.be.sign_in("owner", OWNER_PASSWORD)
        actions = [e for e in self.be.audit_entries() if e["action"] == "login"]
        outcomes = [e["detail"].get("outcome") for e in actions]
        self.assertIn("refused", outcomes)
        self.assertIn("success", outcomes)

    def test_the_last_owner_cannot_be_removed(self):
        # A site with no owner has nobody who can grant access to anyone again.
        result = self.be.remove_user("owner")
        self.assertFalse(result["ok"])

    def test_you_cannot_remove_your_own_account(self):
        self.be.add_user("second", OWNER_PASSWORD, role=OWNER)
        self.assertFalse(self.be.remove_user("owner")["ok"])

    def test_the_audit_trail_still_verifies_after_all_of_it(self):
        self.be.add_user("sam", OWNER_PASSWORD, role=OPERATOR)
        self.be.set_user_role("sam", INSTALLER)
        self.assertTrue(self.be.audit_verify()["ok"])


class FirstRunTest(unittest.TestCase):
    def test_nothing_ships_with_a_default_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            from cvti.app.console_backend import ConsoleBackend
            root = Path(tmp)
            (root / "site.json").write_text('{"cameras": []}')
            be = ConsoleBackend(site_path=str(root / "site.json"),
                                db_path=str(root / "events.db"), enable_demo=False)
            state = be.auth_state()
            self.assertFalse(state["configured"], "a fresh install already had an account")
            self.assertFalse(state["signed_in"])
            # And with no account, nothing consequential is reachable.
            with self.assertRaises(PermissionDenied):
                be.list_events(limit=1)

    def test_the_first_owner_can_be_created_once_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            from cvti.app.console_backend import ConsoleBackend
            root = Path(tmp)
            (root / "site.json").write_text('{"cameras": []}')
            be = ConsoleBackend(site_path=str(root / "site.json"),
                                db_path=str(root / "events.db"), enable_demo=False)
            self.assertTrue(be.create_first_owner("ayo", OWNER_PASSWORD)["ok"])
            second = be.create_first_owner("mallory", OWNER_PASSWORD)
            self.assertFalse(second["ok"], "a second owner was created unauthenticated")


if __name__ == "__main__":
    unittest.main()
