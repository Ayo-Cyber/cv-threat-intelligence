"""Identity, access and audit (EP-03-T1, T3).

`grep` confirmed 0 files with auth and 0 with audit logging. The failure that
matters most is not misuse of the cameras — it is that anyone could silently
disable detection and leave no record that they did.
"""
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from cvti.security.accounts import (
    MAX_FAILED,
    AccountStore,
    AuthError,
    UnsupportedKDF,
    bootstrap_password,
    hash_password,
)
from cvti.security.audit import GENESIS, AuditLog, entry_hash


class PasswordHashingTest(unittest.TestCase):
    def test_a_password_is_never_stored_recoverably(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "auth.db"
            store = AccountStore(db)
            store.create_user("ayo", "correct-horse-battery")
            store.close()
            blob = db.read_bytes()
            self.assertNotIn(b"correct-horse-battery", blob)

    def test_the_same_password_hashes_differently_each_time(self):
        # Per-user salt: two accounts with the same password must not be
        # visibly identical in the database.
        a = hash_password("same-password")
        b = hash_password("same-password")
        self.assertNotEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])

    def test_the_kdf_is_recorded_so_hashes_stay_readable(self):
        _, _, kdf = hash_password("x" * 12)
        self.assertTrue(kdf.startswith(("scrypt$", "pbkdf2$")), kdf)

    def test_a_hash_verifies_against_its_own_recorded_parameters(self):
        salt, digest, kdf = hash_password("a-real-password")
        again = hash_password("a-real-password", salt, kdf)
        self.assertEqual(again[1], digest)

    def test_an_unreadable_hash_format_is_not_reported_as_a_wrong_password(self):
        with self.assertRaises(UnsupportedKDF):
            hash_password("x", b"salt" * 4, "bcrypt$something")

    def test_bootstrap_passwords_are_random(self):
        self.assertNotEqual(bootstrap_password(), bootstrap_password())
        self.assertGreaterEqual(len(bootstrap_password()), 12)


class AuthenticationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = AccountStore(Path(self._tmp.name) / "auth.db")
        self.store.create_user("ayo", "correct-horse-battery", role="owner")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_correct_credentials_authenticate(self):
        self.assertEqual(self.store.authenticate("ayo", "correct-horse-battery").role, "owner")

    def test_a_wrong_password_is_refused(self):
        with self.assertRaises(AuthError):
            self.store.authenticate("ayo", "not-the-password")

    def test_an_unknown_user_is_refused_the_same_way(self):
        # Different messages here would enumerate valid usernames.
        try:
            self.store.authenticate("nobody", "whatever")
        except AuthError as exc:
            unknown = str(exc)
        try:
            self.store.authenticate("ayo", "wrong")
        except AuthError as exc:
            wrong = str(exc)
        self.assertEqual(unknown, wrong)

    def test_repeated_failures_lock_the_account(self):
        for _ in range(MAX_FAILED):
            with self.assertRaises(AuthError):
                self.store.authenticate("ayo", "wrong")
        self.assertTrue(self.store.locked_out("ayo"))
        with self.assertRaises(AuthError):
            self.store.authenticate("ayo", "correct-horse-battery")   # even the right one

    def test_a_successful_login_clears_the_failure_streak(self):
        for _ in range(MAX_FAILED - 1):
            with self.assertRaises(AuthError):
                self.store.authenticate("ayo", "wrong")
        self.store.authenticate("ayo", "correct-horse-battery")
        for _ in range(MAX_FAILED - 1):
            with self.assertRaises(AuthError):
                self.store.authenticate("ayo", "wrong")
        self.assertFalse(self.store.locked_out("ayo"))

    def test_a_short_password_is_refused(self):
        with self.assertRaises(ValueError):
            self.store.create_user("weak", "short")

    def test_an_unknown_role_is_refused(self):
        with self.assertRaises(ValueError):
            self.store.create_user("x", "a-good-password", role="superadmin")


class SessionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = AccountStore(Path(self._tmp.name) / "auth.db")
        self.store.create_user("ayo", "correct-horse-battery", role="owner")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_a_token_resolves_to_its_user(self):
        token = self.store.open_session("ayo")
        self.assertEqual(self.store.session_user(token).username, "ayo")

    def test_an_expired_session_stops_resolving(self):
        token = self.store.open_session("ayo", timeout=-1)
        self.assertIsNone(self.store.session_user(token))

    def test_an_unknown_or_empty_token_resolves_to_nobody(self):
        self.assertIsNone(self.store.session_user("made-up"))
        self.assertIsNone(self.store.session_user(""))

    def test_changing_a_password_revokes_existing_sessions(self):
        # Otherwise a password change after a compromise changes nothing.
        token = self.store.open_session("ayo")
        self.store.set_password("ayo", "a-brand-new-password")
        self.assertIsNone(self.store.session_user(token))

    def test_logout_invalidates_the_token(self):
        token = self.store.open_session("ayo")
        self.store.close_session(token)
        self.assertIsNone(self.store.session_user(token))


class AuditAppendOnlyTest(unittest.TestCase):
    """No application path may modify or delete an entry."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log = AuditLog(Path(self._tmp.name) / "audit.db")

    def tearDown(self):
        self.log.close()
        self._tmp.cleanup()

    def test_there_is_no_api_that_mutates_an_entry(self):
        for forbidden in ("update", "delete", "edit", "remove", "clear", "purge"):
            self.assertFalse(hasattr(self.log, forbidden),
                             f"AuditLog exposes {forbidden}() — the log is append-only")

    def test_the_database_itself_refuses_an_update(self):
        self.log.record("ayo", "login")
        with self.assertRaises(sqlite3.IntegrityError):
            self.log._db.execute("UPDATE audit SET actor='someone-else' WHERE seq=1")

    def test_the_database_itself_refuses_a_delete(self):
        self.log.record("ayo", "login")
        with self.assertRaises(sqlite3.IntegrityError):
            self.log._db.execute("DELETE FROM audit WHERE seq=1")

    def test_it_records_every_class_the_plan_requires(self):
        from cvti.security.audit import ACTIONS
        for action in ACTIONS:
            self.log.record("ayo", action, target="thing", detail={"k": "v"})
        got = {e.action for e in self.log.entries()}
        self.assertEqual(got, set(ACTIONS))

    def test_each_entry_carries_actor_timestamp_and_target(self):
        self.log.record("ayo", "config_change", "detector:weapons", {"to": False})
        entry = self.log.entries()[0]
        self.assertEqual(entry.actor, "ayo")
        self.assertEqual(entry.target, "detector:weapons")
        self.assertEqual(entry.detail["to"], False)
        self.assertGreater(entry.ts, 0)


class AuditTamperEvidenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "audit.db"
        self.log = AuditLog(self.path)

    def tearDown(self):
        self.log.close()
        self._tmp.cleanup()

    def test_an_untouched_chain_verifies(self):
        for i in range(5):
            self.log.record("ayo", "login", detail={"i": i})
        result = self.log.verify()
        self.assertTrue(result["ok"])
        self.assertEqual(result["checked"], 5)

    def test_the_first_entry_links_to_genesis(self):
        self.log.record("ayo", "login")
        self.assertEqual(self.log.entries()[0].prev_hash, GENESIS)

    def test_editing_an_entry_is_detected(self):
        # The realistic attack: someone with a SQLite browser quietly changes
        # who disabled the detector.
        self.log.record("ayo", "login")
        self.log.record("mallory", "config_change", "detector:weapons", {"to": False})
        self.log.record("ayo", "login")
        self.log.close()

        con = sqlite3.connect(self.path)
        con.execute("DROP TRIGGER audit_no_update")          # they own the disk
        con.execute("UPDATE audit SET actor='ayo' WHERE actor='mallory'")
        con.commit()
        con.close()

        result = AuditLog(self.path).verify()
        self.assertFalse(result["ok"], "an edited entry verified as intact")
        self.assertEqual(result["broken_at"], 2)
        self.assertIn("edited", result["reason"])

    def test_removing_an_entry_is_detected(self):
        for _ in range(3):
            self.log.record("ayo", "login")
        self.log.close()
        con = sqlite3.connect(self.path)
        con.execute("DROP TRIGGER audit_no_delete")
        con.execute("DELETE FROM audit WHERE seq=2")
        con.commit()
        con.close()
        result = AuditLog(self.path).verify()
        self.assertFalse(result["ok"], "a removed entry left no trace")

    def test_the_hash_covers_the_contents_not_just_the_order(self):
        a = entry_hash(1.0, "ayo", "login", "", {}, GENESIS)
        b = entry_hash(1.0, "mallory", "login", "", {}, GENESIS)
        self.assertNotEqual(a, b)

    def test_export_carries_its_own_verification(self):
        import json
        self.log.record("ayo", "login")
        out = self.log.export(Path(self._tmp.name) / "audit-export.json")
        data = json.loads(out.read_text())
        self.assertTrue(data["verification"]["ok"])
        self.assertEqual(len(data["entries"]), 1)


class StoredSeparatelyTest(unittest.TestCase):
    def test_credentials_and_audit_do_not_live_in_events_db(self):
        # A single deletion must not remove both the footage and the record of
        # who touched it.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AccountStore(root / "auth.db")
            audit = AuditLog(root / "audit.db")
            store.create_user("ayo", "correct-horse-battery")
            audit.record("ayo", "login")
            store.close()
            audit.close()
            self.assertTrue((root / "auth.db").exists())
            self.assertTrue((root / "audit.db").exists())
            self.assertFalse((root / "events.db").exists())


if __name__ == "__main__":
    unittest.main()
