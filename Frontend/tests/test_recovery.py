import importlib.util
from pathlib import Path
import tempfile
import unittest
from cvti.security.accounts import AccountStore
from cvti.security.audit import AuditLog

spec = importlib.util.spec_from_file_location('recovery', Path(__file__).resolve().parents[1] / 'scripts' / 'recover_account.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = AccountStore(self.root / 'auth.db')
        self.store.create_user('owner', 'Original-Password-2026', 'owner')
        self.store.create_user('operator', 'Operator-Password-2026', 'operator')
        self.token = self.store.open_session('owner')
        self.other = self.store.open_session('operator')
        self.evidence = self.root / 'events.db'
        self.evidence.write_bytes(b'untouched evidence')
    def tearDown(self):
        self.store.close()
        self.temp.cleanup()
    def test_reset_preserves_users_evidence_and_revokes_only_target_sessions(self):
        recovery.reset_account(self.evidence, 'owner', 'Replacement-Password-2026', actor='local:test')
        self.assertEqual(self.store.authenticate('owner', 'Replacement-Password-2026').role, 'owner')
        self.assertIsNone(self.store.session_user(self.token))
        self.assertIsNotNone(self.store.session_user(self.other))
        self.assertEqual(len(self.store.list_users()), 2)
        self.assertEqual(self.evidence.read_bytes(), b'untouched evidence')
        audit = AuditLog(self.root / 'audit.db')
        self.assertTrue(audit.verify()['ok'])
        audit.close()
    def test_invalid_user_and_short_password_do_not_change_credentials(self):
        for user, password in [('missing', 'Replacement-Password-2026'), ('owner', 'short')]:
            with self.assertRaises(ValueError):
                recovery.reset_account(self.evidence, user, password, actor='local:test')
        self.assertIsNotNone(self.store.authenticate('owner', 'Original-Password-2026'))
    def test_refuses_missing_store(self):
        with self.assertRaises(ValueError):
            recovery.reset_account(self.root / 'missing' / 'events.db', 'owner', 'Replacement-Password-2026', actor='local:test')
    def test_reset_clears_only_target_lockout(self):
        import sqlite3
        import time
        with sqlite3.connect(self.root / 'auth.db') as db:
            db.executemany('INSERT INTO login_failures (username, at) VALUES (?, ?)', [('owner', time.time())] * 5 + [('operator', time.time())] * 5)
        self.assertTrue(self.store.locked_out('owner'))
        recovery.reset_account(self.evidence, 'owner', 'Replacement-Password-2026', actor='local:test')
        self.assertFalse(self.store.locked_out('owner'))
        self.assertTrue(self.store.locked_out('operator'))
