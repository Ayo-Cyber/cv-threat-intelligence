import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('desktop_bridge', Path(__file__).resolve().parents[1] / 'bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

class Backend:
    current_user = None
    def auth_state(self):
        return {'signed_in': bool(self.current_user)}
    def _require(self, permission):
        raise PermissionError(permission)
    def set_site(self, *args):
        self._require('configure_site')
    def list_events(self, *args):
        return [{'evidence_dir': '/safe/event'}]

class DispatchTests(unittest.TestCase):
    def test_never_exposes_destructive_owner_override(self):
        with self.assertRaises(ValueError):
            bridge.dispatch(Backend(), 'create_owner_override', ['new', 'Long-Password-2026'])
    def test_signed_out_cannot_create_additional_accounts(self):
        with self.assertRaises(PermissionError):
            bridge.dispatch(Backend(), 'add_user', ['new', 'Long-Password-2026', 'owner'])
    def test_pre_auth_read(self):
        self.assertFalse(bridge.dispatch(Backend(), 'auth_state', [])['signed_in'])
    def test_requires_login(self):
        with self.assertRaises(PermissionError):
            bridge.dispatch(Backend(), 'list_cameras', [])
    def test_disallows_arbitrary_method(self):
        with self.assertRaises(ValueError):
            bridge.dispatch(Backend(), '__getattribute__', ['accounts'])
    def test_preserves_backend_permissions(self):
        b = Backend(); b.current_user = 'operator'
        with self.assertRaises(PermissionError):
            bridge.dispatch(b, 'set_site', ['Changed'])
    def test_guards_legacy_unguarded_mutation(self):
        b = Backend(); b.current_user = 'operator'
        with self.assertRaises(PermissionError):
            bridge.dispatch(b, 'mark_configured', [])
    def test_rejects_unrelated_evidence_path(self):
        b = Backend(); b.current_user = 'operator'
        with self.assertRaises(PermissionError):
            bridge.dispatch(b, 'event_clip', ['/private/other'])

if __name__ == '__main__':
    unittest.main()
