"""The mock gate confirms every alert. Starting an engine on it must be loud."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from cvti.verification.gate import (
    ALLOW_MOCK_GATE_ENV,
    MOCK_GATE_BANNER,
    MockGateRefused,
    assert_engine_gate_allowed,
)


class MockGateGuardTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop(ALLOW_MOCK_GATE_ENV, None)

    def tearDown(self):
        os.environ.pop(ALLOW_MOCK_GATE_ENV, None)
        if self._saved is not None:
            os.environ[ALLOW_MOCK_GATE_ENV] = self._saved

    def test_real_providers_pass(self):
        for provider in ("ollama", "local", "anthropic", "openrouter"):
            self.assertFalse(assert_engine_gate_allowed(provider), provider)

    def test_mock_refused_by_default(self):
        with self.assertRaises(MockGateRefused) as ctx:
            assert_engine_gate_allowed("mock")
        self.assertIn(ALLOW_MOCK_GATE_ENV, str(ctx.exception))

    def test_mock_allowed_only_by_exact_optin(self):
        for value in ("0", "", "true", "yes"):
            os.environ[ALLOW_MOCK_GATE_ENV] = value
            with self.assertRaises(MockGateRefused):
                assert_engine_gate_allowed("mock")
        os.environ[ALLOW_MOCK_GATE_ENV] = "1"
        self.assertTrue(assert_engine_gate_allowed("mock"))

    def test_engine_refuses_before_loading_anything(self):
        # A missing site config would normally raise on load; the guard must fire
        # first, which proves it runs before any model or stream is touched.
        from cvti.serving.pipeline import run_site
        with self.assertRaises(MockGateRefused):
            run_site("/nonexistent/site.json", gate_provider="mock")


class GateHealthTest(unittest.TestCase):
    """The System panel reads engine gate stats from a file the engine writes."""

    def _backend(self, tmp):
        from cvti.app.console_backend import ConsoleBackend
        return ConsoleBackend(db_path=str(Path(tmp) / "events.db"))

    def test_absent_health_file_reads_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._backend(tmp)._gate_health(), {})

    def test_fresh_health_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "gate_health.json").write_text(json.dumps(
                {"errors": 3, "last_error": "cam1::weapons — timeout",
                 "mock": True, "banner": MOCK_GATE_BANNER, "updated_at": time.time()}))
            health = self._backend(tmp)._gate_health()
            self.assertEqual(health["errors"], 3)
            self.assertTrue(health["mock"])

    def test_stale_health_is_discarded(self):
        # An engine that exited an hour ago must not paint a healthy gate.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "gate_health.json").write_text(json.dumps(
                {"errors": 0, "updated_at": time.time() - 3600}))
            self.assertEqual(self._backend(tmp)._gate_health(), {})


if __name__ == "__main__":
    unittest.main()
