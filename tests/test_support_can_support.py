"""Support can actually support (latency audit 1 Sep, O3 + O2a).

Both pilot debugging sessions needed the same two things and had neither:
monitor.log (the engine subprocess's stdout — where its tracebacks land) was
missing from the Diagnose bundle, and no surface said WHICH BUILD wrote the
evidence. And monitor.log itself grew without bound while every watchdog
respawn leaked another open handle to it.

The contract:

- monitor.log rotates at spawn time when it outgrows 5 MB, keeping ONE
  predecessor so a crash's final words survive the restart that follows;
- every respawn closes the previous spawn's handle first;
- the Diagnose bundle ships monitor.log and monitor.log.1;
- the version is stamped where support looks: the heartbeat every engine
  writes, the diagnostics health snapshot, and the first line of every log —
  all reading the ONE source (cvti.utils.argus_version).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.app.console_backend import (MONITOR_LOG_CAP_BYTES, ConsoleBackend,
                                      _rotate_monitor_log)
from cvti.utils import argus_version

from _backend_helper import signed_in


class RotationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log = Path(self._tmp.name) / "monitor.log"

    def tearDown(self):
        self._tmp.cleanup()

    def test_an_oversized_log_rotates_to_dot_one(self):
        self.log.write_bytes(b"x" * (MONITOR_LOG_CAP_BYTES + 1))
        _rotate_monitor_log(self.log)
        self.assertFalse(self.log.exists())
        self.assertTrue(self.log.with_name("monitor.log.1").exists())

    def test_a_small_log_is_left_alone(self):
        self.log.write_text("recent crash context")
        _rotate_monitor_log(self.log)
        self.assertEqual(self.log.read_text(), "recent crash context")
        self.assertFalse(self.log.with_name("monitor.log.1").exists())

    def test_the_predecessor_is_replaced_not_accumulated(self):
        self.log.with_name("monitor.log.1").write_text("ancient history")
        self.log.write_bytes(b"y" * (MONITOR_LOG_CAP_BYTES + 1))
        _rotate_monitor_log(self.log)
        rotated = self.log.with_name("monitor.log.1")
        self.assertEqual(rotated.stat().st_size, MONITOR_LOG_CAP_BYTES + 1)


class SpawnHandleTests(unittest.TestCase):
    """Every respawn closes its predecessor's handle and rotates first."""

    def _backend(self, tmp: str) -> ConsoleBackend:
        return signed_in(site_path=str(Path(tmp) / "site.json"),
                         db_path=str(Path(tmp) / "events.db"),
                         enable_demo=False)

    def test_respawn_closes_the_previous_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(tmp)
            with mock.patch("cvti.verification.ollama.ensure_server",
                            return_value=True), \
                 mock.patch("cvti.app.console_backend.subprocess.Popen",
                            return_value=mock.Mock()):
                backend._spawn_engine()
                first = backend._engine_log_file
                self.assertFalse(first.closed)
                backend._spawn_engine()
                second = backend._engine_log_file
            self.assertTrue(first.closed)
            self.assertFalse(second.closed)
            self.assertIsNot(first, second)
            backend._close_engine_log()

    def test_spawn_rotates_an_oversized_log_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(tmp)
            log = Path(tmp) / "monitor.log"
            log.write_bytes(b"z" * (MONITOR_LOG_CAP_BYTES + 1))
            with mock.patch("cvti.verification.ollama.ensure_server",
                            return_value=True), \
                 mock.patch("cvti.app.console_backend.subprocess.Popen",
                            return_value=mock.Mock()):
                backend._spawn_engine()
            backend._close_engine_log()
            self.assertTrue(log.with_name("monitor.log.1").exists())
            self.assertLess(log.stat().st_size, 1024)   # a fresh file

    def test_stop_monitoring_releases_the_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(tmp)
            with mock.patch("cvti.verification.ollama.ensure_server",
                            return_value=True), \
                 mock.patch("cvti.app.console_backend.subprocess.Popen",
                            return_value=mock.Mock()):
                backend._spawn_engine()
                handle = backend._engine_log_file
                backend._monitor = None      # nothing real to terminate
                backend.stop_monitoring()
            self.assertTrue(handle.closed)


class BundleTests(unittest.TestCase):
    def test_the_diagnose_zip_ships_the_engines_own_log(self):
        from cvti.diagnostics import build_bundle
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "monitor.log").write_text("Traceback: the whole story\n")
            (out / "monitor.log.1").write_text("the previous chapter\n")
            bundle = build_bundle(out)
            with zipfile.ZipFile(bundle) as zf:
                names = set(zf.namelist())
                self.assertIn("logs/monitor.log", names)
                self.assertIn("logs/monitor.log.1", names)
                self.assertIn("health.json", names)

    def test_the_bundle_still_excludes_personal_data(self):
        from cvti.diagnostics import build_bundle
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "monitor.log").write_text("fine\n")
            bundle = build_bundle(out)
            with zipfile.ZipFile(bundle) as zf:
                for name in zf.namelist():
                    self.assertFalse(name.endswith((".jpg", ".mp4", ".db")), name)


class VersionEverywhereTests(unittest.TestCase):
    def test_the_heartbeat_says_which_build_wrote_it(self):
        from cvti.serving.health_doc import build_health_doc
        doc = build_health_doc(started_at=0.0, cameras=[], gate={}, disk={},
                               memory={}, components={"degraded": []})
        self.assertEqual(doc["version"], argus_version())

    def test_the_diagnostics_snapshot_says_which_build(self):
        from cvti.diagnostics import health_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            snap = health_snapshot(tmp)
        self.assertEqual(snap["argus"]["version"], argus_version())

    def test_the_sidebar_and_everything_else_read_the_same_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = signed_in(site_path=str(Path(tmp) / "s.json"),
                                db_path=str(Path(tmp) / "e.db"),
                                enable_demo=False)
            self.assertEqual(backend.app_version(), argus_version())

    def test_every_log_opens_by_naming_the_build(self):
        import inspect
        from cvti import logging_setup
        src = inspect.getsource(logging_setup.setup_logging)
        self.assertIn("argus_version", src)
        self.assertIn("argus=%s", src)


if __name__ == "__main__":
    unittest.main()
