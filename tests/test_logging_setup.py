"""Logging and the diagnostics bundle (EP-01-T1).

Two things are load-bearing here. Logs must exist in the packaged build, where
the user has no terminal to fall back on. And the support bundle must not carry
images of people — this is a surveillance product, and "send us your logs" must
not quietly mean "send us your footage".
"""
import json
import logging
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from cvti import diagnostics
from cvti.logging_setup import (
    DIR_ENV,
    LEVEL_ENV,
    get_logger,
    reset_for_tests,
    resolve_log_dir,
    setup_logging,
)


class LoggingSetupTest(unittest.TestCase):
    def setUp(self):
        reset_for_tests()
        self._env = {k: os.environ.get(k) for k in (LEVEL_ENV, DIR_ENV)}
        for k in (LEVEL_ENV, DIR_ENV):
            os.environ.pop(k, None)

    def tearDown(self):
        reset_for_tests()
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_logger_carries_module_attribution(self):
        self.assertEqual(get_logger("cvti.serving.pipeline").name, "cvti.serving.pipeline")

    def test_writes_a_file_that_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = setup_logging(tmp, component="t", console=False)
            get_logger("cvti.test").info("first process")
            logging.shutdown()
            self.assertIn("first process", path.read_text())

            # Second process, same directory: appends rather than truncating.
            reset_for_tests()
            setup_logging(tmp, component="t", console=False)
            get_logger("cvti.test").info("second process")
            logging.shutdown()
            body = path.read_text()
            self.assertIn("first process", body)
            self.assertIn("second process", body)

    def test_rotation_actually_rotates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = setup_logging(tmp, component="t", console=False,
                                 max_bytes=2048, backup_count=2)
            log = get_logger("cvti.test")
            for i in range(400):
                log.info("padding line %d %s", i, "x" * 80)
            logging.shutdown()
            backups = sorted(path.parent.glob("t.log.*"))
            self.assertTrue(backups, "no rotated file was produced")
            self.assertLessEqual(len(backups), 2, "backup_count not honoured")

    def test_level_comes_from_the_environment(self):
        os.environ[LEVEL_ENV] = "WARNING"
        with tempfile.TemporaryDirectory() as tmp:
            path = setup_logging(tmp, component="t", console=False)
            get_logger("cvti.test").info("should not appear")
            get_logger("cvti.test").warning("should appear")
            logging.shutdown()
            body = path.read_text()
            self.assertNotIn("should not appear", body)
            self.assertIn("should appear", body)

    def test_a_nonsense_level_falls_back_rather_than_crashing(self):
        os.environ[LEVEL_ENV] = "LOUD"
        with tempfile.TemporaryDirectory() as tmp:
            setup_logging(tmp, component="t", console=False)
            self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_setup_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_logging(tmp, component="t", console=False)
            before = len(logging.getLogger().handlers)
            setup_logging(tmp, component="t", console=False)
            self.assertEqual(len(logging.getLogger().handlers), before)

    def test_the_two_processes_do_not_share_one_rotating_file(self):
        # App and engine are pointed at the same output dir. One shared handle
        # loses records on POSIX and fails outright on Windows.
        with tempfile.TemporaryDirectory() as tmp:
            engine = setup_logging(tmp, component="argus-engine", console=False)
            app = setup_logging(tmp, component="argus-app", console=False)
            self.assertNotEqual(engine, app)

    def test_frozen_build_logs_to_the_user_directory_not_the_cwd(self):
        # The case that matters: a bundle launched from a read-only mount would
        # silently fail to create a relative log dir, in the one build where
        # there is no terminal to fall back on.
        with tempfile.TemporaryDirectory() as tmp:
            sys.frozen = True                      # type: ignore[attr-defined]
            sys._MEIPASS = tmp                     # type: ignore[attr-defined]
            try:
                resolved = resolve_log_dir(tmp)
            finally:
                del sys.frozen                     # type: ignore[attr-defined]
                del sys._MEIPASS                   # type: ignore[attr-defined]
            self.assertNotEqual(resolved, Path(tmp) / "logs")
            self.assertEqual(resolved.name, "logs")
            self.assertTrue(resolved.is_absolute())

    def test_frozen_without_meipass_does_not_crash_at_import(self):
        # PyInstaller sets both; other freezers set only `frozen`. Reading
        # _MEIPASS unguarded took the app down before it could log why.
        import importlib

        import cvti.utils
        sys.frozen = True                          # type: ignore[attr-defined]
        try:
            importlib.reload(cvti.utils)           # must not raise
            self.assertTrue(cvti.utils.user_data_dir().is_absolute())
        finally:
            del sys.frozen                         # type: ignore[attr-defined]
            importlib.reload(cvti.utils)

    def test_explicit_dir_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[DIR_ENV] = str(Path(tmp) / "elsewhere")
            self.assertEqual(resolve_log_dir("/ignored"), Path(tmp) / "elsewhere")

    def test_an_unwritable_log_dir_does_not_stop_the_process(self):
        os.environ[DIR_ENV] = "/proc/argus-cannot-write-here"
        setup_logging(component="t", console=False)     # must not raise
        get_logger("cvti.test").info("still running")

    def test_a_detector_exception_produces_a_record_with_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = setup_logging(tmp, component="t", console=False)
            log = get_logger("cvti.detector.fake")
            try:
                raise ValueError("detector blew up mid-frame")
            except ValueError:
                log.exception("detector failed")
            logging.shutdown()
            body = path.read_text()
            self.assertIn("detector failed", body)
            self.assertIn("ValueError", body)
            self.assertIn("detector blew up mid-frame", body)
            self.assertIn("cvti.detector.fake", body)      # attribution


class DiagnosticsBundleTest(unittest.TestCase):
    def setUp(self):
        reset_for_tests()
        os.environ.pop(DIR_ENV, None)

    def tearDown(self):
        reset_for_tests()
        os.environ.pop(DIR_ENV, None)

    def _populated(self, tmp: str) -> Path:
        """An output dir shaped like a real one: logs, evidence, a database."""
        out = Path(tmp)
        setup_logging(out, component="argus-engine", console=False)
        get_logger("cvti.test").info("a log line worth shipping")
        logging.shutdown()

        evidence = out / "events" / "20260819_cam1_theft"
        evidence.mkdir(parents=True)
        (evidence / "frame_01.jpg").write_bytes(b"\xff\xd8not-a-real-jpeg")
        (evidence / "clip.mp4").write_bytes(b"not-a-real-mp4")

        import sqlite3
        con = sqlite3.connect(out / "events.db")
        con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts REAL, review TEXT, "
                    "camera_id TEXT, reason TEXT)")
        con.execute("INSERT INTO events (ts, review, camera_id, reason) VALUES "
                    "(1, NULL, 'aisle_1', 'a man in a red coat concealed a bottle')")
        con.commit()
        con.close()
        (out / "gate_health.json").write_text(json.dumps({"errors": 0, "mock": False}))
        return out

    def test_bundle_contains_logs_and_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._populated(tmp)
            names = zipfile.ZipFile(diagnostics.build_bundle(out)).namelist()
            self.assertIn("health.json", names)
            self.assertIn("MANIFEST.txt", names)
            self.assertTrue(any(n.startswith("logs/") for n in names), names)

    def test_bundle_contains_no_images_video_or_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._populated(tmp)
            zf = zipfile.ZipFile(diagnostics.build_bundle(out))
            for name in zf.namelist():
                self.assertFalse(name.lower().endswith((".jpg", ".jpeg", ".png", ".mp4",
                                                        ".avi", ".mov", ".db")),
                                 f"personal data leaked into the bundle: {name}")

    def test_no_event_text_leaks_through_the_health_snapshot(self):
        # Counts are fine. The reason field describes a person, and must not appear.
        with tempfile.TemporaryDirectory() as tmp:
            out = self._populated(tmp)
            zf = zipfile.ZipFile(diagnostics.build_bundle(out))
            blob = b"".join(zf.read(n) for n in zf.namelist())
            self.assertNotIn(b"red coat", blob)
            self.assertNotIn(b"aisle_1", blob)

    def test_health_snapshot_reports_counts_not_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._populated(tmp)
            health = json.loads(zipfile.ZipFile(diagnostics.build_bundle(out)).read("health.json"))
            self.assertEqual(health["events"]["events_total"], 1)
            self.assertEqual(health["events"]["events_unreviewed"], 1)
            self.assertIn("platform", health)
            self.assertIn("disk", health)

    def test_manifest_states_what_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._populated(tmp)
            manifest = zipfile.ZipFile(diagnostics.build_bundle(out)).read("MANIFEST.txt").decode()
            self.assertIn("DOES NOT CONTAIN", manifest)
            self.assertIn("evidence frames", manifest)

    def test_bundle_builds_on_a_bare_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = diagnostics.build_bundle(tmp)      # no logs, no db, no health
            self.assertTrue(path.exists())
            self.assertIn("MANIFEST.txt", zipfile.ZipFile(path).namelist())


if __name__ == "__main__":
    unittest.main()
