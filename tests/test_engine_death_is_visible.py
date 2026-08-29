"""'It doesn't work on my machine' must arrive as an error, not a screenshot.

A pilot's Windows machine spent a day dark (29 Aug field photos): the engine
died at startup, the wall showed broken tiles, the Start button flipped to
'Stop monitoring' over a footer saying 'engine not running', and nothing
anywhere said WHY. Three fixes pinned here: feed data directories are
writable on installs, a dead engine's status carries its exit code + the
log's last telling line + the log path, and the UI reconciles its optimism
with the polled truth.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvti.app.console_backend import ConsoleBackend


def _backend(tmp):
    site = Path(tmp) / "site"
    site.mkdir(exist_ok=True)
    return ConsoleBackend(site_path=str(site / "site.json"),
                          db_path=str(site / "events.db"), enable_demo=False)


class DeadEngineSpeaksTest(unittest.TestCase):

    def test_status_carries_exit_code_reason_and_log_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            (Path(tmp) / "site" / "monitor.log").write_text(
                "starting up\nPermissionError: [Errno 13] Permission denied: 'runs/feeds'\n")
            cb._monitor = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
            cb._monitor.wait()
            s = cb.monitoring_status()
            self.assertFalse(s["running"])
            self.assertEqual(s["exit_code"], 3)
            self.assertIn("Permission denied", s["last_error"])
            self.assertTrue(s["log_path"].endswith("monitor.log"))

    def test_a_running_engine_reports_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            cb._monitor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                s = cb.monitoring_status()
                self.assertTrue(s["running"])
                self.assertNotIn("exit_code", s)
            finally:
                cb._monitor.kill()


class FeedDirsAreWritableOnInstallsTest(unittest.TestCase):

    def test_frozen_feed_dirs_live_in_the_user_data_dir(self):
        """Path('runs/feeds') on an install is Program Files — the engine's
        first mkdir was a PermissionError and it died before its first frame."""
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            with mock.patch.object(sys, "frozen", create=True, new=True), \
                 mock.patch("cvti.utils.user_data_dir", return_value=Path(tmp) / "userdata"):
                p = cb._db_for_feed("demo", str(Path(tmp) / "elsewhere.json"))
            self.assertTrue(p.startswith(str(Path(tmp) / "userdata")),
                            f"feed store landed at {p} — inside the install dir on a real machine")


class UiShowsTheReasonTest(unittest.TestCase):

    def setUp(self):
        self.html = Path("cvti/app/web/index.html").read_text()

    def test_the_footer_and_wall_surface_the_failure(self):
        self.assertIn("engine exited (code ", self.html)
        self.assertIn("The engine failed to start", self.html)
        self.assertIn("state.engineFail", self.html)

    def test_the_start_button_reconciles_with_the_polled_truth(self):
        start_branch = self.html.split('call("startMonitoring"')[1][:600]
        self.assertIn("setTimeout(refreshMonitor", start_branch,
                      "the optimistic flip is never checked against reality")


if __name__ == "__main__":
    unittest.main()


class ZoneSavesWorkOnInstallsTest(unittest.TestCase):
    """Saving a zone on the pilot's Windows install died with
    [WinError 5] Access is denied: 'configs' (field screenshot, 29 Aug):
    the regenerated per-camera rules file targeted the bundle directory, and
    the base preset was READ relatively — silently yielding zero baseline
    rules on any machine where the cwd is not this repo."""

    def test_frozen_rule_files_land_in_the_user_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            cam = {"id": "cam1"}
            with mock.patch.object(sys, "frozen", create=True, new=True), \
                 mock.patch("cvti.utils.user_data_dir", return_value=Path(tmp) / "userdata"):
                cb._regen_zone_rules("cam1", cam, [{"name": "entrance",
                                                    "dwell_alert_seconds": 16}])
            self.assertTrue(cam["config"].startswith(str(Path(tmp) / "userdata")),
                            f"rules file at {cam['config']} — the install dir on a real machine")
            rules = json.loads(Path(cam["config"]).read_text())["rules"]
            self.assertTrue(any(r["name"] == "loitering_entrance" for r in rules))

    def test_the_base_preset_survives_a_foreign_cwd(self):
        """The preset read must resolve via resource_path when the relative
        path misses — otherwise the generated file silently loses every
        baseline rule."""
        import os
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            cam = {"id": "cam1", "config": "configs/all_threats_v1.json"}
            cwd = os.getcwd()
            os.chdir(tmp)                       # anywhere but the repo
            try:
                cb._regen_zone_rules("cam1", cam, [{"name": "entrance",
                                                    "dwell_alert_seconds": 16}])
                rules = json.loads(Path(cam["config"]).read_text())["rules"]
            finally:
                os.chdir(cwd)
            base = [r for r in rules if not r["name"].startswith("loitering_")]
            self.assertTrue(base, "the baseline preset vanished from the generated rules")


class CrashLoopIsVisibleTest(unittest.TestCase):
    """A crash LOOP hid from the dead-engine check: the watchdog respawns
    faster than the UI polls, so almost every sample lands on a just-born
    process that looks alive. Pilot screenshot (29 Aug): 'Stop monitoring'
    over an engine that had died repeatedly, with the promised reason never
    shown. Restart churn is a first-class state now."""

    def test_a_live_but_looping_engine_reports_the_loop(self):
        import time as _t
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            (Path(tmp) / "site" / "monitor.log").write_text(
                "boot\nRuntimeError: model file missing\n")
            cb._monitor = subprocess.Popen([sys.executable, "-c",
                                            "import time; time.sleep(30)"])
            try:
                cb._restarts = 3
                cb._last_exit_code = 1
                cb._last_death_at = _t.time()
                s = cb.monitoring_status()
                self.assertTrue(s["running"], "the respawned process IS alive")
                self.assertTrue(s["crash_looping"])
                self.assertEqual(s["restarts"], 3)
                self.assertIn("model file missing", s["last_error"])
            finally:
                cb._monitor.kill()

    def test_a_stable_engine_reports_no_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            cb._monitor = subprocess.Popen([sys.executable, "-c",
                                            "import time; time.sleep(30)"])
            try:
                s = cb.monitoring_status()
                self.assertTrue(s["running"])
                self.assertNotIn("crash_looping", s)
            finally:
                cb._monitor.kill()

    def test_the_wall_banner_says_nothing_is_recorded(self):
        html = Path("cvti/app/web/index.html").read_text()
        self.assertIn("crash-looping", html)
        self.assertIn("nothing is being detected or recorded", html)
