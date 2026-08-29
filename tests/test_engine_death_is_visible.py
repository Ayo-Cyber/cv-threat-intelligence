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


class HeartbeatDecidesTest(unittest.TestCase):
    """'Noo.. it's saying not monitoring' (29 Aug): three status strings from
    three sources of truth — a PID, a heartbeat file, an optimistic button —
    all allowed to disagree on one screen. The heartbeat decides now: a live
    process that stopped writing gate_health.json is 'stalled', because a PID
    is not monitoring."""

    def _alive_backend(self, tmp):
        cb = _backend(tmp)
        cb._monitor = subprocess.Popen([sys.executable, "-c",
                                        "import time; time.sleep(30)"])
        return cb

    def test_a_live_pid_with_no_heartbeat_is_stalled(self):
        import time as _t
        with tempfile.TemporaryDirectory() as tmp:
            cb = self._alive_backend(tmp)
            try:
                cb._engine_started_at = _t.time() - 300     # long past warm-up
                s = cb.monitoring_status()
                self.assertTrue(s["running"])
                self.assertTrue(s.get("stalled"), "a hung engine passed as healthy")
            finally:
                cb._monitor.kill()

    def test_a_fresh_heartbeat_is_healthy(self):
        import json as _json
        import time as _t
        with tempfile.TemporaryDirectory() as tmp:
            cb = self._alive_backend(tmp)
            try:
                cb._engine_started_at = _t.time() - 300
                (Path(tmp) / "site" / "gate_health.json").write_text(
                    _json.dumps({"generated_at": _t.time()}))
                s = cb.monitoring_status()
                self.assertNotIn("stalled", s)
                self.assertLess(s["heartbeat_age_s"], 5)
            finally:
                cb._monitor.kill()

    def test_model_loading_gets_a_grace_period(self):
        import time as _t
        with tempfile.TemporaryDirectory() as tmp:
            cb = self._alive_backend(tmp)
            try:
                cb._engine_started_at = _t.time() - 10      # just spawned
                s = cb.monitoring_status()
                self.assertNotIn("stalled", s, "flagged during model warm-up")
            finally:
                cb._monitor.kill()

    def test_the_ui_names_the_stalled_state(self):
        html = Path("cvti/app/web/index.html").read_text()
        self.assertIn("unresponsive", html)
        self.assertIn("no heartbeat", html)


class TrueSightStatusIsRealTest(unittest.TestCase):
    """'The custom english is not detecting anything' (pilot, 29 Aug) — while
    the footer said 'TrueSight · live'. The string was set by a 25-second
    TIMER after clicking Start, model or no model; meanwhile the gate stamped
    every alert UNVERIFIED and the English rules (which ARE the VLM) scanned
    nothing. The footer now derives from the gate's real health, and the
    rules panel says in place when its rules are not running."""

    def setUp(self):
        self.html = Path("cvti/app/web/index.html").read_text()

    def test_only_the_health_poller_declares_the_ai_alive(self):
        """The 25s setTimeout that stamped 'TrueSight · live' regardless of
        the model's existence is gone; the string may only be written from
        pollGateHealth, where it derives from gate.reachable."""
        poller = self.html.split("function pollGateHealth()")[1].split("function ")[0]
        total = self.html.count("TrueSight · live")
        in_poller = poller.count("TrueSight · live")
        self.assertGreaterEqual(in_poller, 1, "the poller no longer reports live state")
        self.assertEqual(total, in_poller,
                         "something outside the health poller declares TrueSight live")
        toggle = self.html.split("function toggleMonitor()")[1].split("function ")[0]
        self.assertNotIn("TrueSight · live", toggle, "the start-button timer is back")

    def test_unreachable_gate_names_the_consequences(self):
        self.assertIn("alerts arrive unverified; English rules are paused", self.html)

    def test_the_rules_panel_warns_in_place(self):
        self.assertIn("These rules are not running:", self.html)
        self.assertIn("state.gateDown", self.html)


class SlowMachinesAndMissingModelsTest(unittest.TestCase):
    """Pilot health panel, 29 Aug: verify median 90.2s, detector.cam1 failing
    100% (509 errors/28min), 'weapons detector configured but its model failed
    to load: No module named hubconf'. Three faults, pinned together."""

    def test_the_local_vlm_call_outlives_a_slow_verify(self):
        """A timeout at the measured MEDIAN fails half of all calls — the
        English rules rode a 90s timeout on a 90.2s-median machine."""
        src = Path("cvti/scene/agent_mapper.py").read_text()
        import re
        m = re.search(r"timeout=(\d+)\) as response:\n\s+parsed = json", src)
        self.assertIsNotNone(m)
        self.assertGreaterEqual(int(m.group(1)), 150)

    def test_a_failed_weapon_model_disables_the_flag_it_serves(self):
        src = Path("cvti/serving/pipeline.py").read_text()
        block = src.split("weapons detector configured but its model failed")[1][:700]
        self.assertIn('c["weapons"] = False', block,
                      "cameras keep a weapons flag with a None model — one throw per frame")

    def test_the_vendored_yolov5_ships(self):
        spec = Path("packaging/argus.spec").read_text()
        self.assertIn('_tree("external/yolov5"', spec,
                      "the weapon detector's hub repo is not in the bundle")
        import subprocess
        r = subprocess.run(["git", "ls-files", "external/yolov5/hubconf.py"],
                           capture_output=True, text=True)
        self.assertIn("hubconf.py", r.stdout, "the repo the spec bundles is not tracked")
