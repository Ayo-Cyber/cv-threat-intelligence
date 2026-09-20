"""A camera added through the UI must not kill the engine.

The UI's add_camera writes {id, source, area_id}. The rules `config` is set
later, by apply_template on the wizard's "Use case" step — which the wizard
reaches only AFTER cameras. So an operator who adds a camera and presses Start
before choosing a use case used to take the whole engine down instantly:

    KeyError: 'config'   (cvti/serving/camera.py, build_camera_states)

Nothing said so. The API's /monitor derived state from heartbeat freshness
alone, so a dead engine read `phase: "stopped"` exactly like one the operator
had stopped on purpose, and /cameras/{id}/stream answered "no live stream —
monitoring is stopped". The field report was "the camera connects but we
cannot see the streams" (20 Sep).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cvti.api.sources import engine_log_tail, monitor_state
from cvti.serving.camera import DEFAULT_RULES_CONFIG, rules_config_for


class ACameraWithNoUseCaseChosenYet(unittest.TestCase):
    def test_it_falls_back_to_the_default_rule_set(self):
        camera = {"id": "bay1", "source": "rtsp://host/stream"}
        self.assertEqual(rules_config_for(camera), DEFAULT_RULES_CONFIG)

    def test_an_applied_template_still_wins(self):
        camera = {"id": "bay1", "source": "rtsp://host/stream",
                  "config": "configs/all_threats_video_v1.json"}
        self.assertEqual(rules_config_for(camera),
                         "configs/all_threats_video_v1.json")

    def test_an_empty_config_is_not_a_config(self):
        for empty in ("", None):
            with self.subTest(config=empty):
                camera = {"id": "bay1", "source": "x", "config": empty}
                self.assertEqual(rules_config_for(camera), DEFAULT_RULES_CONFIG)

    def test_the_default_rule_set_actually_exists(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / DEFAULT_RULES_CONFIG).is_file(),
                        f"{DEFAULT_RULES_CONFIG} is the fallback every "
                        "unconfigured camera loads; it must ship")


class AStoppedEngineSaysWhy(unittest.TestCase):
    def _site(self, log: str):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        db = root / "events.db"
        db.write_bytes(b"")
        (root / "monitor.log").write_text(log)
        return str(db)

    def test_a_crash_reaches_the_operator(self):
        db = self._site(
            "INFO cvti.logging_setup - logging started\n"
            "Traceback (most recent call last):\n"
            "KeyError: 'config'\n")
        state = monitor_state(db)
        self.assertFalse(state["running"])
        self.assertEqual(state["last_error"], "KeyError: 'config'",
                         "a stopped engine must carry the reason, not just "
                         "the fact")
        self.assertTrue(state["log_path"].endswith("monitor.log"))

    def test_it_prefers_the_telling_line_over_the_last_line(self):
        db = self._site(
            "Traceback (most recent call last):\n"
            "KeyError: 'config'\n"
            "INFO shutting down cleanly\n")
        self.assertEqual(monitor_state(db)["last_error"], "KeyError: 'config'")

    def test_a_missing_log_is_not_a_crash(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db = str(Path(self.tmp.name) / "events.db")
        _, last = engine_log_tail(db)
        self.assertEqual(last, "")
        self.assertEqual(monitor_state(db)["last_error"], "")

    def test_a_running_engine_carries_no_error(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        db = root / "events.db"
        db.write_bytes(b"")
        import time
        (root / "gate_health.json").write_text(json.dumps({
            "generated_at": time.time(), "engine": {"phase": "monitoring"}}))
        (root / "monitor.log").write_text("Traceback (most recent call last):\n")
        state = monitor_state(str(db))
        self.assertTrue(state["running"])
        self.assertNotIn("last_error", state,
                         "a healthy engine must not wave an old traceback")


if __name__ == "__main__":
    unittest.main()
