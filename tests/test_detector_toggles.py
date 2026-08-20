"""Detector toggles: the Rules screen must be able to switch on EVERY detector the
engine supports, persist it, and seed sensible tuning params.

Guards the gap this fixes: the UI used to offer only 6 of the engine's 10
detectors, so fire/panic/crowd/fall were unreachable without editing JSON.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.app.console_backend import ConsoleBackend

from _backend_helper import signed_in


def _engine_flags() -> set[str]:
    """The per-camera boolean detector flags PerCameraState actually supports."""
    src = (ROOT / "cvti" / "serving" / "camera.py").read_text()
    return set(re.findall(r"^    ([a-z_]+): bool = False", src, re.M))


def _ui_chips() -> set[str]:
    """Detector keys the Rules screen renders as toggle chips."""
    html = (ROOT / "cvti" / "app" / "web" / "index.html").read_text()
    return set(re.findall(r'\["([a-z_]+)","[^"]+","[^"]+",(?:true|false),"(?:security|safety)"', html))


class ToggleCoverageTests(unittest.TestCase):
    def test_ui_backend_and_engine_agree(self):
        eng, ui, be = _engine_flags(), _ui_chips(), set(ConsoleBackend.RULE_FLAGS)
        self.assertTrue(eng, "should find engine flags")
        self.assertEqual(ui, be, "every UI chip must be accepted by the backend")
        self.assertEqual(be, eng, "operator must be able to toggle every engine detector")

    def test_the_previously_missing_detectors_are_present(self):
        for key in ("fire_smoke", "running", "crowd_formation", "fall"):
            self.assertIn(key, ConsoleBackend.RULE_FLAGS)
            self.assertIn(key, _ui_chips())


class TogglePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.site = self.d / "site.json"
        self.site.write_text(json.dumps({
            "name": "T", "notify": "console", "configured": True,
            "cameras": [{"id": "cam1", "source": "x.mp4", "config": "configs/all_threats_v1.json"}]}))
        self.be = signed_in(site_path=str(self.site), db_path=str(self.d / "e.db"),
                                 enable_demo=False)

    def _cam(self):
        return json.loads(self.site.read_text())["cameras"][0]

    def test_toggle_on_persists_for_every_detector(self):
        for key in ConsoleBackend.RULE_FLAGS:
            self.be.set_camera_rules("cam1", {key: True})
            self.assertTrue(self._cam().get(key), f"{key} should persist as on")

    def test_toggle_off_persists(self):
        self.be.set_camera_rules("cam1", {"fire_smoke": True})
        self.be.set_camera_rules("cam1", {"fire_smoke": False})
        self.assertFalse(self._cam().get("fire_smoke"))

    def test_enabling_seeds_tuning_defaults(self):
        self.be.set_camera_rules("cam1", {"crowd_formation": True, "running": True})
        cam = self._cam()
        self.assertEqual(cam.get("crowd_min_people"), 5)
        self.assertEqual(cam.get("running_min_speed_ratio"), 0.08)

    def test_defaults_never_clobber_operator_values(self):
        self.be.set_camera_rules("cam1", {"running": True})
        cams = json.loads(self.site.read_text())
        cams["cameras"][0]["running_min_speed_ratio"] = 0.25      # operator tuned it
        self.site.write_text(json.dumps(cams))
        self.be.set_camera_rules("cam1", {"running": False})
        self.be.set_camera_rules("cam1", {"running": True})       # re-enable
        self.assertEqual(self._cam()["running_min_speed_ratio"], 0.25)

    def test_toggled_camera_builds_an_engine_state_with_the_flag(self):
        """The toggle must actually reach the engine's per-camera state."""
        from cvti.serving.camera import PerCameraState
        self.be.set_camera_rules("cam1", {"fire_smoke": True, "fall": True})
        cam = self._cam()
        for key in ("fire_smoke", "fall"):
            self.assertTrue(hasattr(PerCameraState, "__dataclass_fields__"))
            self.assertIn(key, PerCameraState.__dataclass_fields__,
                          f"engine has no field for toggled detector {key}")
            self.assertTrue(cam.get(key))


if __name__ == "__main__":
    unittest.main()
