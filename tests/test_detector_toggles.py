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


class MovementConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def _zones(self, *names):
        path = self.directory / "zones.json"
        path.write_text(json.dumps({"zones": [
            {"name": name, "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]}
            for name in names
        ]}))
        return str(path)

    def _site(self, **camera):
        return {"cameras": [{
            "id": "chi_gate",
            "source": "clip.mp4",
            "config": "configs/chi_pilot_v1.json",
            **camera,
        }]}

    def test_movement_configuration_reaches_camera_state(self):
        from cvti.serving.camera import build_camera_states

        state = build_camera_states(self._site(
            normal_movement=True,
            multiple_people_moving=True,
            movement_enter_speed_ratio=0.07,
            movement_exit_speed_ratio=0.03,
            movement_min_track_seconds=0.6,
            movement_min_people=3,
            movement_persistence_seconds=0.8,
            permitted_movement_zones=["gate", "yard"],
            zones=self._zones("gate", "yard"),
        ))["chi_gate"]["state"]

        self.assertTrue(state.normal_movement)
        self.assertTrue(state.multiple_people_moving)
        self.assertEqual(state.movement_enter_speed_ratio, 0.07)
        self.assertEqual(state.movement_exit_speed_ratio, 0.03)
        self.assertEqual(state.movement_min_track_seconds, 0.6)
        self.assertEqual(state.movement_min_people, 3)
        self.assertEqual(state.movement_persistence_seconds, 0.8)
        self.assertEqual(state.permitted_movement_zones, ("gate", "yard"))

    def test_invalid_movement_thresholds_name_the_camera(self):
        from cvti.serving.camera import build_camera_states

        invalid = (
            {"movement_enter_speed_ratio": "fast"},
            {"movement_enter_speed_ratio": float("nan")},
            {"movement_exit_speed_ratio": float("inf")},
            {"movement_min_track_seconds": float("-inf")},
            {"movement_persistence_seconds": float("nan")},
            {"movement_enter_speed_ratio": 0.0},
            {"movement_exit_speed_ratio": 0.0},
            {"movement_enter_speed_ratio": 0.03, "movement_exit_speed_ratio": 0.03},
            {"movement_min_track_seconds": 0.0},
            {"movement_min_people": 1},
            {"movement_min_people": True},
            {"movement_min_people": 2.9},
            {"movement_persistence_seconds": 0.0},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "chi_gate"):
                    build_camera_states(self._site(multiple_people_moving=True, **values))

    def test_float_movement_settings_reject_booleans_before_conversion(self):
        from cvti.serving.camera import build_camera_states

        fields = (
            "movement_enter_speed_ratio",
            "movement_exit_speed_ratio",
            "movement_min_track_seconds",
            "movement_persistence_seconds",
        )
        for field in fields:
            for value in (True, False):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"camera chi_gate: {field} must be a number, not boolean",
                    ):
                        build_camera_states(self._site(**{field: value}))

    def test_detector_feature_flags_reject_strings_and_numbers(self):
        from cvti.serving.camera import build_camera_states

        flags = (
            "concealment", "violence", "weapons", "theft", "tamper", "fall",
            "fire_smoke", "running", "crowd_formation", "normal_movement",
            "multiple_people_moving", "video_action",
        )
        for flag in flags:
            for value in ("false", "true", 0, 1, 0.0, 1.0):
                with self.subTest(flag=flag, value=value):
                    with self.assertRaisesRegex(
                        ValueError, f"camera chi_gate: {flag} must be a boolean"
                    ):
                        build_camera_states(self._site(**{flag: value}))

    def test_permitted_movement_zones_require_a_string_sequence(self):
        from cvti.serving.camera import build_camera_states

        invalid = ("gate", 7, ["gate", ""], ["gate", 7])
        for permitted in invalid:
            with self.subTest(permitted=permitted):
                with self.assertRaisesRegex(ValueError, "chi_gate"):
                    build_camera_states(self._site(
                        zones=self._zones("gate"),
                        permitted_movement_zones=permitted,
                    ))

    def test_permitted_movement_zones_require_a_zone_config(self):
        from cvti.serving.camera import build_camera_states

        with self.assertRaisesRegex(ValueError, "chi_gate"):
            build_camera_states(self._site(permitted_movement_zones=["gate"]))

    def test_permitted_movement_zones_must_name_configured_zones(self):
        from cvti.serving.camera import build_camera_states

        with self.assertRaisesRegex(ValueError, "chi_gate"):
            build_camera_states(self._site(
                zones=self._zones("gate"),
                permitted_movement_zones=["yard"],
            ))


if __name__ == "__main__":
    unittest.main()
