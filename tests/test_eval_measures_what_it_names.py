"""A run named for one threat must exercise the detector that makes that claim.

27 Aug: `python -m cvti.eval --kind weapons --gate ollama` completed 100 clips
and produced a clean report — "recall 98.0% -> 16.0%". The number was real and
it described the wrong thing: --kind selected the CLIPS but not the DETECTORS,
so the run measured the shoplifting/concealment detector's opinion of UCF-Crime
shooting footage. Every saved verification says "the person is holding a box of
products" or "there is no evidence of concealment" — the weapons detector was
never loaded. That figure was one paste away from NUMBERS.md as a weapons row.

The second half of the same bug: the gate was told every clip was "a shop
interior monitored for theft", including street shootings, which leads the
model away from what is being measured.
"""
from __future__ import annotations

import unittest

from cvti.eval.__main__ import _KIND_DETECTORS
from cvti.eval.dataset import _KIND_EXPECTS, EvalClip
from cvti.eval.harness import _scene_context_for


class KindSelectsItsDetectorTest(unittest.TestCase):

    def test_every_measurable_kind_maps_to_a_detector(self):
        for kind in _KIND_EXPECTS:
            self.assertIn(kind, _KIND_DETECTORS,
                          f"--kind {kind} would silently run the default detectors")

    def test_weapons_does_not_run_the_theft_detector(self):
        self.assertEqual(_KIND_DETECTORS["weapons"], "weapons")
        self.assertNotIn("concealment", _KIND_DETECTORS["weapons"])

    def test_violence_does_not_run_the_theft_detector(self):
        self.assertEqual(_KIND_DETECTORS["violence"], "violence")
        self.assertNotIn("concealment", _KIND_DETECTORS["violence"])

    def test_the_mapped_detector_is_one_the_camera_state_accepts(self):
        """A typo here disables the detector silently — the flag is just an
        unused kwarg name, and the run scores 0% against nothing."""
        from cvti.serving.camera import PerCameraState
        fields = set(PerCameraState.__dataclass_fields__)
        for kind, detectors in _KIND_DETECTORS.items():
            for d in detectors.split(","):
                self.assertIn(d, fields,
                              f"--kind {kind} enables '{d}', which PerCameraState "
                              f"has no flag for — it would never fire")


class SceneContextFollowsTheFootageTest(unittest.TestCase):

    def test_street_footage_is_not_introduced_as_a_shop(self):
        clip = EvalClip("data/ucf_crime/Shooting/Shooting014.mp4", True,
                        "weapons", "ucf-crime/Shooting")
        ctx = _scene_context_for(clip)
        self.assertNotEqual(ctx["environment_type"], "retail")
        self.assertNotIn("theft", ctx["scene_description"].lower(),
                         "the gate is being pointed at the wrong threat")

    def test_shop_footage_still_says_shop(self):
        for clip in (EvalClip("x/theft_shop_01.mp4", True, "theft", "local-clips"),
                     EvalClip("y/404.mp4", True, "theft", "camnuvem-test")):
            self.assertEqual(_scene_context_for(clip)["environment_type"], "retail")

    def test_the_context_never_names_the_answer(self):
        """Telling the model what to find is not measurement."""
        for clip in (EvalClip("a/Shooting014.mp4", True, "weapons", "ucf-crime/Shooting"),
                     EvalClip("b/Fighting002.mp4", True, "violence", "ucf-crime/Fighting")):
            text = " ".join(_scene_context_for(clip).values()).lower()
            for leak in ("weapon", "gun", "firearm", "shooting", "violence", "fight", "assault"):
                self.assertNotIn(leak, text, f"scene context leaks '{leak}' to the gate")


if __name__ == "__main__":
    unittest.main()
