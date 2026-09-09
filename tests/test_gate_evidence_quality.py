"""The evidence upgrade: the judge gets a zoomed subject and the detector's cues.

The scorecard's diagnosis (manifest e56b4277): the gate confirms only 28% of
true theft candidates, and its own rejections read "no people are visible" on
clips where the tracker was following someone — a person in full-frame CCTV is
often under 50px tall. The upgrade sends (a) a zoomed crop of the flagged
subject appended after the full frames, (b) the detector's measured reasons in
the prompt, and (c) prompt wording that tells the model what the final image
is. These tests hold that the crop is built safely (never costing an alert)
and that the prompt actually carries the new evidence.
"""
import unittest
from unittest import mock

import numpy as np

from cvti.contracts import CandidateAlert
from cvti.verification.frame_select import append_subject_crop, subject_crop
from cvti.verification.gate import VerificationGate


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _alert(reasons=()):
    return CandidateAlert(
        rule_name="video_theft_candidate", priority="high",
        detector="video_action", title="VIDEO ACTION: theft",
        person_id=3, object_label=None, timestamp=0.0,
        reasons=list(reasons))


class SubjectCropTest(unittest.TestCase):
    def test_a_small_person_box_is_upscaled_to_real_pixels(self):
        crop = subject_crop(_frame(), (300, 200, 330, 260))   # 30x60 person
        self.assertIsNotNone(crop)
        self.assertGreaterEqual(min(crop.shape[:2]), 320)

    def test_the_margin_adds_context_around_the_box(self):
        # A 100x100 box with 40% margin -> ~180x180 region before upscale.
        marked = _frame()
        marked[195:205, 195:205] = 255          # a patch just OUTSIDE the box
        crop = subject_crop(marked, (200, 200, 300, 300), min_side=0)
        self.assertIsNotNone(crop)
        self.assertGreater(int(crop.max()), 0, "margin context was cut off")

    def test_a_box_at_the_frame_edge_is_clamped_not_crashed(self):
        crop = subject_crop(_frame(), (-20, -20, 60, 100))
        self.assertIsNotNone(crop)

    def test_a_degenerate_box_returns_none(self):
        self.assertIsNone(subject_crop(_frame(), (100, 100, 101, 100)))

    def test_append_keeps_full_frames_first_and_crop_last(self):
        frames = [_frame(), _frame()]
        out = append_subject_crop(frames, _frame(), (300, 200, 330, 260))
        self.assertEqual(len(out), 3)
        self.assertIs(out[0], frames[0])        # context frames untouched
        self.assertGreaterEqual(min(out[-1].shape[:2]), 320)

    def test_no_bbox_means_no_crop_and_no_error(self):
        frames = [_frame()]
        self.assertIs(append_subject_crop(frames, _frame(), None), frames)


class ThePromptCarriesTheEvidence(unittest.TestCase):
    def _prompt_for(self, alert, *, cot=False):
        g = VerificationGate(provider="ollama", cot=cot)
        seen = {}

        def fake_provider(prompt, frames_bytes, a):
            seen["prompt"] = prompt
            seen["n_images"] = len(frames_bytes)
            return ('{"confirmed": false, "confidence": 0.5, '
                    '"reason": "test", "alert_priority": "high"}')

        with mock.patch.object(g, "_call_provider", side_effect=fake_provider):
            g.verify([_frame()], alert, {})
        return seen

    def test_detector_reasons_reach_the_judge(self):
        seen = self._prompt_for(_alert(["clip score 0.81 over 12 frames",
                                        "dwell 9s at shelf"]))
        self.assertIn("clip score 0.81 over 12 frames; dwell 9s at shelf",
                      seen["prompt"])

    def test_missing_reasons_say_so_instead_of_formatting_garbage(self):
        seen = self._prompt_for(_alert([]))
        self.assertIn("nothing recorded", seen["prompt"])

    def test_both_templates_explain_the_final_crop_image(self):
        for cot in (False, True):
            seen = self._prompt_for(_alert(), cot=cot)
            self.assertIn("zoomed crop", seen["prompt"],
                          f"cot={cot} template lost the crop instruction")

    def test_a_multi_frame_list_is_sent_as_multiple_images(self):
        g = VerificationGate(provider="ollama")
        seen = {}

        def fake_provider(prompt, frames_bytes, a):
            seen["n"] = len(frames_bytes)
            return ('{"confirmed": true, "confidence": 0.9, '
                    '"reason": "t", "alert_priority": "high"}')

        evidence = append_subject_crop([_frame(), _frame()], _frame(),
                                       (300, 200, 330, 260))
        with mock.patch.object(g, "_call_provider", side_effect=fake_provider):
            g.verify(evidence, _alert(), {})
        self.assertEqual(seen["n"], 3, "the crop must travel as its own image")


if __name__ == "__main__":
    unittest.main()
