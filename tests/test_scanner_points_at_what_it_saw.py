"""The English-rule scanner answers carefully and points at what it saw (3 Sep).

Two operator asks in one contract:

- 'not flagging things that aren't it': the prompt demands concrete visible
  evidence, names near-matches as the failure mode ('a dark jacket is not a
  hoodie'), frames the empty list as the normal answer — and a claim below
  the confidence floor is dropped, because a hesitant yes is a no;
- 'flag the instrument, person, object': each claim carries a target kind and
  a 0-1000 box; the evidence frame ships with that box drawn colour-coded
  (person blue, object amber, instrument red) and the pixel bbox rides in the
  alert payload so the subject shot points at the same thing. A malformed box
  is dropped, never drawn — a fabricated box over evidence is worse than none.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.custom_rules import (MIN_RULE_CONFIDENCE, TARGET_COLOURS,
                                       CustomRuleScanner, _normalized_box,
                                       annotate_hit)

CAM = {"id": "front", "source": "x",
       "custom_threats": [{"name": "hoodie", "description": "someone wearing a hoodie"}]}


def _scan_with(answer: str) -> list[dict]:
    scanner = CustomRuleScanner([CAM], sink=None, model="gemma3:4b")
    with mock.patch("cvti.scene.agent_mapper.call_openai_compatible",
                    return_value=answer):
        return scanner._check(CAM, np.zeros((100, 200, 3), dtype=np.uint8))


class ConfidenceFloorTests(unittest.TestCase):
    def test_a_hesitant_yes_is_a_no(self):
        hits = _scan_with('{"threats": [{"name": "hoodie", "reason": "maybe a dark '
                          'jacket", "confidence": 0.4}]}')
        self.assertEqual(hits, [])

    def test_a_confident_sighting_fires(self):
        hits = _scan_with('{"threats": [{"name": "hoodie", "reason": "grey hood up '
                          'over the head", "confidence": 0.92}]}')
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["confidence"], 0.92)

    def test_a_noncompliant_answer_without_confidence_still_fires(self):
        """The floor tightens compliant answers; it must not strand a model
        that free-styles the old shape."""
        hits = _scan_with('{"threats": [{"name": "hoodie", "reason": "hood up"}]}')
        self.assertEqual(len(hits), 1)

    def test_the_floor_is_a_deliberate_number(self):
        self.assertEqual(MIN_RULE_CONFIDENCE, 0.7)

    def test_the_prompt_demands_evidence_and_normalizes_none(self):
        captured = {}

        def wire(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return '{"threats": []}'

        scanner = CustomRuleScanner([CAM], sink=None, model="gemma3:4b")
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible", wire):
            scanner._check(CAM, np.zeros((64, 64, 3), dtype=np.uint8))
        prompt = captured["prompt"]
        self.assertIn("a dark jacket is not a hoodie", prompt)
        self.assertIn("empty list is the normal answer", prompt)
        self.assertIn('"confidence"', prompt)
        self.assertIn('"box"', prompt)
        self.assertIn('"target"', prompt)


class BoxParsingTests(unittest.TestCase):
    def test_a_valid_box_and_target_ride_the_hit(self):
        hits = _scan_with('{"threats": [{"name": "hoodie", "reason": "hood up", '
                          '"confidence": 0.9, "target": "person", '
                          '"box": [100, 50, 400, 900]}]}')
        self.assertEqual(hits[0]["box"], (100.0, 50.0, 400.0, 900.0))
        self.assertEqual(hits[0]["target"], "person")

    def test_malformed_boxes_are_dropped_not_drawn(self):
        for bad in ("[1,2,3]", "[-5, 0, 100, 100]", "[0, 0, 2000, 100]",
                    '"everywhere"', "[100, 100, 102, 101]"):
            hits = _scan_with('{"threats": [{"name": "hoodie", "reason": "hood", '
                              f'"confidence": 0.9, "box": {bad}}}]}}')
            self.assertEqual(len(hits), 1, bad)
            self.assertNotIn("box", hits[0], f"malformed box survived: {bad}")

    def test_an_unknown_target_becomes_object(self):
        hits = _scan_with('{"threats": [{"name": "hoodie", "reason": "hood", '
                          '"confidence": 0.9, "target": "vibe", '
                          '"box": [100, 100, 500, 500]}]}')
        self.assertEqual(hits[0]["target"], "object")

    def test_normalized_box_validator_directly(self):
        self.assertIsNone(_normalized_box(None))
        self.assertIsNone(_normalized_box([0, 0, 4, 4]))          # sliver
        self.assertEqual(_normalized_box([0, 0, 500, 500]), (0.0, 0.0, 500.0, 500.0))


class AnnotationTests(unittest.TestCase):
    def _frame(self):
        return np.zeros((100, 200, 3), dtype=np.uint8)

    def test_the_box_is_drawn_colour_coded_on_a_copy(self):
        frame = self._frame()
        hit = {"name": "hoodie", "target": "person", "box": (100, 100, 900, 900)}
        evidence, pixel_box = annotate_hit(frame, hit)
        self.assertEqual(pixel_box, (20, 10, 180, 90))            # 0-1000 -> pixels
        self.assertIsNot(evidence, frame)
        self.assertEqual(int(frame.sum()), 0)                     # original untouched
        # The border carries the person colour.
        x1, y1, x2, y2 = pixel_box
        self.assertEqual(tuple(evidence[y1 + 1, (x1 + x2) // 2]),
                         TARGET_COLOURS["person"])

    def test_each_target_kind_gets_its_own_colour(self):
        for target in ("person", "object", "instrument"):
            evidence, box = annotate_hit(
                self._frame(), {"name": "r", "target": target,
                                "box": (100, 100, 900, 900)})
            x1, y1, x2, _ = box
            self.assertEqual(tuple(evidence[y1 + 1, (x1 + x2) // 2]),
                             TARGET_COLOURS[target], target)

    def test_no_box_ships_the_original_frame(self):
        frame = self._frame()
        evidence, pixel_box = annotate_hit(frame, {"name": "hoodie"})
        self.assertIs(evidence, frame)
        self.assertIsNone(pixel_box)

    def test_emit_carries_the_annotated_evidence_and_pixel_bbox(self):
        emitted = {}

        class Sink:
            def handle(self, alert, result):
                emitted["alert"] = alert
                emitted["result"] = result

        scanner = CustomRuleScanner([CAM], sink=Sink(), model="gemma3:4b")
        frame = self._frame()
        scanner._emit(CAM, frame, {"name": "hoodie", "reason": "hood up",
                                   "confidence": 0.9, "target": "instrument",
                                   "box": (100, 100, 900, 900)})
        payload = emitted["alert"].payload
        self.assertEqual(payload["bbox"], (20, 10, 180, 90))
        self.assertGreater(int(payload["frames"][0].sum()), 0)    # box drawn
        self.assertEqual(emitted["result"].confidence, 0.9)

    def test_emit_without_a_box_matches_the_old_behaviour(self):
        emitted = {}

        class Sink:
            def handle(self, alert, result):
                emitted["alert"] = alert

        scanner = CustomRuleScanner([CAM], sink=Sink(), model="gemma3:4b")
        frame = self._frame()
        scanner._emit(CAM, frame, {"name": "hoodie", "reason": "hood up"})
        self.assertNotIn("bbox", emitted["alert"].payload)
        self.assertIs(emitted["alert"].payload["frames"][0], frame)


if __name__ == "__main__":
    unittest.main()
