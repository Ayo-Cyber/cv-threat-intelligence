"""W3: object/attribute sentences get a detector; scenes keep the VLM.

The charter's acceptance list, held as tests: the cap/glasses/bus class of
rule routes to YOLO-World and comes back grounded (a real box from a real
detector); behaviour sentences are untouched; a missing detector degrades
VISIBLY to the VLM path instead of going blind; and the routing decision is
readable in the scanner's status file.
"""
import unittest

import numpy as np

from cvti.detector.openvocab import (
    MIN_SCORE, OpenVocabDetector, _phrase_for, route_rule,
)
from cvti.serving.custom_rules import CustomRuleScanner, annotate_hit


class RouteRuleTest(unittest.TestCase):
    def test_the_charter_cases_route_to_the_detector(self):
        for text in ("person wearing a cap",
                     "someone wearing glasses",
                     "Detect the white bus",
                     "a man with a backpack",
                     "person in a red jacket"):
            self.assertEqual(route_rule(text), "openvocab", text)

    def test_behaviour_sentences_stay_with_the_vlm(self):
        for text in ("someone climbing over the counter",
                     "people fighting in the lobby",
                     "a person running from the entrance",
                     "someone loitering near the tills",
                     "person entering after hours",
                     "bag left unattended"):     # temporal/absence: not a look
            self.assertEqual(route_rule(text), "vlm", text)

    def test_mixed_sentences_err_toward_the_vlm(self):
        # Attribute AND behaviour: the behaviour wins — a wrongly-detector'd
        # rule goes blind, a wrongly-VLM'd one merely stays as accurate as today.
        self.assertEqual(route_rule("person in a hoodie climbing the fence"), "vlm")

    def test_unclassifiable_defaults_to_the_vlm(self):
        self.assertEqual(route_rule("something odd near the door"), "vlm")
        self.assertEqual(route_rule(""), "vlm")

    def test_phrases_are_stripped_of_imperative_scaffolding(self):
        self.assertEqual(_phrase_for("Detect the white bus"), "white bus")
        self.assertEqual(_phrase_for("alert me if you see a person wearing a cap"),
                         "person wearing a cap")
        self.assertEqual(_phrase_for("person wearing a cap?"), "person wearing a cap")


class _FakeBoxes:
    def __init__(self, rows):
        # rows: [(cls, conf, x1, y1, x2, y2)]
        self.cls = [r[0] for r in rows]
        self.conf = [r[1] for r in rows]
        self.xyxy = [r[2:] for r in rows]

    def __len__(self):
        return len(self.cls)


class _FakeResult:
    def __init__(self, rows, names):
        self.boxes = _FakeBoxes(rows)
        self.names = names


class _FakeWorldModel:
    """Stands in for ultralytics.YOLOWorld."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.classes: list = []
        self.set_calls = 0
        self.predict_calls = 0

    def set_classes(self, classes):
        self.classes = list(classes)
        self.set_calls += 1

    def predict(self, frame, **kw):
        self.predict_calls += 1
        names = dict(enumerate(self.classes))
        return [_FakeResult(self.rows, names)]


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


class OpenVocabDetectorTest(unittest.TestCase):
    def test_detections_come_back_as_phrase_score_box(self):
        fake = _FakeWorldModel(rows=[(0, 0.72, 10, 20, 110, 220)])
        det = OpenVocabDetector(model_factory=lambda w: fake)
        out = det.detect(_frame(), ["person wearing a cap"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["phrase"], "person wearing a cap")
        self.assertAlmostEqual(out[0]["score"], 0.72)
        self.assertEqual(out[0]["box"], (10.0, 20.0, 110.0, 220.0))

    def test_set_classes_runs_once_per_phrase_set(self):
        fake = _FakeWorldModel()
        det = OpenVocabDetector(model_factory=lambda w: fake)
        det.detect(_frame(), ["white bus"])
        det.detect(_frame(), ["white bus"])
        self.assertEqual(fake.set_calls, 1)
        det.detect(_frame(), ["white bus", "red car"])
        self.assertEqual(fake.set_calls, 2)

    def test_a_model_that_cannot_load_returns_none_not_a_crash(self):
        def boom(w):
            raise RuntimeError("no weights on this box")
        det = OpenVocabDetector(model_factory=boom)
        self.assertIsNone(det.detect(_frame(), ["white bus"]))
        self.assertIn("no weights", det.load_error)

    def test_an_empty_answer_is_a_grounded_no_not_a_fallback(self):
        det = OpenVocabDetector(model_factory=lambda w: _FakeWorldModel())
        self.assertEqual(det.detect(_frame(), ["white bus"]), [])


class _FakeSink:
    def __init__(self):
        self.handled = []

    def handle(self, alert, result):
        self.handled.append((alert, result))


def _scanner(cameras, openvocab=None):
    return CustomRuleScanner(cameras, _FakeSink(), model="gemma3:4b",
                             openvocab=openvocab)


def _cam(rules):
    return {"id": "cam1", "source": 0,
            "custom_rules": [{"question": q} for q in rules]}


class ScannerRoutingTest(unittest.TestCase):
    CAP = "person wearing a cap"
    CLIMB = "someone climbing over the counter"

    def test_rules_split_by_engine(self):
        s = _scanner([_cam([self.CAP, self.CLIMB])])
        obj, scene = s._split_rules(s.cameras[0])
        self.assertEqual([t["description"] for t in obj], [self.CAP])
        self.assertEqual([t["description"] for t in scene], [self.CLIMB])

    def test_object_rules_answered_grounded_and_vlm_sees_only_scenes(self):
        fake = _FakeWorldModel(rows=[(0, 0.66, 100, 50, 200, 400)])
        det = OpenVocabDetector(model_factory=lambda w: fake)
        s = _scanner([_cam([self.CAP, self.CLIMB])], openvocab=det)
        vlm_threats = []
        s._check = lambda cam, frame, threats=None: (
            vlm_threats.extend(threats or []) or [])
        hits = s._check_all(s.cameras[0], _frame())
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0]["grounded"])
        self.assertEqual(hits[0]["engine"], "yolo-world")
        self.assertEqual(hits[0]["target"], "person")
        self.assertEqual([t["description"] for t in vlm_threats], [self.CLIMB])

    def test_detector_failure_rides_the_vlm_that_cycle(self):
        def boom(w):
            raise RuntimeError("clip missing")
        det = OpenVocabDetector(model_factory=boom)
        s = _scanner([_cam([self.CAP, self.CLIMB])], openvocab=det)
        vlm_threats = []
        s._check = lambda cam, frame, threats=None: (
            vlm_threats.extend(threats or []) or [])
        hits = s._check_all(s.cameras[0], _frame())
        self.assertEqual(hits, [])
        self.assertEqual({t["description"] for t in vlm_threats},
                         {self.CAP, self.CLIMB})

    def test_no_detection_means_no_hit_not_a_fallback(self):
        det = OpenVocabDetector(model_factory=lambda w: _FakeWorldModel())
        s = _scanner([_cam([self.CAP])], openvocab=det)
        s._check = lambda cam, frame, threats=None: [
            {"name": "never", "reason": "vlm must not see this rule"}]
        self.assertEqual(s._check_all(s.cameras[0], _frame()), [])

    def test_status_records_which_engine_answers_which_rule(self):
        det = OpenVocabDetector(model_factory=lambda w: _FakeWorldModel())
        s = _scanner([_cam([self.CAP, self.CLIMB])], openvocab=det)
        s._record(s.cameras[0], [])
        entry = s._status["cam1"]
        self.assertIn("person wearing a cap", entry["routing"]["yolo-world"][0])
        self.assertIn("someone climbing over the", entry["routing"]["vlm"][0])


class GroundedAnnotationTest(unittest.TestCase):
    def test_a_grounded_object_box_is_drawn_not_corner_tagged(self):
        frame = _frame()
        hit = {"name": "white bus", "target": "object", "grounded": True,
               "box": (100, 100, 900, 900)}
        evidence, pixel_box = annotate_hit(frame, hit)
        self.assertIsNotNone(pixel_box, "a detector box must be drawn, even large")

    def test_an_ungrounded_object_claim_still_only_tags(self):
        evidence, pixel_box = annotate_hit(_frame(), {
            "name": "white bus", "target": "object", "box": (100, 100, 300, 300)})
        self.assertIsNone(pixel_box)


class FloorTest(unittest.TestCase):
    def test_the_detector_floor_is_the_documented_default(self):
        self.assertAlmostEqual(MIN_SCORE, 0.30)
        self.assertAlmostEqual(OpenVocabDetector(
            model_factory=lambda w: _FakeWorldModel()).min_score, 0.30)

    def test_worn_garment_phrases_carry_the_higher_floor(self):
        from cvti.detector.openvocab import WORN_MIN_SCORE, floor_for
        self.assertAlmostEqual(WORN_MIN_SCORE, 0.45)
        for phrase in ("person wearing a cap", "person wearing a hoodie",
                       "person in a red jacket"):
            self.assertAlmostEqual(floor_for(phrase), 0.45, phrase)
        for phrase in ("backpack", "bus", "person"):
            self.assertAlmostEqual(floor_for(phrase), 0.30, phrase)

    def test_a_marginal_worn_score_is_filtered_a_solid_one_kept(self):
        # The measured failure mode: bare heads scoring 0.31-0.39 on "cap"
        # while real caps score 0.58+. The per-phrase floor kills the former.
        rows = [(0, 0.36, 1, 1, 5, 5), (0, 0.74, 10, 10, 60, 60)]
        det = OpenVocabDetector(model_factory=lambda w: _FakeWorldModel(rows))
        out = det.detect(_frame(), ["person wearing a cap"])
        self.assertEqual([d["score"] for d in out], [0.74])

    def test_object_phrases_keep_their_low_floor(self):
        rows = [(0, 0.32, 1, 1, 5, 5)]
        det = OpenVocabDetector(model_factory=lambda w: _FakeWorldModel(rows))
        out = det.detect(_frame(), ["backpack"])
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
