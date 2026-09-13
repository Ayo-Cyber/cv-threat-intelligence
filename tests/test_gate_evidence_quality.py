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
import json
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import supervision as sv

from cvti.contracts import CandidateAlert
from cvti.event_adapters import concealment_to_events
from cvti.rules.customization import CustomizationEngine
from cvti.verification.frame_select import (
    append_subject_crop,
    frames_for_rule,
    subject_crop,
)
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

class ConcealmentCueFlowTest(unittest.TestCase):
    def test_assessment_cues_survive_the_event_and_simple_rule(self):
        assessment = SimpleNamespace(
            track_id=7,
            score=0.8125,
            candidate=True,
            destination="bag",
            components={"f_waist": 0.1, "f_bag": 0.9,
                        "f_retract": 0.8, "f_dwell": 0.75},
            reasons=["hand reached a personal bag", "arm retracted to body"],
            limited=False,
            associated_bag=(180.0, 170.0, 240.0, 235.0),
        )

        event = concealment_to_events([assessment], timestamp=3.5)[0]

        self.assertEqual(event.extra, {
            "destination": "bag",
            "score": 0.8125,
            "components": assessment.components,
            "reasons": assessment.reasons,
            "limited": False,
            "associated_bag": assessment.associated_bag,
        })

        engine = CustomizationEngine()
        engine.rules = [{
            "name": "shoplifting",
            "priority": "high",
            "trigger": {"detector": "concealment"},
        }]
        alert = engine.evaluate([event], {"environment_type": "retail"})[0]
        self.assertEqual(alert.reasons, assessment.reasons)
        self.assertEqual(alert.metadata, event.extra)

    def test_nested_assessment_metadata_is_detached_at_each_boundary(self):
        components = {"destination": {"bag": 0.9}}
        reasons = [{"cue": ["hand", "bag"]}]
        assessment = SimpleNamespace(
            track_id=7,
            score=0.8125,
            candidate=True,
            destination="bag",
            components=components,
            reasons=reasons,
            limited=False,
            associated_bag=(180.0, 170.0, 240.0, 235.0),
        )
        event = concealment_to_events([assessment], timestamp=3.5)[0]
        engine = CustomizationEngine()
        engine.rules = [{
            "name": "shoplifting",
            "priority": "high",
            "trigger": {"detector": "concealment"},
        }]
        alert = engine.evaluate([event], {"environment_type": "retail"})[0]

        components["destination"]["bag"] = 0.1
        reasons[0]["cue"].append("mutated")
        self.assertEqual(event.extra["components"], {"destination": {"bag": 0.9}})
        self.assertEqual(event.extra["reasons"], [{"cue": ["hand", "bag"]}])

        event.extra["components"]["destination"]["bag"] = 0.2
        event.extra["reasons"][0]["cue"].append("event-mutated")

        self.assertEqual(alert.metadata["components"], {"destination": {"bag": 0.9}})
        self.assertEqual(alert.metadata["reasons"], [{"cue": ["hand", "bag"]}])
        self.assertEqual(alert.reasons, [{"cue": ["hand", "bag"]}])


class ConcealmentVerificationIntegrationTest(unittest.TestCase):
    def test_true_sight_receives_three_chronological_full_frames_then_subject_crop(self):
        from cvti.detector.core import PosePersonState
        from cvti.retail.concealment import ConcealmentAssessment
        from cvti.serving.camera import PerCameraState

        engine = CustomizationEngine()
        engine.rules = [{
            "name": "shoplifting",
            "priority": "high",
            "trigger": {"detector": "concealment"},
        }]
        state = PerCameraState(
            "cam1", engine, person_filter=False, pose_model=object(),
            concealment=True, heavy_stride=1,
        )
        state._frame_buffer.extend(
            np.full((480, 640, 3), value, dtype=np.uint8)
            for value in (10, 20, 30, 40)
        )
        moment = np.full((480, 640, 3), 50, dtype=np.uint8)
        detections = sv.Detections(
            xyxy=np.array([[300.0, 200.0, 330.0, 260.0]]),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            tracker_id=np.array([7]),
        )
        pose = PosePersonState(
            track_id=7, bbox=(300.0, 200.0, 330.0, 260.0), timestamp=0.5,
            left_shoulder=(305.0, 210.0), right_shoulder=(325.0, 210.0),
            left_elbow=None, right_elbow=None, left_wrist=(315.0, 250.0),
            right_wrist=None, max_wrist_speed=0.0, max_wrist_accel=0.0,
            max_arm_extension_ratio=0.0, weapon_labels=[],
            left_hip=(307.0, 250.0), right_hip=(323.0, 250.0),
        )
        assessment = ConcealmentAssessment(
            track_id=7, score=0.8, candidate=True, destination="waist",
            reasons=["hand reached the waist line"],
            components={"f_waist": 0.9, "f_bag": 0.0,
                        "f_retract": 0.8, "f_dwell": 0.7},
        )

        with mock.patch.object(state._tracker, "update_with_detections",
                               return_value=detections), \
                mock.patch.object(state, "_compute_pose", return_value=[pose]), \
                mock.patch.object(state._conceal, "update", return_value=[assessment]):
            queued = state.process(detections, moment, timestamp=0.5)

        self.assertEqual(len(queued), 1)
        payload = queued[0].payload
        received = {}
        gate = VerificationGate(provider="ollama", cot=False)

        def capture_provider(prompt, frames_bytes, alert):
            received["images"] = [
                cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                for data in frames_bytes
            ]
            return ('{"confirmed": false, "confidence": 0.9, '
                    '"reason": "test", "alert_priority": "high"}')

        with mock.patch.object(gate, "_call_provider", side_effect=capture_provider):
            result = gate.verify(payload["frames"], payload["candidate"], payload["scene"])

        images = received["images"]
        self.assertEqual(len(images), 4)
        self.assertEqual([round(float(image.mean())) for image in images[:3]],
                         [10, 30, 50])
        self.assertTrue(all(image.shape[:2] == (480, 640) for image in images[:3]))
        self.assertGreaterEqual(min(images[-1].shape[:2]), 320)

        from cvti.serving.alert_sink import AlertSink
        with tempfile.TemporaryDirectory() as tmp:
            sink = AlertSink(tmp, save_evidence=False)
            try:
                sink.handle(queued[0], result)
            finally:
                sink.close()
            con = sqlite3.connect(Path(tmp) / "events.db")
            con.row_factory = sqlite3.Row
            try:
                audit = dict(con.execute(
                    "SELECT * FROM concealment_audit ORDER BY id DESC LIMIT 1"
                ).fetchone())
            finally:
                con.close()

        self.assertEqual(audit["camera_id"], "cam1")
        self.assertEqual(audit["candidate_timestamp"], 0.5)
        self.assertEqual(audit["track_id"], 7)
        self.assertEqual(audit["peak_score"], 0.8)
        self.assertEqual(json.loads(audit["components_json"]), assessment.components)
        self.assertEqual(audit["limited"], 0)
        self.assertIsNone(audit["associated_bag_json"])
        self.assertEqual(audit["verdict"], "rejected")


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
        # A local gate caps at LOCAL_MAX_FRAMES (11 Sep pilot: the vision
        # tower pays per image) — but the cap keeps the LAST image, so the
        # subject crop still travels as its own image.
        self.assertEqual(seen["n"], 2, "one context frame + the subject crop")

    def test_concealment_question_rejects_common_normal_actions(self):
        policy = (
            "Reject normal browsing, phone handling, clothing adjustment, openly carried "
            "goods, and placement into a trolley or shopping basket."
        )
        for sensitivity in ("balanced", "strict"):
            gate = VerificationGate(provider="ollama", cot=False,
                                    sensitivity=sensitivity)
            seen = {}

            def fake_provider(prompt, frames_bytes, alert):
                seen["prompt"] = prompt
                return ('{"confirmed": false, "confidence": 0.9, '
                        '"reason": "normal action", "alert_priority": "high"}')

            with mock.patch.object(gate, "_call_provider", side_effect=fake_provider):
                gate.verify([_frame()], CandidateAlert(
                    rule_name="shoplifting", priority="high", detector="concealment",
                    title="POSSIBLE CONCEALMENT", person_id=3, object_label=None,
                    timestamp=0.0,
                ), {"environment_type": "retail"})

            self.assertIn(policy, seen["prompt"], sensitivity)

    def test_simultaneous_movement_uses_three_frames_and_a_specific_question(self):
        self.assertEqual(frames_for_rule("chi_multiple_people_moving"), 3)
        seen = {}
        gate = VerificationGate(provider="ollama", cot=False)

        def fake_provider(prompt, frames_bytes, alert):
            seen["prompt"] = prompt.lower()
            return ('{"confirmed": true, "confidence": 0.9, '
                    '"reason": "movement", "alert_priority": "high"}')

        alert = CandidateAlert(
            rule_name="chi_multiple_people_moving", priority="high",
            detector="multiple_people_moving", title="MULTIPLE PEOPLE MOVING",
            person_id=None, object_label=None, timestamp=1.0,
        )
        with mock.patch.object(gate, "_call_provider", side_effect=fake_provider):
            gate.verify([_frame(), _frame(), _frame()], alert, {})

        self.assertIn("multiple distinct people", seen["prompt"])
        self.assertIn("moving at the same time", seen["prompt"])
        self.assertIn("crowd density", seen["prompt"])
        self.assertIn("panic", seen["prompt"])


if __name__ == "__main__":
    unittest.main()
