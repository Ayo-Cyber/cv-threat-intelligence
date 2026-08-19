"""The subject bounding box — showing WHO an alert is about.

An alert used to say a threat happened somewhere in a 1080p frame. These cover
the chain that now points at the person: capture the box -> save an annotated
shot -> store the box -> surface it separately from the clip frames.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.app.console_backend import ConsoleBackend
from cvti.contracts import CandidateAlert
from cvti.serving.alert_queue import QueuedAlert
from cvti.serving.alert_sink import AlertSink

from _backend_helper import signed_in


class _Result:
    confirmed, confidence, reason = True, 0.9, "lingering"


def _fire(d: Path, bbox):
    sink = AlertSink(str(d), routing_path=None)
    frame = np.full((240, 320, 3), 40, np.uint8)
    cand = CandidateAlert(rule_name="loitering_watch", detector="presence",
                          title="PERSON IN ZONE", person_id=7, object_label=None,
                          priority="medium", timestamp=1.0)
    sink.handle(QueuedAlert(camera_id="cam1", rule_name="loitering_watch", priority="medium",
                            title="T", timestamp=1.0, track_id=7,
                            payload={"candidate": cand, "frames": [frame], "bbox": bbox,
                                     "clip_frames": [], "clip_fps": 0}), _Result())
    return sorted((d / "events").iterdir())[0]


class AnnotationTests(unittest.TestCase):
    def test_annotate_draws_and_never_mutates_the_original(self):
        frame = np.full((100, 100, 3), 30, np.uint8)
        out = AlertSink._annotate(frame, (10, 10, 60, 80), "#1 THEFT", "critical")
        self.assertFalse(np.array_equal(out, frame))
        self.assertTrue(np.array_equal(frame, np.full((100, 100, 3), 30, np.uint8)))

    def test_no_bbox_returns_frame_untouched(self):
        frame = np.full((50, 50, 3), 7, np.uint8)
        self.assertIs(AlertSink._annotate(frame, None, "x"), frame)

    def test_out_of_bounds_box_is_clamped_not_crashed(self):
        frame = np.full((80, 80, 3), 20, np.uint8)
        out = AlertSink._annotate(frame, (-50, -50, 999, 999), "wild", "high")
        self.assertEqual(out.shape, frame.shape)


class SinkSubjectTests(unittest.TestCase):
    def test_subject_shot_saved_and_bbox_persisted(self):
        d = Path(tempfile.mkdtemp())
        ev = _fire(d, (90, 60, 160, 200))
        self.assertTrue((ev / "subject.jpg").exists())
        row = sqlite3.connect(d / "events.db").execute("SELECT bbox FROM events").fetchone()
        self.assertEqual(row[0], "90,60,160,200")

    def test_no_bbox_means_no_subject_shot(self):
        d = Path(tempfile.mkdtemp())
        ev = _fire(d, None)
        self.assertFalse((ev / "subject.jpg").exists())
        row = sqlite3.connect(d / "events.db").execute("SELECT bbox FROM events").fetchone()
        self.assertIsNone(row[0])


class BackendExposureTests(unittest.TestCase):
    def test_subject_is_separate_from_the_cine_loop_frames(self):
        d = Path(tempfile.mkdtemp())
        _fire(d, (90, 60, 160, 200))
        be = signed_in(site_path=str(d / "s.json"), db_path=str(d / "events.db"),
                            enable_demo=False)
        ev = be.list_events(10)[0]
        self.assertTrue(ev["subject"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(ev["bbox"], "90,60,160,200")
        # the annotated shot must NOT appear in the clip frames, or a box would
        # flash at the end of every loop
        self.assertGreaterEqual(len(ev["frames"]), 1)
        self.assertNotIn(ev["subject"], ev["frames"])


if __name__ == "__main__":
    unittest.main()
