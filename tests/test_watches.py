"""Watches: plain-English descriptions bound to tracked people, followed as cases.

The model is faked throughout — what needs proving is the binding (a sentence to
a stable track id), the case lifecycle, and that a hallucinated answer can't
invent a watch or a person.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.serving.watch_runner import WatchRunner
from cvti.serving.watches import (
    CaseBook, Watch, annotate_candidates, build_prompt, parse_matches,
)

WATCHES = [Watch("red-jacket", "the man in the red jacket"),
           Watch("loiterer", "anyone who keeps returning to the spirits aisle")]
MAPPING = {1: 101, 2: 202}          # drawn number -> tracker id


class AnnotateTests(unittest.TestCase):
    def test_numbers_map_to_track_ids_in_order(self):
        frame = np.zeros((200, 200, 3), np.uint8)
        out, mapping = annotate_candidates(frame, [(101, 10, 10, 60, 120), (202, 90, 20, 140, 130)])
        self.assertEqual(mapping, {1: 101, 2: 202})
        self.assertFalse(np.array_equal(out, frame))     # boxes actually drawn
        self.assertTrue(np.array_equal(frame, np.zeros((200, 200, 3), np.uint8)))  # original intact

    def test_prompt_lists_every_watch_and_the_person_count(self):
        p = build_prompt(WATCHES, 2, "a shop aisle")
        self.assertIn("red-jacket", p)
        self.assertIn("loiterer", p)
        self.assertIn("1 to 2", p)
        self.assertIn("a shop aisle", p)


class ParseTests(unittest.TestCase):
    def test_binds_a_watch_to_a_track_id(self):
        raw = '{"matches":[{"watch":"red-jacket","person":2,"reason":"red coat"}]}'
        got = parse_matches(raw, WATCHES, MAPPING)
        self.assertEqual(got, [{"watch": "red-jacket", "track_id": 202, "reason": "red coat"}])

    def test_person_zero_means_nobody_matched(self):
        raw = '{"matches":[{"watch":"red-jacket","person":0,"reason":"not here"}]}'
        self.assertEqual(parse_matches(raw, WATCHES, MAPPING), [])

    def test_invented_watch_is_ignored(self):
        raw = '{"matches":[{"watch":"arson","person":1,"reason":"made up"}]}'
        self.assertEqual(parse_matches(raw, WATCHES, MAPPING), [])

    def test_person_number_we_never_drew_is_ignored(self):
        raw = '{"matches":[{"watch":"red-jacket","person":9,"reason":"hallucinated"}]}'
        self.assertEqual(parse_matches(raw, WATCHES, MAPPING), [])

    def test_garbage_and_prose_do_not_crash(self):
        for raw in ("", "no json here", "{broken", '{"matches": "not a list"}',
                    '{"matches":[{"watch":"red-jacket","person":"two"}]}'):
            self.assertEqual(parse_matches(raw, WATCHES, MAPPING), [])

    def test_tolerates_surrounding_prose(self):
        raw = 'Sure! {"matches":[{"watch":"loiterer","person":1,"reason":"back again"}]} hope that helps'
        self.assertEqual(parse_matches(raw, WATCHES, MAPPING)[0]["track_id"], 101)


class CaseBookTests(unittest.TestCase):
    def test_first_sighting_opens_a_case_repeats_update_it(self):
        book = CaseBook(stale_after=30)
        c1, new1 = book.observe("cam1", "red-jacket", 101, now=100.0)
        self.assertTrue(new1)
        c2, new2 = book.observe("cam1", "red-jacket", 101, now=110.0)
        self.assertFalse(new2)                    # same subject: no second alert
        self.assertIs(c1, c2)
        self.assertEqual(c2.sightings, 2)
        self.assertEqual(c2.duration, 10.0)

    def test_different_track_is_a_different_case(self):
        book = CaseBook()
        book.observe("cam1", "red-jacket", 101, now=1.0)
        _, new = book.observe("cam1", "red-jacket", 999, now=2.0)
        self.assertTrue(new)

    def test_stale_cases_close_and_leave_active(self):
        book = CaseBook(stale_after=30)
        book.observe("cam1", "red-jacket", 101, now=100.0)
        self.assertEqual(len(book.active(now=120.0)), 1)
        closed = book.expire(now=200.0)
        self.assertEqual(len(closed), 1)
        self.assertEqual(book.active(now=200.0), [])

    def test_a_returning_subject_opens_a_fresh_case(self):
        book = CaseBook(stale_after=30)
        book.observe("cam1", "red-jacket", 101, now=100.0)
        book.expire(now=200.0)
        _, new = book.observe("cam1", "red-jacket", 101, now=210.0)
        self.assertTrue(new)                      # came back -> a new case, alert again

    def test_bbox_is_kept_current(self):
        book = CaseBook()
        book.observe("cam1", "w", 1, bbox=(1, 2, 3, 4), now=1.0)
        case, _ = book.observe("cam1", "w", 1, bbox=(5, 6, 7, 8), now=2.0)
        self.assertEqual(case.bbox, (5, 6, 7, 8))


class _State:
    def __init__(self, boxes):
        self._box_by_track = boxes
        self.scene_context = {"scene_description": "a shop aisle"}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.cam = {"id": "cam1", "watches": [{"name": "red-jacket",
                                               "description": "the man in the red jacket"}]}
        self.states = {"cam1": _State({101: (10, 10, 60, 120), 202: (90, 20, 140, 130)})}
        self.sent = []
        self.sink = type("S", (), {"handle": lambda _s, a, r: self.sent.append((a, r))})()

    def _runner(self, reply):
        r = WatchRunner([self.cam], self.states, self.sink, model="fake")
        r._ask = lambda prompt, frame_bytes: reply
        return r

    def test_opens_a_case_and_alerts_once(self):
        r = self._runner('{"matches":[{"watch":"red-jacket","person":1,"reason":"red coat"}]}')
        frame = np.zeros((200, 200, 3), np.uint8)
        opened = r.scan_camera(self.cam, frame, now=100.0)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["track_id"], 101)
        self.assertEqual(len(self.sent), 1)
        alert, _ = self.sent[0]
        self.assertEqual(alert.rule_name, "watch:red-jacket")
        self.assertEqual(alert.track_id, 101)
        self.assertEqual(alert.payload["bbox"], (10, 10, 60, 120))   # box travels with it

        # seeing them again must NOT alert again
        r.scan_camera(self.cam, frame, now=105.0)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(r.active_cases(now=105.0)[0]["sightings"], 2)

    def test_no_tracked_people_means_no_vlm_call(self):
        self.states["cam1"]._box_by_track = {}
        r = self._runner("should not be used")
        called = []
        r._ask = lambda *a, **k: called.append(1) or "{}"
        self.assertEqual(r.scan_camera(self.cam, np.zeros((10, 10, 3), np.uint8)), [])
        self.assertEqual(called, [])            # nobody to bind to -> don't burn a verdict

    def test_camera_without_watches_is_ignored(self):
        r = WatchRunner([{"id": "cam2"}], self.states, self.sink, model="fake")
        self.assertEqual(r.cameras, [])


if __name__ == "__main__":
    unittest.main()
