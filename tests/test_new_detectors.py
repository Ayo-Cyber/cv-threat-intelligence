"""Unit tests for the fire/smoke, fall, and crowd CV pre-filter detectors.

These are triggers (the VLM gate is the real judge), so the tests assert the cheap
signal fires on the obvious positive and stays quiet on the obvious negative.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.detector.crowd import CrowdDetector
from cvti.detector.fall import FallDetector
from cvti.detector.fire import FireSmokeDetector


class FireSmokeTests(unittest.TestCase):
    def test_flickering_flame_fires(self):
        det = FireSmokeDetector()
        fired = None
        for i in range(15):
            frame = np.zeros((200, 200, 3), np.uint8)
            size = 70 if i % 2 == 0 else 40          # oscillating area => flicker
            frame[10:10 + size, 10:10 + size] = (0, 120, 255)   # bright orange (BGR)
            fired = det.update(frame, timestamp=i * 0.25) or fired
        self.assertIsNotNone(fired)
        self.assertEqual(fired["kind"], "fire")

    def test_steady_grey_is_quiet(self):
        det = FireSmokeDetector()
        out = None
        for i in range(15):
            out = det.update(np.full((200, 200, 3), 100, np.uint8), timestamp=i * 0.25) or out
        self.assertIsNone(out)

    def test_static_orange_wall_does_not_flicker(self):
        # A steady orange block is flame-coloured but NOT flickering -> no fire.
        det = FireSmokeDetector()
        out = None
        for i in range(15):
            frame = np.zeros((200, 200, 3), np.uint8)
            frame[10:80, 10:80] = (0, 120, 255)      # constant size => no flicker
            out = det.update(frame, timestamp=i * 0.25) or out
        self.assertIsNone(out)


class FallTests(unittest.TestCase):
    def test_horizontal_person_fires(self):
        det = FallDetector(min_frames=3)
        res = None
        for i in range(6):
            res = det.update([(1, 10, 100, 210, 160)], frame_area=200 * 200, timestamp=i * 0.25) or res
        self.assertIsNotNone(res)
        self.assertEqual(res["kind"], "fall")
        self.assertEqual(res["track_id"], 1)

    def test_upright_person_is_quiet(self):
        det = FallDetector(min_frames=3)
        res = None
        for i in range(6):
            res = det.update([(1, 80, 10, 120, 180)], frame_area=200 * 200, timestamp=i * 0.25) or res
        self.assertIsNone(res)

    def test_brief_bend_is_quiet(self):
        # One horizontal frame then upright -> streak never reaches min_frames.
        det = FallDetector(min_frames=4)
        det.update([(1, 10, 100, 210, 160)], frame_area=200 * 200, timestamp=0.0)
        res = det.update([(1, 80, 10, 120, 180)], frame_area=200 * 200, timestamp=0.25)
        self.assertIsNone(res)


class CrowdTests(unittest.TestCase):
    def test_crowd_fires_on_sustained_headcount(self):
        det = CrowdDetector(crowd_count=5, min_frames=3, warmup=2)
        base = np.zeros((120, 120, 3), np.uint8)
        res = None
        for i in range(6):
            res = det.update(7, base, timestamp=i * 0.25) or res
        self.assertIsNotNone(res)
        self.assertIn(res["kind"], ("crowd", "stampede"))

    def test_few_people_quiet(self):
        det = CrowdDetector(crowd_count=8, min_frames=3, warmup=2)
        base = np.zeros((120, 120, 3), np.uint8)
        res = None
        for i in range(6):
            res = det.update(2, base, timestamp=i * 0.25) or res
        self.assertIsNone(res)

    def test_stampede_on_motion_spike(self):
        det = CrowdDetector(stampede_count=3, motion_spike=2.0, motion_floor=5.0, warmup=3)
        # calm baseline frames, then a violently different frame with people present
        got = None
        for i in range(6):
            got = det.update(4, np.full((100, 100, 3), 50, np.uint8), timestamp=i * 0.2) or got
        # big motion: flip the whole frame bright
        got = det.update(6, np.full((100, 100, 3), 220, np.uint8), timestamp=2.0) or got
        self.assertIsNotNone(got)
        self.assertEqual(got["kind"], "stampede")


if __name__ == "__main__":
    unittest.main()
