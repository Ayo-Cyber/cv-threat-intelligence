"""The detector only runs when something changed — and never misses because of it.

Detection samples on a timer, so an idle camera costs exactly what a busy one
costs. This gate removes that floor, and its whole design is about the rails:
skipping a frame must never lose an event.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.motion_gate import MotionGate


def scene(fill: int = 40, size=(120, 160)) -> np.ndarray:
    return np.full((*size, 3), fill, dtype=np.uint8)


def with_subject(base: np.ndarray, x: int = 40, w: int = 30) -> np.ndarray:
    f = base.copy()
    f[30:90, x:x + w] = 230            # a bright block: a person-sized change
    return f


class TheGateSavesWorkOnStillScenes(unittest.TestCase):
    def test_an_unchanging_scene_runs_once_then_is_skipped(self):
        g = MotionGate(heartbeat_seconds=60.0)
        room = scene()
        self.assertTrue(g.should_detect("cam", room, now=0.0))     # first frame always
        for i in range(1, 20):
            self.assertFalse(g.should_detect("cam", room, now=i * 0.25))
        self.assertEqual(g.stats()["ran"], 1)
        self.assertEqual(g.stats()["skipped"], 19)

    def test_sensor_noise_alone_does_not_wake_the_detector(self):
        g = MotionGate(heartbeat_seconds=60.0)
        rng = np.random.default_rng(4)
        base = scene()
        self.assertTrue(g.should_detect("cam", base, now=0.0))
        for i in range(1, 15):
            noisy = np.clip(base.astype(np.int16)
                            + rng.integers(-8, 9, base.shape), 0, 255).astype(np.uint8)
            self.assertFalse(g.should_detect("cam", noisy, now=i * 0.25))


class TheRailsNeverLoseAnEvent(unittest.TestCase):
    def test_a_subject_entering_wakes_it_immediately(self):
        g = MotionGate(heartbeat_seconds=60.0)
        room = scene()
        g.should_detect("cam", room, now=0.0)
        self.assertFalse(g.should_detect("cam", room, now=0.25))
        self.assertTrue(g.should_detect("cam", with_subject(room), now=0.5))

    def test_a_live_track_disables_the_gate_entirely(self):
        """A person standing still to loiter moves almost no pixels. While the
        camera holds a track the gate must not skip a single frame."""
        g = MotionGate(heartbeat_seconds=60.0)
        room = scene()
        g.should_detect("cam", room, now=0.0)
        for i in range(1, 12):
            self.assertTrue(g.should_detect("cam", room, tracked=1, now=i * 0.25))
        self.assertEqual(g.stats()["skipped"], 0)

    def test_the_heartbeat_bounds_how_long_a_camera_can_be_skipped(self):
        g = MotionGate(heartbeat_seconds=1.0)
        room = scene()
        g.should_detect("cam", room, now=0.0)
        self.assertFalse(g.should_detect("cam", room, now=0.5))
        self.assertTrue(g.should_detect("cam", room, now=1.0))     # forced

    def test_a_reconnect_at_the_same_shape_is_still_comparable(self):
        """Rescaling to a fixed analysis width makes an aspect-preserving
        resolution change comparable, so the gate keeps working across a
        reconnect rather than treating every one as new motion."""
        g = MotionGate(heartbeat_seconds=60.0)
        g.should_detect("cam", scene(), now=0.0)
        self.assertFalse(g.should_detect("cam", scene(size=(240, 320)), now=0.25))

    def test_a_changed_aspect_ratio_always_detects(self):
        """A genuinely different geometry has no comparable reference."""
        g = MotionGate(heartbeat_seconds=60.0)
        g.should_detect("cam", scene(), now=0.0)
        self.assertTrue(g.should_detect("cam", scene(size=(240, 240)), now=0.25))

    def test_cameras_are_independent(self):
        g = MotionGate(heartbeat_seconds=60.0)
        room = scene()
        g.should_detect("a", room, now=0.0)
        self.assertTrue(g.should_detect("b", room, now=0.1))       # b's first frame
        self.assertFalse(g.should_detect("a", room, now=0.2))


class TheMeasuredSeparation(unittest.TestCase):
    def test_the_default_sits_between_idle_and_occupied(self):
        """Measured 25 Sep: idle fixed cameras move 0.00% of pixels, clips with
        a person move 6.5%-35%. The shipped default must sit clear of both."""
        g = MotionGate()
        self.assertGreater(g.min_changed_fraction, 0.0)
        self.assertLess(g.min_changed_fraction, 0.065)

    def test_change_reports_the_moved_fraction(self):
        g = MotionGate()
        room = scene()
        self.assertIsNone(g.change("cam", room))        # nothing to compare yet
        g.should_detect("cam", room, now=0.0)
        self.assertEqual(g.change("cam", room), 0.0)
        self.assertGreater(g.change("cam", with_subject(room)), 0.065)


if __name__ == "__main__":
    unittest.main()
