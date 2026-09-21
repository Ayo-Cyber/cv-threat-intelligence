"""One vehicle through a gate is one crossing, however many ids the tracker gives it.

A car waiting at a barrier arm is lost and re-acquired several times. Every
fresh id that then inches over the tripwire used to be a new crossing:
replayed on the barrier clip, ONE car -> 6 ids -> 5 entries (21 Sep). A new id
whose box overlaps where a recently-crossed track was last seen is that same
vehicle, and is not counted again. A genuinely different vehicle -- elsewhere
on the line, or later than the memory window -- still is.
"""
from __future__ import annotations

import unittest

import numpy as np
import supervision as sv

from cvti.serving.camera import PerCameraState, _box_iou


def _det(box, tid):
    return sv.Detections(xyxy=np.array([box], dtype=float), class_id=np.array([2]),
                         confidence=np.array([0.9]), tracker_id=np.array([tid]))


def _state():
    st = PerCameraState.__new__(PerCameraState)
    st.vehicle_line = {"name": "gate", "start": [0.5, 0.0], "end": [0.5, 1.0],
                       "normalized": True, "flip": False}
    st._vehicle_line_zone = None
    st._vehicle_line_recent = []
    return st


class OneVehicleOneCrossing(unittest.TestCase):
    HW = (200, 400)   # 400 wide: the line sits at x=200

    def _cross(self, st, tid, y, t0):
        """Move one id from left of the line to right of it, two frames."""
        ev = []
        ev += st._vehicle_line_events(_det((120, y, 180, y + 40), tid), self.HW, t0)
        ev += st._vehicle_line_events(_det((220, y, 280, y + 40), tid), self.HW, t0 + 0.2)
        return ev

    def test_a_reacquired_id_at_the_same_spot_is_not_recounted(self):
        st = _state()
        first = self._cross(st, tid=1, y=100, t0=10.0)
        again = self._cross(st, tid=2, y=104, t0=18.0)      # new id, same place, 8s later
        self.assertEqual(len(first), 1)
        self.assertEqual(again, [], "the same vehicle under a new id crossed twice")

    def test_a_different_vehicle_elsewhere_on_the_line_still_counts(self):
        st = _state()
        self._cross(st, tid=1, y=20, t0=10.0)
        other = self._cross(st, tid=2, y=140, t0=12.0)      # far down the line
        self.assertEqual(len(other), 1)

    def test_the_memory_expires(self):
        st = _state()
        self._cross(st, tid=1, y=100, t0=10.0)
        later = self._cross(st, tid=2, y=100, t0=10.0 + PerCameraState.VEHICLE_LINE_MEMORY_S + 5)
        self.assertEqual(len(later), 1, "a vehicle at the same spot a minute later is a new one")

    def test_opposite_direction_is_a_different_event(self):
        st = _state()
        self._cross(st, tid=1, y=100, t0=10.0)
        ev = []
        ev += st._vehicle_line_events(_det((220, 100, 280, 140), 3), self.HW, 14.0)
        ev += st._vehicle_line_events(_det((120, 100, 180, 140), 3), self.HW, 14.2)
        self.assertEqual([e.detector for e in ev], ["vehicle_exit"])

    def test_iou_helper(self):
        self.assertAlmostEqual(_box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(_box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)


if __name__ == "__main__":
    unittest.main()
