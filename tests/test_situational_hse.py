from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.detector.situational import (
    CrowdFormationDetector,
    FireSmokeCandidateDetector,
    RunningPanicDetector,
)


def test_running_detector_fires_after_sustained_fast_person_motion() -> None:
    detector = RunningPanicDetector(min_speed_ratio=0.08, min_frames=3)
    frame_shape = (480, 640, 3)
    fired = None

    for idx, x in enumerate([50, 120, 190, 260]):
        fired = detector.update(
            track_id=7,
            bbox=(x, 180, x + 60, 300),
            timestamp=float(idx),
            frame_shape=frame_shape,
        )

    assert fired is not None
    assert fired["track_id"] == 7
    assert fired["kind"] == "running"
    assert fired["speed_ratio"] >= 0.08


def test_running_detector_ignores_slow_person_motion() -> None:
    detector = RunningPanicDetector(min_speed_ratio=0.22, min_frames=3)
    frame_shape = (480, 640, 3)

    for idx, x in enumerate([50, 55, 61, 67, 73]):
        assert detector.update(7, (x, 180, x + 60, 300), float(idx), frame_shape) is None


def test_crowd_formation_detector_fires_for_persistent_cluster() -> None:
    detector = CrowdFormationDetector(min_people=4, min_frames=2, max_cluster_ratio=0.22)
    frame_shape = (480, 640, 3)
    people = [
        {"track_id": 1, "bbox": (100, 100, 150, 220)},
        {"track_id": 2, "bbox": (155, 105, 205, 225)},
        {"track_id": 3, "bbox": (115, 230, 165, 350)},
        {"track_id": 4, "bbox": (170, 235, 220, 355)},
    ]

    assert detector.update(people, timestamp=0.0, frame_shape=frame_shape) is None
    fired = detector.update(people, timestamp=1.0, frame_shape=frame_shape)

    assert fired is not None
    assert fired["kind"] == "crowd_formation"
    assert fired["people_count"] == 4


def test_crowd_formation_detector_ignores_spread_out_people() -> None:
    detector = CrowdFormationDetector(min_people=4, min_frames=2, max_cluster_ratio=0.22)
    frame_shape = (480, 640, 3)
    people = [
        {"track_id": 1, "bbox": (10, 10, 60, 130)},
        {"track_id": 2, "bbox": (540, 20, 600, 140)},
        {"track_id": 3, "bbox": (20, 340, 80, 460)},
        {"track_id": 4, "bbox": (520, 330, 600, 460)},
    ]

    for ts in [0.0, 1.0, 2.0]:
        assert detector.update(people, timestamp=ts, frame_shape=frame_shape) is None


def test_fire_smoke_candidate_detector_fires_for_persistent_flame_colored_region() -> None:
    detector = FireSmokeCandidateDetector(min_frames=2, min_hot_area_ratio=0.015)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    # OpenCV BGR: bright orange/yellow block.
    frame[40:80, 50:100] = (0, 140, 255)

    assert detector.update(frame, timestamp=0.0) is None
    fired = detector.update(frame, timestamp=1.0)

    assert fired is not None
    assert fired["kind"] == "fire_smoke"
    assert fired["hot_area_ratio"] >= 0.015


def test_fire_smoke_candidate_detector_ignores_dark_normal_frame() -> None:
    detector = FireSmokeCandidateDetector(min_frames=2, min_hot_area_ratio=0.015)
    frame = np.full((120, 160, 3), 40, dtype=np.uint8)

    for ts in [0.0, 1.0, 2.0]:
        assert detector.update(frame, timestamp=ts) is None


def test_fire_smoke_candidate_detector_ignores_plain_grey_wall_scene() -> None:
    detector = FireSmokeCandidateDetector(min_frames=2, min_hot_area_ratio=0.015)
    frame = np.full((120, 160, 3), 130, dtype=np.uint8)

    for ts in [0.0, 1.0, 2.0]:
        assert detector.update(frame, timestamp=ts) is None
