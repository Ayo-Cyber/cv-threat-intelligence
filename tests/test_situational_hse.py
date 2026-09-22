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


def _fire_detector(**kw) -> FireSmokeCandidateDetector:
    return FireSmokeCandidateDetector(min_frames=2, min_hot_area_ratio=0.015,
                                      warmup_frames=3, **kw)


def _learn(detector: FireSmokeCandidateDetector, frame: np.ndarray, frames: int = 6) -> None:
    for i in range(frames):
        assert detector.update(frame, timestamp=float(i)) is None


def _flame(base: np.ndarray, rng: np.random.Generator, t: int) -> np.ndarray:
    """A bright orange region that flickers: size and edge jitter every frame."""
    frame = base.copy()
    jitter = int(rng.integers(0, 12))
    frame[40:80 + jitter, 50:100 + (t % 3) * 6] = (0, 140, 255)
    return frame


def test_fire_smoke_candidate_detector_fires_when_flame_appears_and_flickers() -> None:
    detector = _fire_detector()
    rng = np.random.default_rng(7)
    room = np.full((120, 160, 3), 40, dtype=np.uint8)
    _learn(detector, room)

    results = [detector.update(_flame(room, rng, t), timestamp=10.0 + t) for t in range(4)]
    fired = [r for r in results if r is not None]

    assert results[0] is None                       # min_frames=2: one frame is not a fire
    assert len(fired) == 1, results
    assert fired[0]["kind"] == "fire_smoke"
    assert fired[0]["hot_area_ratio"] >= 0.015
    assert fired[0]["change_ratio"] >= detector.min_change_ratio


def test_fire_smoke_candidate_detector_ignores_the_scenes_own_warm_colours() -> None:
    """The pilot's case: a warm-lit room or an orange wall is the background.
    It was a candidate on EVERY frame before 22 Sep."""
    detector = _fire_detector()
    room = np.full((120, 160, 3), 40, dtype=np.uint8)
    room[20:100, 30:140] = (0, 140, 255)           # a large orange wall, always there

    for ts in range(40):
        assert detector.update(room, timestamp=float(ts)) is None


def test_fire_smoke_candidate_detector_ignores_a_warm_object_that_appears_and_holds_still() -> None:
    detector = _fire_detector()
    room = np.full((120, 160, 3), 40, dtype=np.uint8)
    _learn(detector, room)
    parked = room.copy()
    parked[40:80, 50:100] = (0, 140, 255)          # an orange car parks: new, but static

    results = [detector.update(parked, timestamp=10.0 + t) for t in range(8)]
    assert all(r is None for r in results), results


def test_fire_smoke_candidate_detector_ignores_monochrome_ir_night_scenes() -> None:
    """Night IR video is grey everywhere; the colour 'smoke' path is meaningless
    there and read the whole yard as smoke (116/116 frames of the driveway clip)."""
    detector = _fire_detector()
    rng = np.random.default_rng(3)
    for ts in range(40):
        grey = rng.integers(90, 200, size=(120, 160), dtype=np.uint8)   # IR noise, drifting
        frame = np.stack([grey, grey, grey], axis=-1)
        assert detector.update(frame, timestamp=float(ts)) is None


def test_fire_smoke_candidate_detector_ignores_dark_normal_frame() -> None:
    detector = _fire_detector()
    frame = np.full((120, 160, 3), 40, dtype=np.uint8)

    for ts in range(10):
        assert detector.update(frame, timestamp=float(ts)) is None


def test_fire_smoke_candidate_detector_rearms_only_after_the_window() -> None:
    detector = _fire_detector(rearm_seconds=300.0)
    rng = np.random.default_rng(11)
    room = np.full((120, 160, 3), 40, dtype=np.uint8)
    _learn(detector, room)

    def episode(t0: float) -> list:
        out = [detector.update(_flame(room, rng, t), timestamp=t0 + t) for t in range(4)]
        for t in range(30):                          # scene returns to normal, background re-learns
            detector.update(room, timestamp=t0 + 4 + t)
        return [r for r in out if r is not None]

    assert len(episode(10.0)) == 1
    assert episode(60.0) == []                      # same camera, 50 s later: held
    assert len(episode(400.0)) == 1                 # past the window: reported again


def test_fire_smoke_candidate_detector_ignores_plain_grey_wall_scene() -> None:
    detector = _fire_detector()
    frame = np.full((120, 160, 3), 130, dtype=np.uint8)

    for ts in range(10):
        assert detector.update(frame, timestamp=float(ts)) is None
