from __future__ import annotations

import pytest

from cvti.detector.person_motion import (
    PersonMotion,
    PersonMotionTracker,
    SimultaneousMovementDetector,
)


FRAME = (100, 100)


def person(track_id: int, x: float) -> tuple[int, float, float, float, float]:
    return (track_id, x, 20.0, x + 20.0, 80.0)


def motion(
    track_id: int,
    bbox: tuple[float, float, float, float],
    *,
    moving: bool = True,
) -> PersonMotion:
    return PersonMotion(
        track_id=track_id,
        bbox=bbox,
        speed_ratio=0.08,
        moving=moving,
        moving_seconds=1.2 if moving else 0.0,
        zone_names=("permitted",),
    )


def test_sustained_displacement_becomes_moving() -> None:
    tracker = PersonMotionTracker(
        enter_speed_ratio=0.05,
        exit_speed_ratio=0.02,
        min_track_seconds=0.4,
    )

    assert not tracker.update([person(1, 10)], 0.0, FRAME)[0].moving
    result = tracker.update([person(1, 70)], 0.5, FRAME)[0]

    assert result.moving


def test_bbox_jitter_stays_stationary() -> None:
    tracker = PersonMotionTracker(
        enter_speed_ratio=0.05,
        exit_speed_ratio=0.02,
        min_track_seconds=0.4,
    )

    for index, x in enumerate((10, 11, 9, 12, 10)):
        motion = tracker.update([person(1, x)], index * 0.2, FRAME)[0]

    assert not motion.moving


def test_moving_state_uses_lower_exit_threshold() -> None:
    tracker = PersonMotionTracker(
        enter_speed_ratio=0.05,
        exit_speed_ratio=0.02,
        min_track_seconds=0.4,
        ema_alpha=1.0,
    )
    tracker.update([person(1, 10)], 0.0, FRAME)

    entered = tracker.update([person(1, 20)], 0.5, FRAME)[0]
    between_thresholds = tracker.update([person(1, 22)], 1.0, FRAME)[0]
    stopped = tracker.update([person(1, 22)], 1.5, FRAME)[0]

    assert entered.moving
    assert between_thresholds.moving
    assert between_thresholds.moving_seconds == pytest.approx(0.5)
    assert not stopped.moving
    assert stopped.moving_seconds == 0.0


def test_stale_track_expires_before_track_id_is_reused() -> None:
    tracker = PersonMotionTracker(
        enter_speed_ratio=0.05,
        exit_speed_ratio=0.02,
        min_track_seconds=0.4,
        ema_alpha=1.0,
        track_expiry_seconds=0.5,
    )
    tracker.update([person(1, 10)], 0.0, FRAME)

    assert tracker.update([], 0.6, FRAME) == []
    reused = tracker.update([person(1, 90)], 0.7, FRAME)[0]

    assert reused.speed_ratio == 0.0
    assert not reused.moving


def test_motion_snapshot_contains_bbox_and_zone_names() -> None:
    tracker = PersonMotionTracker(min_track_seconds=0.0)

    motion = tracker.update(
        [person(7, 10)],
        0.0,
        FRAME,
        zones_by_track={7: ["aisle", "checkout"]},
    )[0]

    assert motion.track_id == 7
    assert motion.bbox == (10.0, 20.0, 30.0, 80.0)
    assert motion.zone_names == ("aisle", "checkout")


def test_two_sustained_moving_tracks_fire_one_candidate() -> None:
    detector = SimultaneousMovementDetector(min_people=2, persistence_seconds=0.5)
    motions = [
        motion(1, (10.0, 20.0, 30.0, 80.0)),
        motion(2, (40.0, 10.0, 70.0, 90.0)),
    ]

    assert detector.update(motions, 0.0) is None
    assert detector.update(motions, 0.4) is None
    event = detector.update(motions, 0.5)

    assert event is not None
    assert event["track_ids"] == [1, 2]
    assert event["group_bbox"] == (10.0, 10.0, 70.0, 90.0)
    assert event["motions"][0] == {
        "track_id": 1,
        "speed_ratio": 0.08,
        "moving_seconds": 1.2,
        "zone_names": ["permitted"],
    }


def test_one_moving_track_does_not_fire() -> None:
    detector = SimultaneousMovementDetector(min_people=2, persistence_seconds=0.0)

    assert detector.update([motion(1, (10.0, 20.0, 30.0, 80.0))], 0.0) is None


def test_stationary_cluster_does_not_fire() -> None:
    detector = SimultaneousMovementDetector(min_people=2, persistence_seconds=0.0)
    clustered = [
        motion(1, (10.0, 10.0, 30.0, 80.0), moving=False),
        motion(2, (20.0, 10.0, 40.0, 80.0), moving=False),
        motion(3, (30.0, 10.0, 50.0, 80.0), moving=False),
    ]

    assert detector.update(clustered, 0.0) is None


def test_spread_out_moving_people_fire() -> None:
    detector = SimultaneousMovementDetector(min_people=2, persistence_seconds=0.0)
    spread_out = [
        motion(1, (0.0, 0.0, 20.0, 50.0)),
        motion(2, (900.0, 500.0, 950.0, 600.0)),
    ]

    event = detector.update(spread_out, 0.0)

    assert event is not None
    assert event["group_bbox"] == (0.0, 0.0, 950.0, 600.0)


def test_latch_resets_only_after_moving_count_drops() -> None:
    detector = SimultaneousMovementDetector(min_people=2, persistence_seconds=0.5)
    together = [
        motion(1, (10.0, 20.0, 30.0, 80.0)),
        motion(2, (40.0, 20.0, 60.0, 80.0)),
    ]

    assert detector.update(together, 0.0) is None
    assert detector.update(together, 0.5) is not None
    assert detector.update(together, 1.0) is None
    assert detector.update([together[0]], 1.1) is None
    assert detector.update(together, 1.2) is None
    assert detector.update(together, 1.7) is not None
