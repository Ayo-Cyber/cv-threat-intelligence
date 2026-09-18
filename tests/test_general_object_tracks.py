from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from cvti.detector.object_tracks import GeneralObjectTracker


NAMES = {0: "person", 1: "bag", 2: "bicycle"}


def detections(*rows) -> sv.Detections:
    return sv.Detections(
        xyxy=np.asarray([row[:4] for row in rows], dtype=float).reshape((-1, 4)),
        confidence=np.asarray([row[4] for row in rows], dtype=float),
        class_id=np.asarray([row[5] for row in rows], dtype=int),
    )


def tracker(**kwargs) -> GeneralObjectTracker:
    return GeneralObjectTracker(
        camera_id="front",
        session_id="shift-a",
        names=NAMES,
        **kwargs,
    )


def test_actual_bytetrack_preserves_identity_during_motion_and_short_occlusion() -> None:
    subject = tracker(expected_fps=5.0, lost_after_seconds=1.0)
    subject.update(detections((0, 0, 20, 20, 0.9, 1)), 0.0)
    subject.update(detections((2, 0, 22, 20, 0.9, 1)), 0.2)
    first = subject.snapshot(0.2)["tracks"][0]

    subject.update(sv.Detections.empty(), 0.4)
    assert subject.snapshot(0.4)["tracks"][0]["state"] == "lost"
    subject.update(detections((4, 0, 24, 20, 0.9, 1)), 0.6)

    resumed = subject.snapshot(0.6)["tracks"][0]
    assert resumed["id"] == first["id"]
    assert resumed["state"] == "observed"
    assert resumed["trajectory"] == [
        [0.0, 10.0, 10.0],
        [0.2, 12.0, 10.0],
        [0.6, 14.0, 10.0],
    ]


def test_expired_identity_cannot_be_resurrected_after_long_gap() -> None:
    subject = tracker(lost_after_seconds=0.5)
    subject.update(detections((0, 0, 20, 20, 0.9, 1)), 0.0)
    old_id = subject.snapshot(0.0)["tracks"][0]["id"]

    subject.update(detections((1, 0, 21, 20, 0.9, 1)), 0.6)
    snapshot = subject.snapshot(0.6)

    assert [(item["id"], item["state"]) for item in snapshot["tracks"]] == [
        (old_id, "ended"),
        ("front/shift-a/2", "observed"),
    ]
    assert snapshot["tracks"][0]["end_reason"] == "lost_timeout"


def test_empty_is_successful_absence_but_none_is_unavailable() -> None:
    subject = tracker(lost_after_seconds=1.0)
    assert subject.snapshot(0.0)["status"] == "starting"

    subject.update(None, 0.0)
    unavailable = subject.snapshot(0.0)
    assert unavailable["status"] == "unavailable"
    assert unavailable["last_success_at"] is None

    subject.update(sv.Detections.empty(), 0.2)
    successful = subject.snapshot(0.2)
    assert successful["status"] == "ok"
    assert successful["last_success_at"] == 0.2
    subject.update(None, 0.4)
    assert subject.snapshot(0.4)["status"] == "unavailable"
    assert subject.snapshot(1.21)["status"] == "stale"

    never_succeeded = tracker(lost_after_seconds=1.0)
    never_succeeded.update(None, 0.0)
    never_succeeded.update(None, 2.0)
    assert never_succeeded.snapshot(2.0)["status"] == "stale"


def test_persons_and_unknown_classes_are_excluded_and_counted_correctly() -> None:
    subject = tracker()
    raw = sv.Detections(
        xyxy=np.asarray(
            [
                (0, 0, 10, 10),
                (1, 1, 11, 11),
                (5, 5, 4, 10),
                (0, 0, 10, 10),
            ],
            dtype=float,
        ),
        confidence=np.asarray((0.99, 0.9, 0.8, np.nan)),
        class_id=np.asarray((0, 99, 1, 1)),
    )
    original_xyxy = raw.xyxy.copy()
    original_confidence = raw.confidence.copy()

    subject.update(raw, 0.0)

    assert subject.snapshot(0.0)["tracks"] == []
    assert subject.snapshot(0.0)["counters"]["invalid_detections"] == 3
    np.testing.assert_array_equal(raw.xyxy, original_xyxy)
    np.testing.assert_array_equal(raw.confidence, original_confidence)


def test_classes_and_tracker_instances_do_not_cross_associate() -> None:
    subject = tracker()
    subject.update(
        detections((0, 0, 20, 20, 0.9, 1), (0, 0, 20, 20, 0.8, 2)),
        0.0,
    )
    tracks = subject.snapshot(0.0)["tracks"]

    assert [(item["id"], item["class_id"]) for item in tracks] == [
        ("front/shift-a/1", 1),
        ("front/shift-a/2", 2),
    ]
    assert subject._trackers[1] is not subject._trackers[2]

    other = GeneralObjectTracker(camera_id="rear", session_id="shift-a", names=NAMES)
    other.update(detections((0, 0, 20, 20, 0.9, 1)), 0.0)
    assert other.snapshot(0.0)["tracks"][0]["id"] == "rear/shift-a/1"


def test_expiry_in_one_class_does_not_reset_another_class_tracker() -> None:
    subject = tracker(lost_after_seconds=0.5)
    subject.update(
        detections((0, 0, 20, 20, 0.9, 1), (50, 0, 70, 20, 0.9, 2)),
        0.0,
    )
    bicycle_id = subject.snapshot(0.0)["tracks"][1]["id"]
    subject.update(detections((51, 0, 71, 20, 0.9, 2)), 0.4)
    subject.update(detections((52, 0, 72, 20, 0.9, 2)), 0.6)

    bicycles = [
        item for item in subject.snapshot(0.6)["tracks"] if item["class_id"] == 2
    ]
    assert len(bicycles) == 1
    assert bicycles[0]["id"] == bicycle_id
    assert bicycles[0]["state"] == "observed"


def test_one_same_class_expiry_preserves_other_ids_and_public_bounds() -> None:
    subject = tracker(
        expected_fps=5.0,
        lost_after_seconds=0.5,
        max_tracks=3,
        max_ended_tracks=1,
    )
    subject.update(
        detections(
            (0, 0, 20, 20, 0.9, 1),
            (100, 0, 120, 20, 0.9, 1),
            (200, 0, 220, 20, 0.9, 1),
        ),
        0.0,
    )
    initial = subject.snapshot(0.0)["tracks"]
    stable_ids = [initial[0]["id"], initial[1]["id"]]
    expired_id = initial[2]["id"]
    saw_expected_expiry = False

    for step in range(1, 31):
        timestamp = step * 0.2
        subject.update(
            detections(
                (step, 0, 20 + step, 20, 0.9, 1),
                (100 + step, 0, 120 + step, 20, 0.9, 1),
            ),
            timestamp,
        )
        snapshot = subject.snapshot(timestamp)
        live = [item for item in snapshot["tracks"] if item["state"] != "ended"]
        ended = [item for item in snapshot["tracks"] if item["state"] == "ended"]
        assert len(live) <= subject.max_tracks
        assert len(ended) <= subject.max_ended_tracks
        if ended:
            assert [(item["id"], item["end_reason"]) for item in ended] == [
                (expired_id, "lost_timeout")
            ]
            saw_expected_expiry = True

    snapshot = subject.snapshot(6.0)
    observed = [item for item in snapshot["tracks"] if item["state"] == "observed"]
    assert [item["id"] for item in observed] == stable_ids
    assert saw_expected_expiry


def test_snapshot_projects_age_without_mutating_and_overlays_require_fresh_observation() -> None:
    subject = tracker(lost_after_seconds=1.0, overlay_max_age_seconds=0.25)
    subject.update(detections((0.4, 1.6, 20.2, 21.8, 0.9, 1)), 0.0)

    overlay = subject.overlays(0.2)[0]
    assert overlay == {
        "track_id": -1,
        "bbox": (0, 2, 20, 22),
        "label": "bag O1",
        "colour": (255, 180, 0),
        "namespace": "object",
    }
    assert subject.overlays(0.3) == []
    assert subject.snapshot(0.1)["tracks"][0]["state"] == "observed"
    assert subject.snapshot(1.1)["tracks"][0]["state"] == "ended"
    assert subject.snapshot(0.0)["tracks"][0]["state"] == "observed"


def test_unavailable_and_successful_empty_updates_suppress_observed_tracks() -> None:
    subject = tracker(lost_after_seconds=1.0)
    subject.update(detections((0, 0, 20, 20, 0.9, 1)), 0.0)
    track_id = subject.snapshot(0.01)["tracks"][0]["id"]

    subject.update(None, 0.1)
    unavailable = subject.snapshot(0.1)
    assert unavailable["status"] == "unavailable"
    assert unavailable["tracks"][0]["state"] == "lost"
    assert subject.overlays(0.1) == []

    subject.update(detections((1, 0, 21, 20, 0.9, 1)), 0.2)
    assert subject.snapshot(0.21)["tracks"][0]["id"] == track_id
    assert subject.snapshot(0.21)["tracks"][0]["state"] == "observed"

    subject.update(sv.Detections.empty(), 0.3)
    absent = subject.snapshot(0.3)
    assert absent["status"] == "ok"
    assert absent["tracks"][0]["state"] == "lost"
    assert subject.overlays(0.3) == []


def test_empty_name_uses_class_id_fallback() -> None:
    subject = GeneralObjectTracker(
        camera_id="front",
        session_id="shift-a",
        names={1: ""},
        person_class_ids=(),
    )
    subject.update(detections((0, 0, 20, 20, 0.9, 1)), 0.0)

    assert subject.snapshot(0.0)["tracks"][0]["label"] == "class_1"


def test_duplicate_is_noop_backwards_rejected_and_reset_allows_replay() -> None:
    subject = tracker()
    subject.update(detections((0, 0, 20, 20, 0.9, 1)), 10.0)
    subject.update(detections((50, 0, 70, 20, 0.9, 1)), 10.0)
    assert subject.snapshot(10.0)["tracks"][0]["bbox"] == [0.0, 0.0, 20.0, 20.0]

    with pytest.raises(ValueError, match="backwards"):
        subject.update(sv.Detections.empty(), 9.0)
    assert subject.snapshot(10.0)["last_update_at"] == 10.0

    subject.reset(1.0)
    assert subject.snapshot(1.0)["status"] == "starting"
    subject.update(detections((0, 0, 20, 20, 0.9, 1)), 1.0)
    assert subject.snapshot(1.0)["tracks"][-1]["id"] == "front/shift-a/2"


def test_input_and_trajectory_caps_keep_highest_confidence() -> None:
    subject = tracker(max_tracks=2, max_detections=3, trajectory_points=2)
    subject.update(
        detections(
            (0, 0, 10, 10, 0.4, 1),
            (20, 0, 30, 10, 0.9, 1),
            (40, 0, 50, 10, 0.8, 1),
            (60, 0, 70, 10, 0.7, 1),
        ),
        0.0,
    )
    snapshot = subject.snapshot(0.0)
    assert [item["confidence"] for item in snapshot["tracks"]] == [0.9, 0.8]
    assert snapshot["counters"]["dropped_detections"] == 2

    subject.update(detections((21, 0, 31, 10, 0.9, 1)), 0.1)
    subject.update(detections((22, 0, 32, 10, 0.9, 1)), 0.2)
    observed = [item for item in subject.snapshot(0.2)["tracks"] if item["state"] == "observed"]
    assert len(observed[0]["trajectory"]) == 2


def test_internal_track_bound_triggers_counted_reset_without_resurrection() -> None:
    subject = tracker(max_tracks=1, lost_after_seconds=5.0)
    subject.update(detections((0, 0, 10, 10, 0.9, 1)), 0.0)
    first_id = subject.snapshot(0.0)["tracks"][0]["id"]
    subject.update(detections((100, 0, 110, 10, 0.9, 1)), 0.2)

    snapshot = subject.snapshot(0.2)
    assert snapshot["counters"]["capacity_resets"] == 1
    assert snapshot["tracks"][0]["id"] == first_id
    assert snapshot["tracks"][0]["state"] == "ended"
    assert snapshot["tracks"][0]["end_reason"] == "capacity_reset"
    assert snapshot["tracks"][1]["id"] == "front/shift-a/2"


def test_ended_tracks_are_bounded_by_age_and_count() -> None:
    subject = tracker(max_ended_tracks=1, ended_retention_seconds=0.5)
    subject.update(detections((0, 0, 10, 10, 0.9, 1)), 0.0)
    subject.reset(0.1, reason="seek")
    subject.update(detections((20, 0, 30, 10, 0.9, 1)), 0.2)
    subject.reset(0.3, reason="seek")

    ended = subject.snapshot(0.3)["tracks"]
    assert len(ended) == 1
    assert ended[0]["id"] == "front/shift-a/2"
    assert subject.snapshot(0.81)["tracks"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"camera_id": ""},
        {"session_id": ""},
        {"expected_fps": float("nan")},
        {"lost_after_seconds": 0.0},
        {"overlay_max_age_seconds": -1.0},
        {"trajectory_points": 0},
        {"max_tracks": True},
        {"max_detections": 0},
        {"ended_retention_seconds": float("inf")},
        {"max_ended_tracks": 0},
    ],
)
def test_configuration_rejects_invalid_bounds(kwargs) -> None:
    values = {"camera_id": "front", "session_id": "shift-a", "names": NAMES}
    values.update(kwargs)
    with pytest.raises(ValueError):
        GeneralObjectTracker(**values)
