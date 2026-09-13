"""Validates concealment.py on synthetic skeleton sequences. No torch needed.

Run:  python tests/test_concealment.py

Geometry: torso with shoulders at y=100, hips at y=250 (body scale = 150px, torso axis
x=100). We drive a single active wrist through motions and assert the score behaves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.retail import concealment  # noqa: E402

ConcealmentDetector = concealment.ConcealmentDetector
PoseFrame = concealment.PoseFrame

_SHOULDERS = {"left_shoulder": (80.0, 100.0), "right_shoulder": (120.0, 100.0)}
_HIPS = {"left_hip": (85.0, 250.0), "right_hip": (115.0, 250.0)}


def frame(ts: float, wrist: tuple[float, float], hips: bool = True) -> PoseFrame:
    kp: dict = {**_SHOULDERS, "left_wrist": wrist, "right_wrist": None}
    if hips:
        kp.update(_HIPS)
    return PoseFrame(track_id=1, timestamp=ts, keypoints=kp, bbox=(60, 90, 140, 260))


def translated_frame(track_id: int, x_offset: float, wrist: tuple[float, float]) -> PoseFrame:
    return PoseFrame(
        track_id=track_id,
        timestamp=0.0,
        keypoints={
            "left_shoulder": (80.0 + x_offset, 100.0),
            "right_shoulder": (120.0 + x_offset, 100.0),
            "left_wrist": wrist,
            "right_wrist": None,
            "left_hip": (85.0 + x_offset, 250.0),
            "right_hip": (115.0 + x_offset, 250.0),
        },
        bbox=(60.0 + x_offset, 90.0, 140.0 + x_offset, 260.0),
    )


def _run(detector: ConcealmentDetector, frames: list[PoseFrame]):
    last = None
    for f in frames:
        last = detector.update([f], f.timestamp)[0]
    return last


def test_concealment_motion_fires() -> None:
    det = ConcealmentDetector()
    frames = []
    t = 0.0
    for _ in range(4):                       # reach OUT to the shelf
        frames.append(frame(t, (200.0, 110.0))); t += 0.1
    for _ in range(9):                       # pull hand IN to the waist and hold
        frames.append(frame(t, (105.0, 245.0))); t += 0.1
    result = _run(det, frames)
    assert result.score >= 0.6, result.score
    assert result.candidate, "a reach-then-conceal-to-waist motion should become a candidate"
    assert any("waist" in r for r in result.reasons), result.reasons
    assert not result.limited
    print(f"PASS concealment motion fires (score={result.score:.2f}, components={result.components})")


def test_normal_browsing_does_not_fire() -> None:
    det = ConcealmentDetector()
    # Hand stays up at the shelf the whole time, never goes to the waist.
    frames = [frame(i * 0.1, (180.0, 110.0)) for i in range(15)]
    result = _run(det, frames)
    assert result.score < 0.6, result.score
    assert not result.candidate, "browsing at a shelf must not fire a concealment candidate"
    print(f"PASS normal browsing stays quiet (score={result.score:.2f})")


def test_phone_to_pocket_stays_below_candidate_persistence() -> None:
    det = ConcealmentDetector()
    frames = [frame(i * 0.1, (105.0, 130.0)) for i in range(4)]
    frames += [frame((i + 4) * 0.1, (105.0, 245.0)) for i in range(7)]

    result = _run(det, frames)

    assert not result.candidate, "putting a phone into a pocket must not persist as concealment"
    assert det._over_threshold[1] < det.min_candidate_frames


def test_brief_clothing_adjustment_stays_below_candidate_persistence() -> None:
    det = ConcealmentDetector()
    frames = [frame(i * 0.1, (105.0, 245.0)) for i in range(7)]

    result = _run(det, frames)

    assert not result.candidate, "a brief clothing adjustment must not persist as concealment"
    assert det._over_threshold[1] < det.min_candidate_frames


def test_openly_carried_item_stays_below_candidate_persistence() -> None:
    det = ConcealmentDetector()
    frames = [frame(i * 0.1, (180.0, 170.0)) for i in range(15)]

    result = _run(det, frames)

    assert not result.candidate, "openly carrying an item must not become concealment"
    assert det._over_threshold[1] < det.min_candidate_frames


def test_occluded_hips_degrades_gracefully() -> None:
    det = ConcealmentDetector()
    # No hips visible (caption banner / occlusion). Same motion, but waist features blind.
    frames = []
    t = 0.0
    for _ in range(4):
        frames.append(frame(t, (200.0, 110.0), hips=False)); t += 0.1
    for _ in range(9):
        frames.append(frame(t, (105.0, 245.0), hips=False)); t += 0.1
    result = _run(det, frames)
    assert result.limited, "no-hips case should set the limited flag"
    assert result.score < 0.6, "without hips the waist signal cannot confirm -> degraded"
    assert any("LIMITED" in r for r in result.reasons), result.reasons
    print(f"PASS occluded hips degrade gracefully (score={result.score:.2f}, limited={result.limited})")


def test_empty_window_is_zero() -> None:
    det = ConcealmentDetector()
    score, reasons, components, limited, destination = det.score_window([])
    assert score == 0.0 and not reasons and not limited and destination is None
    print("PASS empty window scores zero")


# A personal bag sitting at the person's side (not at the hip/waist line).
_BAG_BBOX = (180.0, 170.0, 240.0, 235.0)
_DROP_IN_BAG = (210.0, 200.0)   # inside the bag bbox, and far enough from the hip to not be "waist"


def _run_with_bags(detector, frames, bag_bboxes):
    last = None
    for f in frames:
        last = detector.update([f], f.timestamp, bag_bboxes=bag_bboxes)[0]
    return last


def test_bag_concealment_fires_with_destination_bag() -> None:
    det = ConcealmentDetector()
    frames = []
    t = 0.0
    for _ in range(4):                                   # reach OUT to the shelf
        frames.append(frame(t, (280.0, 150.0))); t += 0.1
    for _ in range(9):                                   # bring hand INTO the bag and hold
        frames.append(frame(t, _DROP_IN_BAG)); t += 0.1
    result = _run_with_bags(det, frames, [_BAG_BBOX])
    assert result.candidate, "reach-then-put-in-personal-bag should fire"
    assert result.destination == "bag", result.destination
    assert any("bag" in r for r in result.reasons), result.reasons
    print(f"PASS bag concealment fires (score={result.score:.2f}, dest={result.destination})")


def test_trolley_destination_is_safe() -> None:
    # IDENTICAL hand motion, but the destination is a trolley/basket — NOT a personal bag,
    # so no bag bbox is passed. Putting goods in a cart is normal shopping; must NOT fire.
    det = ConcealmentDetector()
    frames = []
    t = 0.0
    for _ in range(4):
        frames.append(frame(t, (280.0, 150.0))); t += 0.1
    for _ in range(9):
        frames.append(frame(t, _DROP_IN_BAG)); t += 0.1
    result = _run_with_bags(det, frames, None)           # no personal bag at the destination
    assert not result.candidate, "placing an item in a trolley must not fire a concealment candidate"
    assert result.destination is None, result.destination
    print(f"PASS trolley destination stays safe (score={result.score:.2f}, dest={result.destination})")


def test_personal_bag_boxes_extracts_only_coco_personal_bags() -> None:
    detections = sv.Detections(
        xyxy=np.array([
            [10.0, 10.0, 80.0, 180.0],
            [130.0, 100.0, 180.0, 160.0],
            [190.0, 80.0, 290.0, 180.0],
        ]),
        class_id=np.array([0, 26, 56]),
        confidence=np.array([0.95, 0.8, 0.9]),
    )

    assert concealment.personal_bag_boxes(detections) == [(130.0, 100.0, 180.0, 160.0)]


def test_bag_evidence_is_grounded_to_the_nearby_pose_track() -> None:
    bag = (145.0, 170.0, 190.0, 235.0)
    nearby = frame(0.0, (160.0, 200.0))
    far_away = PoseFrame(
        track_id=2,
        timestamp=0.0,
        keypoints={
            "left_shoulder": (380.0, 100.0),
            "right_shoulder": (420.0, 100.0),
            "left_wrist": (400.0, 200.0),
            "right_wrist": None,
            "left_hip": (385.0, 250.0),
            "right_hip": (415.0, 250.0),
        },
        bbox=(360.0, 90.0, 440.0, 260.0),
    )
    poses = [nearby, far_away]
    bags_by_track = {
        pose.track_id: concealment.bags_for_pose(pose, [bag]) for pose in poses
    }

    assessments = ConcealmentDetector().update(
        poses,
        timestamp=0.0,
        bag_bboxes_by_track=bags_by_track,
    )

    assert assessments[0].components["f_bag"] > 0.0
    assert assessments[0].associated_bag == bag
    assert assessments[1].components["f_bag"] == 0.0
    assert assessments[1].associated_bag is None


def test_frame_level_bag_association_assigns_one_owner_among_adjacent_shoppers() -> None:
    bag = (125.0, 170.0, 145.0, 235.0)
    poses = [
        translated_frame(1, 0.0, (125.0, 200.0)),
        translated_frame(2, 90.0, (160.0, 200.0)),
    ]

    bags_by_track = concealment.associate_bags_to_tracks(poses, [bag])

    assert bags_by_track == {1: [bag], 2: []}


def test_frame_level_bag_association_leaves_an_equidistant_tie_unassigned() -> None:
    bag = (135.0, 170.0, 155.0, 235.0)
    poses = [
        translated_frame(1, 0.0, (135.0, 200.0)),
        translated_frame(2, 90.0, (155.0, 200.0)),
    ]

    bags_by_track = concealment.associate_bags_to_tracks(poses, [bag])

    assert bags_by_track == {1: [], 2: []}


def test_physical_bag_owner_does_not_jitter_between_near_equal_tracks() -> None:
    """A frame-local nearest-track flip must not seed both temporal buffers."""
    bag = (135.0, 170.0, 155.0, 235.0)
    detector = ConcealmentDetector(window_seconds=1.2)

    for index in range(10):
        # Alternate a tiny horizontal shift that makes each shopper nearest on
        # every other sampled frame. Both remain plausible owners throughout.
        left_offset, right_offset = ((0.0, 91.0) if index % 2 == 0 else (-1.0, 90.0))
        poses = [
            translated_frame(1, left_offset, (140.0, 200.0)),
            translated_frame(2, right_offset, (150.0, 200.0)),
        ]
        timestamp = index * 0.1
        poses = [
            PoseFrame(
                track_id=pose.track_id,
                timestamp=timestamp,
                keypoints=pose.keypoints,
                bbox=pose.bbox,
            )
            for pose in poses
        ]
        detector.update_with_bag_detections(poses, timestamp, [bag])

    tracks_with_bag_evidence = {
        track_id
        for track_id, window in detector._buffers.items()
        if any(feature.hand_to_bag is not None for feature in window)
    }
    assert tracks_with_bag_evidence == {1}


def test_physical_bag_can_transfer_after_prior_evidence_ages_out() -> None:
    bag = (135.0, 170.0, 155.0, 235.0)
    detector = ConcealmentDetector(window_seconds=1.2, state_grace_seconds=1.5)
    first = [
        translated_frame(1, 0.0, (140.0, 200.0)),
        translated_frame(2, 91.0, (150.0, 200.0)),
    ]
    detector.update_with_bag_detections(first, 0.0, [bag])

    second = [
        translated_frame(1, -1.0, (140.0, 200.0)),
        translated_frame(2, 90.0, (150.0, 200.0)),
    ]
    second = [
        PoseFrame(poses.track_id, 2.0, poses.keypoints, poses.bbox)
        for poses in second
    ]
    assessments = detector.update_with_bag_detections(second, 2.0, [bag])

    by_track = {assessment.track_id: assessment for assessment in assessments}
    assert by_track[1].components["f_bag"] == 0.0
    assert by_track[2].components["f_bag"] > 0.0


def test_assessment_reports_the_bag_that_drove_the_temporal_score() -> None:
    scoring_bag = (180.0, 170.0, 240.0, 235.0)
    farther_bag = (250.0, 170.0, 300.0, 235.0)
    detector = ConcealmentDetector()
    detector.update(
        [frame(0.0, (210.0, 200.0))],
        timestamp=0.0,
        bag_bboxes=[farther_bag, scoring_bag],
    )

    result = detector.update(
        [frame(0.1, (210.0, 200.0))],
        timestamp=0.1,
        bag_bboxes=[],
    )[0]

    assert result.destination == "bag"
    assert result.associated_bag == scoring_bag


def test_track_specific_bags_take_precedence_over_legacy_global_bags() -> None:
    result = ConcealmentDetector().update(
        [frame(0.0, _DROP_IN_BAG)],
        timestamp=0.0,
        bag_bboxes=[_BAG_BBOX],
        bag_bboxes_by_track={1: []},
    )[0]

    assert result.components["f_bag"] == 0.0
    assert result.associated_bag is None


def test_unsampled_gap_does_not_erase_concealment_history() -> None:
    det = ConcealmentDetector(state_grace_seconds=1.5)
    det.update([frame(0.0, (200.0, 110.0))], 0.0)
    assert 1 in det._buffers
    det.expire(0.2)
    assert 1 in det._buffers


def test_stale_track_is_expired_after_grace() -> None:
    det = ConcealmentDetector(state_grace_seconds=1.0)
    det.update([frame(0.0, (200.0, 110.0))], 0.0)
    det.expire(1.01)
    assert 1 not in det._buffers
    assert 1 not in det._over_threshold
    assert 1 not in det._last_seen


def test_empty_update_expires_stale_track_for_standalone_callers() -> None:
    det = ConcealmentDetector(state_grace_seconds=1.0)
    det.update([frame(0.0, (200.0, 110.0))], 0.0)
    det.update([], 1.01, bag_bboxes=[])
    assert 1 not in det._buffers
    assert 1 not in det._over_threshold
    assert 1 not in det._last_seen


if __name__ == "__main__":
    test_concealment_motion_fires()
    test_normal_browsing_does_not_fire()
    test_occluded_hips_degrades_gracefully()
    test_empty_window_is_zero()
    test_bag_concealment_fires_with_destination_bag()
    test_trolley_destination_is_safe()
    test_unsampled_gap_does_not_erase_concealment_history()
    test_stale_track_is_expired_after_grace()
    test_empty_update_expires_stale_track_for_standalone_callers()
    print("\nAll concealment tests passed.")
