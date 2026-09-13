"""Production concealment callers must all use exclusive, stable bag ownership."""

from __future__ import annotations

from cvti.detector.core import PosePersonState
from cvti.retail.concealment import ConcealmentDetector, PoseFrame


def _pose_frame(track_id: int, x_offset: float, wrist_x: float) -> PoseFrame:
    return PoseFrame(
        track_id=track_id,
        timestamp=0.0,
        keypoints={
            "left_shoulder": (80.0 + x_offset, 100.0),
            "right_shoulder": (120.0 + x_offset, 100.0),
            "left_wrist": (wrist_x, 200.0),
            "right_wrist": None,
            "left_hip": (85.0 + x_offset, 250.0),
            "right_hip": (115.0 + x_offset, 250.0),
        },
        bbox=(60.0 + x_offset, 90.0, 140.0 + x_offset, 260.0),
    )


def _pose_person(track_id: int, x_offset: float, wrist_x: float) -> PosePersonState:
    frame = _pose_frame(track_id, x_offset, wrist_x)
    return PosePersonState(
        track_id=track_id,
        bbox=frame.bbox,
        timestamp=0.0,
        left_shoulder=frame.keypoints["left_shoulder"],
        right_shoulder=frame.keypoints["right_shoulder"],
        left_elbow=None,
        right_elbow=None,
        left_wrist=frame.keypoints["left_wrist"],
        right_wrist=None,
        max_wrist_speed=0.0,
        max_wrist_accel=0.0,
        max_arm_extension_ratio=0.0,
        weapon_labels=[],
        left_hip=frame.keypoints["left_hip"],
        right_hip=frame.keypoints["right_hip"],
    )


def _assert_one_owner(assessments) -> None:
    by_track = {assessment.track_id: assessment for assessment in assessments}
    assert by_track[1].components["f_bag"] > 0.0
    assert by_track[2].components["f_bag"] == 0.0


def test_retail_pipeline_scores_each_bag_for_only_one_track() -> None:
    from cvti.pipelines import retail_pipeline

    detector = ConcealmentDetector()
    poses = [_pose_frame(1, 0.0, 140.0), _pose_frame(2, 90.0, 150.0)]
    assessments = retail_pipeline.score_concealment_frame(
        detector, poses, 0.0, [(125.0, 170.0, 145.0, 235.0)]
    )
    _assert_one_owner(assessments)


def test_desktop_worker_scores_each_bag_for_only_one_track() -> None:
    from cvti.app import worker

    detector = ConcealmentDetector()
    poses = [_pose_frame(1, 0.0, 140.0), _pose_frame(2, 90.0, 150.0)]
    assessments = worker.score_concealment_frame(
        detector, poses, 0.0, [(125.0, 170.0, 145.0, 235.0)]
    )
    _assert_one_owner(assessments)


def test_detector_core_scores_each_bag_for_only_one_track() -> None:
    from cvti.detector import core

    detector = ConcealmentDetector()
    poses = [_pose_person(1, 0.0, 140.0), _pose_person(2, 90.0, 150.0)]
    assessments = core.score_concealment_frame(
        detector, poses, 0.0, [(125.0, 170.0, 145.0, 235.0)]
    )
    _assert_one_owner(assessments)
