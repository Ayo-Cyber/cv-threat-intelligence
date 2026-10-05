"""Concealment evidence must not cross independent tracking ID namespaces."""
from unittest.mock import Mock, patch

import numpy as np
import pytest
import supervision as sv

from cvti.contracts import CandidateAlert
from cvti.serving.camera import PerCameraState


@pytest.mark.parametrize("subject", [(60, 10, 90, 90), None])
def test_pose_subject_wins_over_colliding_person_id(subject):
    metadata = {"subject_bbox": subject} if subject is not None else {}
    alert = CandidateAlert(rule_name="product_concealment", priority="high",
                           detector="concealment", title="Possible concealment",
                           person_id=7, object_label=None, timestamp=1,
                           metadata=metadata)
    engine = Mock()
    engine.evaluate.return_value = [alert]
    engine.context_decisions = []
    state = PerCameraState("test", engine, person_filter=False, theft=True)
    detected = sv.Detections(xyxy=np.array([[0., 0., 40., 90.]]),
                             confidence=np.array([.99]), class_id=np.array([0]),
                             tracker_id=np.array([7]))
    with patch.object(state._tracker, "update_with_detections", return_value=detected), \
         patch.object(PerCameraState, "_assessment_events", return_value=[object()]), \
         patch("cvti.verification.frame_select.append_subject_crop", return_value=[]) as crop:
        state.process(detected, np.zeros((100, 100, 3), dtype=np.uint8), 1.)
    assert crop.call_args.args[2] == subject
