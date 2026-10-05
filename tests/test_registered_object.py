import numpy as np
import pytest
import cv2

from cvti.detector.registered_object import RegisteredObjectMonitor


REGION = (30, 30, 70, 70)


def frame(present=True):
    image = np.random.default_rng(2).integers(70, 90, (100, 100, 3), dtype=np.uint8)
    image[30:70, 30:70] = 80
    if present:
        image[34:66, 34:66] = 220
        image[44:56, 44:56] = 20
    return image


def monitor():
    m = RegisteredObjectMonitor(confirm_seconds=3)
    m.register([frame()] * 3, REGION, [[], [], []])
    return m


def test_change_without_object_class_and_only_one_transition():
    m = monitor()
    assert m.update(frame(), 0, []).state == "present"
    for t in (1, 2, 3):
        assert not m.update(frame(False), t, []).changed
    assert m.update(frame(False), 4, []).changed
    assert not m.update(frame(False), 5, []).changed
    assert m.update(frame(), 6, []).state == "review_required"


def test_large_person_covering_small_region_is_occlusion():
    m = monitor()
    assert m.update(frame(False), 0, [(0, 0, 100, 100)]).state == "occluded"


def test_person_handling_does_not_invalidate_before_clear_view():
    m = monitor()
    m.update(frame(), 0, [])
    obstructed = np.full_like(frame(), 200)
    assert m.update(obstructed, 1, [(0, 0, 100, 100)]).state == "occluded"
    for t in (2, 3, 4):
        assert not m.update(frame(False), t, []).changed
    assert m.update(frame(False), 5, []).changed


def test_camera_change_after_occlusion_still_requires_revalidation():
    m = monitor()
    changed = np.full_like(frame(), 200)
    m.update(changed, 0, [(0, 0, 100, 100)])
    result = m.update(changed, 1, [])
    assert result.state == "revalidation_required"
    assert result.reason == "insufficient_image_detail"
    assert m.update(frame(), 2, []).reason == result.reason


def test_gap_reports_persistent_reason():
    m = monitor()
    m.update(frame(), 0, [])
    assert m.update(frame(), 10, []).reason == "frame_timing_discontinuity"
    assert m.update(frame(), 11, []).reason == "frame_timing_discontinuity"


def test_registration_rejects_changing_background_even_with_stable_object():
    original = frame()
    shifted = np.clip(original.astype(float) + 50, 0, 255).astype(np.uint8)
    shifted[30:70, 30:70] = original[30:70, 30:70]
    m = RegisteredObjectMonitor()
    with pytest.raises(ValueError, match="background is not stable"):
        m.register([original, shifted, shifted], REGION, [[], [], []])
    assert m.update(original, 0, []).state == "revalidation_required"


def test_moderate_uniform_exposure_preserves_reference_and_detects_removal():
    m = monitor()
    def brighter(present):
        return np.clip(frame(present).astype(float) + 25, 0, 255).astype(np.uint8)
    assert m.update(brighter(True), 0, []).state == "present"
    for t in (1, 2, 3):
        assert not m.update(brighter(False), t, []).changed
    assert m.update(brighter(False), 4, []).changed


def test_large_exposure_shift_still_requires_revalidation():
    m = monitor()
    shifted = np.clip(frame().astype(float) + 60, 0, 255).astype(np.uint8)
    for t in range(3):
        assert m.update(shifted, t, []).state == "waiting_for_stable_view"
    assert m.update(shifted, 3, []).reason == "scene_or_lighting_changed"


def test_nonuniform_background_change_is_not_exposure():
    m = monitor()
    changed = frame()
    changed[:, :30] = 220
    for t in range(3):
        assert m.update(changed, t, []).state == "waiting_for_stable_view"
    assert m.update(changed, 3, []).reason == "scene_or_lighting_changed"


def test_transient_scene_change_recovers_without_learning_empty_region():
    m = monitor()
    changed = frame()
    changed[:, :30] = 220
    m.update(frame(), 0, [])
    assert m.update(changed, 1, []).state == "waiting_for_stable_view"
    for t in (2, 3, 4):
        assert not m.update(frame(False), t, []).changed
    assert m.update(frame(False), 5, []).changed


@pytest.mark.parametrize("timestamp,boxes", [(10, []), (-1, []), (1, None)])
def test_discontinuity_requires_explicit_revalidation(timestamp, boxes):
    m = monitor()
    m.update(frame(), 0, [])
    assert m.update(frame(False), timestamp, boxes).state == "revalidation_required"
    assert not m.update(frame(False), 11, []).changed


def test_global_change_does_not_create_removal():
    m = monitor()
    for t in range(3):
        result = m.update(255 - frame(), t, [])
        assert result.state == "waiting_for_stable_view"
        assert not result.changed
    assert m.update(255 - frame(), 3, []).state == "revalidation_required"


def test_reject_occluded_and_untextured_registration():
    m = RegisteredObjectMonitor()
    with pytest.raises(ValueError, match="occluded"):
        m.register([frame()] * 3, REGION, [[(0, 0, 100, 100)]] * 3)
    with pytest.raises(ValueError, match="edge detail"):
        m.register([frame(False)] * 3, REGION, [[], [], []])


def test_brief_occlusion_restarts_visible_confirmation():
    m = monitor()
    m.update(frame(False), 0, [])
    m.update(frame(False), 1, [])
    m.update(frame(False), 2, [(0, 0, 100, 100)])
    for t in (3, 4, 5):
        assert not m.update(frame(False), t, []).changed
    assert m.update(frame(False), 6, []).changed


def test_reference_roundtrip_and_camera_binding(tmp_path):
    path = tmp_path / "object.npz"
    monitor().save_reference(path, camera_id="cam1", source_fingerprint="source-a", name="Laptop")
    loaded = RegisteredObjectMonitor()
    with pytest.raises(ValueError, match="approval"):
        loaded.load_reference(path, camera_id="cam1", source_fingerprint="source-a")
    with pytest.raises(ValueError, match="camera source"):
        loaded.load_reference(path, camera_id="cam2", source_fingerprint="source-a", approved=True)
    assert loaded.load_reference(path, camera_id="cam1", source_fingerprint="source-a", approved=True) == "Laptop"
    assert loaded.update(frame(), 0, []).state == "present"


def test_transition_carries_frozen_before_after_evidence_once():
    m = monitor()
    current = frame(False)
    for t in range(3):
        assert m.update(current, t, []).evidence_png is None
    result = m.update(current, 3, [])
    before_mutation = result.evidence_png
    current[:] = 0
    panel = cv2.imdecode(np.frombuffer(before_mutation, np.uint8), cv2.IMREAD_COLOR)
    assert panel.shape == (148, 200, 3)
    assert not np.array_equal(panel[48:, :100], panel[48:, 100:])
    assert m.update(frame(False), 4, []).evidence_png is None


def test_long_occlusion_requires_registration():
    m = monitor()
    for t in range(31):
        assert m.update(frame(), t, [(0, 0, 100, 100)]).state == "occluded"
    assert m.update(frame(), 31, [(0, 0, 100, 100)]).state == "revalidation_required"
