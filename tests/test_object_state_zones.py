from types import SimpleNamespace

import numpy as np
import pytest

from cvti.detector.object_state import ObjectZoneMonitor, ObjectZonePolicy


ZONES = [SimpleNamespace(name="walkway", polygon=((10, 10), (65, 10), (65, 65), (10, 65)))]
BOX = (25, 25, 45, 45)


def detection(label="suitcase", bbox=BOX, confidence=0.9):
    return SimpleNamespace(label=label, bbox=bbox, confidence=confidence)


def frame(present=False):
    image = np.random.default_rng(5).integers(60, 100, (100, 100, 3), dtype=np.uint8)
    if present:
        image[25:45, 25:45] = 230
    return image


def monitor(mode="removed"):
    return ObjectZoneMonitor([dict(zone="walkway", labels=["suitcase"], mode=mode,
                                  stable_seconds=2, absent_seconds=3, dwell_seconds=3)])


def observe(m, start=0, end=2):
    for timestamp in range(start, end + 1):
        assert m.update(frame(True), [detection()], timestamp, ZONES) == []


def test_removed_needs_stable_presence_and_sustained_visible_change():
    m = monitor()
    observe(m)
    for timestamp in (3, 4, 5):
        assert not m.update(frame(), [], timestamp, ZONES)
    events = m.update(frame(), [], 6, ZONES)
    assert len(events) == 1
    event, panel = events[0]
    assert event.state == "object_removed"
    assert event.extra["zone"] == "walkway"
    assert event.extra["dwell_seconds"] == 3
    assert panel.shape == (508, 960, 3)
    assert not np.array_equal(panel[:, :480], panel[:, 480:])
    for timestamp in range(7, 20):
        assert not m.update(frame(), [], timestamp, ZONES)


def test_one_detection_is_not_a_stable_baseline():
    m = monitor()
    m.update(frame(True), [detection()], 0, ZONES)
    for timestamp in range(1, 10):
        assert not m.update(frame(), [], timestamp, ZONES)


def test_detector_misses_with_unchanged_pixels_are_not_removal():
    m = monitor()
    observe(m)
    for timestamp in range(3, 15):
        assert not m.update(frame(True), [], timestamp, ZONES)


def test_left_behind_dwell_restarts_after_missed_detection():
    m = monitor("left_behind")
    for t in range(3):
        assert not m.update(frame(), [], t, ZONES)
    observe(m, 3, 5)
    assert not m.update(frame(True), [], 6, ZONES)
    observe(m, 7, 9)
    assert m.update(frame(True), [detection()], 10, ZONES)[0][0].state == "object_left_behind"


@pytest.mark.parametrize("cause", ["gap", "rewind", "dark", "scene_change", "unavailable", "geometry"])
def test_discontinuities_reset_baseline(cause):
    m = monitor()
    observe(m)
    timestamp = 3
    if cause == "gap":
        timestamp = 50
        assert not m.update(frame(), [], timestamp, ZONES)
    elif cause == "rewind":
        assert not m.update(frame(), [], 0, ZONES)
    elif cause == "dark":
        assert not m.update(np.zeros((100, 100, 3), np.uint8), [], timestamp, ZONES)
    elif cause == "scene_change":
        assert not m.update(frame() + 60, [], timestamp, ZONES)
    elif cause == "unavailable":
        assert not m.update(frame(), None, timestamp, ZONES)
    else:
        assert not m.update(frame(), [], timestamp, [])
    for t in range(timestamp + 1, timestamp + 10):
        assert not m.update(frame(), [], t, ZONES)


def test_person_occlusion_does_not_count_as_absence():
    m = monitor()
    observe(m)
    for timestamp in range(3, 10):
        assert not m.update(frame(), [detection("person", (10, 10, 70, 90))], timestamp, ZONES)
    for timestamp in (10, 11, 12):
        assert not m.update(frame(), [], timestamp, ZONES)
    assert m.update(frame(), [], 13, ZONES)[0][0].state == "object_removed"


def test_overlapping_unknown_detection_blocks_removal():
    m = monitor()
    observe(m)
    for timestamp in range(3, 12):
        assert not m.update(frame(), [detection("box")], timestamp, ZONES)


def test_left_behind_requires_initial_empty_zone_then_new_stationary_object():
    m = monitor("left_behind")
    for timestamp in range(3):
        assert not m.update(frame(), [], timestamp, ZONES)
    observe(m, 3, 5)
    assert m.update(frame(True), [detection()], 6, ZONES)[0][0].state == "object_left_behind"
    for timestamp in range(7, 15):
        assert not m.update(frame(True), [detection()], timestamp, ZONES)


def test_human_can_place_an_object_after_clear_baseline():
    m = monitor("left_behind")
    for t in range(3):
        m.update(frame(), [], t, ZONES)
    m.update(frame(True), [detection("person"), detection()], 3, ZONES)
    observe(m, 4, 6)
    assert m.update(frame(True), [detection()], 7, ZONES)[0][0].state == "object_left_behind"


def test_startup_stock_is_not_left_behind():
    m = monitor("left_behind")
    observe(m, 0, 20)


def test_late_recognition_of_existing_object_is_not_arrival():
    m = monitor("left_behind")
    for t in range(3):
        assert not m.update(frame(True), [], t, ZONES)
    observe(m, 3, 20)


def test_multiple_targets_fail_closed():
    m = monitor()
    observe(m)
    assert not m.update(frame(True), [detection(), detection(bbox=(46, 25, 60, 45))], 3, ZONES)
    for t in range(4, 12):
        assert not m.update(frame(), [], t, ZONES)


@pytest.mark.parametrize("field,value", [("absent_seconds", 0), ("max_gap_seconds", float("nan")),
                                         ("min_observations", 1), ("labels", "suitcase"),
                                         ("labels", ["person"]), ("mode", "theft")])
def test_invalid_policy_rejected(field, value):
    policy = dict(zone="walkway", labels=["suitcase"], mode="removed")
    policy[field] = value
    with pytest.raises(ValueError):
        ObjectZonePolicy(**policy)


def test_rule_and_serving_queue_receive_before_after_evidence(tmp_path):
    import json
    import supervision as sv
    from cvti.serving.camera import build_camera_states, refresh_camera_rules

    zone_path = tmp_path / "zones.json"
    zone_path.write_text(json.dumps({"zones": [
        {"name": "storage_position", "kind": "storage", "polygon": ZONES[0].polygon}
    ]}))
    policy = dict(zone="storage_position", labels=["suitcase"], mode="removed",
                  stable_seconds=2, absent_seconds=3)
    camera = dict(id="test", source="unused.mp4", config="configs/chi_object_state_v1.json",
                  zones=str(zone_path), object_state_zones=[policy])
    state = build_camera_states({"cameras": [camera]}, output_dir=tmp_path)["test"]["state"]
    for t in range(3):
        assert not state.process(sv.Detections.empty(), frame(True), t, [detection()])
    for t in (3, 4, 5):
        assert not state.process(sv.Detections.empty(), frame(), t, [])
    queued = state.process(sv.Detections.empty(), frame(), 6, [])
    assert len(queued) == 1
    alert = queued[0]
    assert alert.rule_name == "chi_object_removed_from_position"
    assert alert.zone == "storage_position"
    candidate = alert.payload["candidate"]
    assert candidate.detector == "object_state"
    assert candidate.metadata["state"] == "object_removed"
    assert candidate.metadata["identity_scope"] == "generic_detector_class"
    assert len(alert.payload["frames"]) == 1
    assert not alert.payload["frames"][0].flags.writeable
    from cvti.verification.gate import _DETECTOR_QUESTIONS
    assert "Removal is not evidence of theft" in _DETECTOR_QUESTIONS[candidate.detector]
    refresh_camera_rules(state, {**camera, "object_state_zones": []})
    assert state.object_state_zones == []


def test_unknown_zone_is_configuration_error(tmp_path):
    from cvti.serving.camera import build_camera_states
    with pytest.raises(ValueError, match="missing zone"):
        build_camera_states({"cameras": [dict(
            id="test", source="unused.mp4", config="configs/chi_object_state_v1.json",
            object_state_zones=[dict(zone="unknown", labels=["suitcase"], mode="removed")]
        )]}, output_dir=tmp_path)


@pytest.mark.parametrize("reset", ["source", "inference"])
def test_serving_resets_object_state_on_source_or_inference_failure(reset):
    from cvti.serving.camera import PerCameraState
    from cvti.rules.customization import CustomizationEngine
    state = PerCameraState("cam", CustomizationEngine())
    state._object_state_monitor = monitor()
    observe(state._object_state_monitor)
    if reset == "source":
        state.reset_object_watch(1)
    else:
        state.update_general_object_tracks(None, 3)
    for t in range(4, 15):
        assert not state._object_state_monitor.update(frame(), [], t, ZONES)
