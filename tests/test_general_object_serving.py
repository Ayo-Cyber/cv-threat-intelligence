from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import supervision as sv


def _site(tmp_path, **camera_values):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"use_case_id": "objects", "rules": []}))
    return {"cameras": [{
        "id": "cam1", "source": "clip.mp4", "config": str(rules),
        **camera_values,
    }]}


def test_general_tracking_is_opt_in_and_independent_of_object_watch(tmp_path):
    from cvti.serving.camera import build_camera_states

    disabled = build_camera_states(_site(tmp_path))["cam1"]["state"]
    assert disabled.general_object_tracking is False
    assert disabled.general_object_snapshot(1.0) is None

    enabled = build_camera_states(_site(
        tmp_path, general_object_tracking=True,
        general_object_tracking_overlays=True, object_watch=False,
    ))["cam1"]["state"]
    enabled.ensure_general_object_tracker({0: "person", 2: "car"}, 7.0)
    tracker = enabled._general_object_tracker
    enabled.ensure_general_object_tracker({0: "person", 2: "car"}, 99.0)
    assert enabled._general_object_tracker is tracker
    assert tracker.expected_fps == 7.0
    assert enabled.object_watch is False


def test_route_feeds_unfiltered_shared_detections_before_person_processing():
    from cvti.serving.pipeline import MultiStreamPipeline
    from cvti.serving.streams import Frame

    shared = sv.Detections(
        xyxy=np.asarray([[0, 0, 10, 20], [20, 0, 40, 20]], dtype=float),
        confidence=np.asarray([0.9, 0.8]), class_id=np.asarray([0, 2]),
    )
    calls = []

    class State:
        camera_id = "cam1"
        _health = SimpleNamespace(failed=lambda *_a, **_k: None)

        def ensure_general_object_tracker(self, names, fps):
            calls.append(("ensure", names, fps))

        def update_general_object_tracks(self, detections, timestamp):
            calls.append(("objects", detections, timestamp))

        def process(self, detections, image, timestamp, object_detections=None):
            calls.append(("people", detections, timestamp))
            return []

    pipe = MultiStreamPipeline(
        {"cam1": "clip"}, camera_states={"cam1": State()},
        alert_queue=SimpleNamespace(add=lambda _alert: True), target_fps=5,
    )
    pipe._names = {0: "person", 2: "car"}
    pipe._threat_classes = set()
    result = SimpleNamespace(boxes=[])
    frame = Frame("cam1", np.zeros((8, 8, 3), np.uint8), 1, 0.25)
    with mock.patch("supervision.Detections.from_ultralytics", return_value=shared), \
            mock.patch("cvti.detector.core.extract_detections", return_value=[]):
        pipe._route_to_queue(frame, result)

    assert calls[1][0] == "objects" and calls[1][1] is shared
    assert calls[2][0] == "people" and calls[2][1] is shared


def test_optional_tracker_failure_never_blocks_person_safety():
    from cvti.serving.pipeline import MultiStreamPipeline
    from cvti.serving.streams import Frame

    processed = []

    class State:
        camera_id = "cam1"
        _health = SimpleNamespace(failed=lambda *_a, **_k: None)
        ensure_general_object_tracker = lambda *_a: None

        def update_general_object_tracks(self, *_args):
            raise RuntimeError("object lane failed")

        def process(self, *_args, **_kwargs):
            processed.append(True)
            return []

    pipe = MultiStreamPipeline(
        {"cam1": "clip"}, camera_states={"cam1": State()},
        alert_queue=SimpleNamespace(add=lambda _alert: True), target_fps=5,
    )
    pipe._names = {0: "person", 2: "car"}
    pipe._threat_classes = set()
    frame = Frame("cam1", np.zeros((8, 8, 3), np.uint8), 1, 0.25)
    with mock.patch("supervision.Detections.from_ultralytics", return_value=sv.Detections.empty()), \
            mock.patch("cvti.detector.core.extract_detections", return_value=[]):
        pipe._route_to_queue(frame, SimpleNamespace(boxes=[]))
    assert processed == [True]


def test_object_namespace_cannot_inherit_person_alert_colour():
    from cvti.serving.frame_publisher import FrameOverlay, FramePublisher

    pub = FramePublisher(max_width=0)
    pub._viewer_started("cam", tracking=True)
    frame = np.full((80, 120, 3), 60, np.uint8)
    overlay = FrameOverlay(-1, (10, 10, 60, 60), "car O1", (255, 180, 0), "object")
    pub.publish("cam", frame, [overlay])
    normal = pub.frame("cam", tracking=True)
    pub.mark_alerting("cam", {-1})
    pub.publish("cam", frame, [overlay])
    assert pub.frame("cam", tracking=True) == normal
    assert pub.snapshot()["tracks"]["cam"] == []


def test_publisher_detaches_object_snapshots_and_clears_on_disabled_publish():
    from cvti.serving.frame_publisher import FramePublisher

    pub = FramePublisher(max_width=0)
    frame = np.zeros((8, 8, 3), np.uint8)
    payload = {"status": "ok", "tracks": [{"id": "cam/session/1"}]}
    pub.publish("cam", frame, object_tracks=payload)
    payload["tracks"][0]["id"] = "mutated"
    first = pub.snapshot()
    assert first["object_tracks"]["cam"]["tracks"][0]["id"] == "cam/session/1"
    first["object_tracks"]["cam"]["tracks"].clear()
    assert pub.snapshot()["object_tracks"]["cam"]["tracks"]

    pub.publish("cam", frame)
    assert pub.snapshot()["object_tracks"]["cam"] is None


def test_cached_metadata_and_object_pixels_age_without_another_publish():
    from cvti.serving.frame_publisher import FrameOverlay, FramePublisher

    pub = FramePublisher(max_width=0)
    pub._viewer_started("cam", tracking=True)
    frame = np.full((80, 120, 3), 60, np.uint8)
    person = FrameOverlay(7, (5, 5, 35, 65), "#7", (0, 200, 0))
    obj = FrameOverlay(-1, (60, 5, 110, 65), "car O1", (255, 180, 0), "object")
    snapshot = {
        "timestamp_clock": "monotonic", "timestamp": 100.4,
        "last_success_at": 100.0, "status": "ok",
        "freshness": {
            "overlay_max_age_seconds": 0.5,
            "lost_after_seconds": 2.0,
            "ended_retention_seconds": 5.0,
        },
        "tracks": [{
            "id": "cam/session/1", "state": "observed", "last_seen": 100.0,
            "ended_at": None, "end_reason": None,
        }],
    }
    with mock.patch("cvti.serving.frame_publisher.time.monotonic", return_value=100.4):
        pub.publish("cam", frame, [person, obj], object_tracks=snapshot)
        combined = pub.frame("cam", tracking=True)

    with mock.patch("cvti.serving.frame_publisher.time.monotonic", return_value=100.51):
        person_only = pub.frame("cam", tracking=True)
        assert person_only != combined
        assert pub.snapshot()["object_tracks"]["cam"]["tracks"][0]["state"] == "observed"

    with mock.patch("cvti.serving.frame_publisher.time.monotonic", return_value=102.01):
        aged = pub.snapshot()["object_tracks"]["cam"]
    assert aged["status"] == "stale"
    assert aged["tracks"][0]["state"] == "ended"
    assert pub.frame("cam") is not None, "raw glass must remain untouched"


def test_disconnect_fences_inflight_result_and_invalidates_without_new_publish():
    from cvti.serving.pipeline import MultiStreamPipeline
    from cvti.serving.streams import Frame

    updates = []
    processed = []
    invalidated = []

    class State:
        camera_id = "cam1"
        _health = SimpleNamespace(failed=lambda *_a, **_k: None)
        ensure_general_object_tracker = lambda *_a: None

        def update_general_object_tracks(self, detections, timestamp):
            updates.append((detections, timestamp))

        def reset_general_object_tracks(self, timestamp, reason):
            updates.append(("reset", timestamp, reason))

        def general_object_snapshot(self, timestamp):
            return {"status": "starting", "timestamp": timestamp,
                    "timestamp_clock": "monotonic", "tracks": []}

        def process(self, *_args, **_kwargs):
            processed.append(True)
            return []

    publisher = SimpleNamespace(
        invalidate_object_tracking=lambda camera_id, snapshot:
            invalidated.append((camera_id, snapshot)),
        publish=lambda *_args, **_kwargs: None,
    )
    state = State()
    pipe = MultiStreamPipeline(
        {"cam1": "clip"}, camera_states={"cam1": state},
        alert_queue=SimpleNamespace(add=lambda _alert: True), publisher=publisher,
        publish_fps=0, target_fps=5,
    )
    pipe._names = {0: "person", 2: "car"}
    pipe._threat_classes = set()
    frame = Frame("cam1", np.zeros((8, 8, 3), np.uint8), 1, 0.25)
    pipe._inference_context[id(frame)] = (0, 40.0)

    with mock.patch("cvti.serving.pipeline.time.monotonic", return_value=41.0):
        pipe._handle_link_change("cam1", "connected", "reconnecting", 1.0)
    assert invalidated[0][0] == "cam1"
    assert invalidated[0][1]["tracks"] == []

    shared = sv.Detections.empty()
    with mock.patch("supervision.Detections.from_ultralytics", return_value=shared), \
            mock.patch("cvti.detector.core.extract_detections", return_value=[]), \
            mock.patch("cvti.serving.pipeline.time.time", return_value=999999.0):
        pipe._route_to_queue(frame, SimpleNamespace(boxes=[]))

    assert updates == [("reset", 41.0, "source_reset")]
    assert processed == [True], "generation fencing must not suppress person safety"


def test_observation_time_is_captured_before_delayed_inference():
    from cvti.serving.pipeline import MultiStreamPipeline
    from cvti.serving.streams import Frame

    observed = []
    state = SimpleNamespace(
        camera_id="cam1",
        _health=SimpleNamespace(failed=lambda *_a, **_k: None),
        ensure_general_object_tracker=lambda *_a: None,
        update_general_object_tracks=lambda _detections, timestamp: observed.append(timestamp),
        process=lambda *_a, **_k: [],
    )
    pipe = MultiStreamPipeline(
        {"cam1": "clip"}, camera_states={"cam1": state},
        alert_queue=SimpleNamespace(add=lambda _alert: True), target_fps=5,
    )
    pipe._names = {0: "person", 2: "car"}
    pipe._threat_classes = set()
    frame = Frame("cam1", np.zeros((8, 8, 3), np.uint8), 1, 123.0)
    pipe._inference_context[id(frame)] = (0, 50.0)
    with mock.patch("supervision.Detections.from_ultralytics", return_value=sv.Detections.empty()), \
            mock.patch("cvti.detector.core.extract_detections", return_value=[]), \
            mock.patch("cvti.serving.pipeline.time.monotonic", return_value=500.0), \
            mock.patch("cvti.serving.pipeline.time.time", return_value=-1000.0):
        pipe._route_to_queue(frame, SimpleNamespace(boxes=[]))
    assert observed == [50.0]


def test_source_reset_rejects_an_inflight_old_object_publication():
    import threading
    import time

    from cvti.serving.frame_publisher import FrameOverlay, FramePublisher

    pub = FramePublisher(max_width=0)
    pub._viewer_started("cam", tracking=True)
    started = threading.Event()
    release = threading.Event()
    real_encode = pub._encode_tracking

    def blocked_encode(*args):
        started.set()
        assert release.wait(2)
        return real_encode(*args)

    pub._encode_tracking = blocked_encode
    frame = np.full((80, 120, 3), 60, np.uint8)
    overlay = FrameOverlay(-1, (10, 10, 60, 60), "car O1", (255, 180, 0), "object")
    now = time.monotonic()
    old = {"timestamp_clock": "monotonic", "timestamp": now,
            "freshness": {"overlay_max_age_seconds": 0.5},
           "tracks": [{"id": "cam/session/1", "state": "observed",
                       "last_seen": now}]}
    worker = threading.Thread(
        target=pub.publish,
        args=("cam", frame, [overlay]),
        kwargs={"object_tracks": old},
    )
    worker.start()
    assert started.wait(2)
    reset = {"timestamp_clock": "monotonic", "timestamp": 11.0,
             "status": "starting", "tracks": []}
    pub.invalidate_object_tracking("cam", reset)
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert pub.snapshot()["object_tracks"]["cam"]["tracks"] == []
    assert pub.frame("cam", tracking=True) == pub.frame("cam")


def test_reset_between_snapshot_collection_and_publish_entry_is_fenced():
    import cv2
    import time

    from cvti.serving.frame_publisher import FramePublisher
    from cvti.serving.pipeline import _publish_frame, _publish_jpeg

    frame = np.full((80, 120, 3), 60, np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok

    for jpeg_path in (False, True):
        pub = FramePublisher(max_width=0)
        pub._viewer_started("cam", tracking=True)
        reset = {"timestamp_clock": "monotonic", "timestamp": 11.0,
                 "status": "starting", "tracks": []}
        now = time.monotonic()
        old = {"timestamp_clock": "monotonic", "timestamp": now,
               "freshness": {"overlay_max_age_seconds": 0.5},
               "tracks": [{"id": "cam/session/1", "state": "observed",
                            "last_seen": now}]}

        class State:
            _motion_overlays = []

            def general_object_overlays(self, _timestamp):
                return [{"track_id": -1, "bbox": (10, 10, 60, 60),
                         "label": "car O1", "colour": (255, 180, 0),
                         "namespace": "object"}]

            def general_object_snapshot(self, _timestamp):
                # The generation token has already been captured by the helper.
                pub.invalidate_object_tracking("cam", reset)
                return old

        if jpeg_path:
            _publish_jpeg(pub, "cam", encoded.tobytes(), State(), frame.shape[:2])
        else:
            _publish_frame(pub, "cam", frame, State())

        assert pub.snapshot()["object_tracks"]["cam"]["tracks"] == []
        assert pub.frame("cam", tracking=True) == pub.frame("cam")


def test_person_only_tracking_encodes_once_per_publish_path():
    import cv2

    from cvti.serving.frame_publisher import FrameOverlay, FramePublisher

    pub = FramePublisher(max_width=0)
    pub._viewer_started("cam", tracking=True)
    frame = np.full((80, 120, 3), 60, np.uint8)
    person = FrameOverlay(7, (5, 5, 35, 65), "#7", (0, 200, 0))
    calls = []
    real_encode = pub._encode_tracking

    def counting_encode(*args):
        calls.append(True)
        return real_encode(*args)

    pub._encode_tracking = counting_encode
    pub.publish("cam", frame, [person])
    assert len(calls) == 1

    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    pub.publish_jpeg("cam", encoded.tobytes(), [person], frame.shape[:2])
    assert len(calls) == 2


def test_overlay_snapshot_race_fails_closed_for_both_publish_paths():
    import cv2

    from cvti.serving.frame_publisher import FramePublisher
    from cvti.serving.pipeline import _publish_frame, _publish_jpeg

    frame = np.full((80, 120, 3), 60, np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok

    class State:
        _motion_overlays = []

        def general_object_overlays(self, _timestamp):
            return [{"track_id": -1, "bbox": (10, 10, 60, 60),
                     "label": "car O1", "colour": (255, 180, 0),
                     "namespace": "object"}]

        def general_object_snapshot(self, timestamp):
            # Successful empty update landed between the two state reads.
            return {"timestamp_clock": "monotonic", "timestamp": timestamp,
                    "freshness": {"overlay_max_age_seconds": 0.5},
                    "tracks": [{"id": "cam/session/1", "state": "lost",
                                "last_seen": timestamp - 0.1}]}

    for jpeg_path in (False, True):
        pub = FramePublisher(max_width=0)
        pub._viewer_started("cam", tracking=True)
        if jpeg_path:
            _publish_jpeg(pub, "cam", encoded.tobytes(), State(), frame.shape[:2])
        else:
            _publish_frame(pub, "cam", frame, State())
        with mock.patch("cvti.serving.frame_publisher.time.monotonic",
                        return_value=10_000_000.0):
            assert pub.frame("cam", tracking=True) == pub.frame("cam")


def test_object_overlays_require_matching_finite_freshness_evidence():
    import time

    from cvti.serving.frame_publisher import FrameOverlay, FramePublisher

    now = time.monotonic()
    overlay = FrameOverlay(-1, (10, 10, 60, 60), "car O1", (255, 180, 0), "object")
    base = {"timestamp_clock": "monotonic", "timestamp": now,
            "freshness": {"overlay_max_age_seconds": 0.5}}
    invalid = [
        None,
        {**base, "tracks": []},
        {**base, "tracks": [{"id": "cam/session/2", "state": "observed",
                              "last_seen": now}]},
        {**base, "tracks": [{"id": "cam/session/1", "state": "lost",
                              "last_seen": now}]},
        {**base, "tracks": [{"id": "cam/session/1", "state": "observed",
                              "last_seen": float("nan")}]},
        {"timestamp_clock": "monotonic",
         "freshness": {"overlay_max_age_seconds": 0.5},
         "tracks": [{"id": "cam/session/1", "state": "observed",
                     "last_seen": now}]},
        {**base, "freshness": {"overlay_max_age_seconds": float("inf")},
         "tracks": [{"id": "cam/session/1", "state": "observed",
                     "last_seen": now}]},
    ]
    frame = np.full((80, 120, 3), 60, np.uint8)
    for snapshot in invalid:
        pub = FramePublisher(max_width=0)
        pub._viewer_started("cam", tracking=True)
        pub.publish("cam", frame, [overlay], object_tracks=snapshot)
        assert pub.frame("cam", tracking=True) == pub.frame("cam")
        assert "cam" not in pub._object_overlay_expires_at


def test_draw_boxes_false_never_encodes_supplied_object_overlays():
    from cvti.serving.frame_publisher import FrameOverlay, FramePublisher

    pub = FramePublisher(max_width=0, draw_boxes=False)
    pub._viewer_started("cam", tracking=True)
    frame = np.full((80, 120, 3), 60, np.uint8)
    overlay = FrameOverlay(-1, (10, 10, 60, 60), "car O1", (255, 180, 0), "object")
    snapshot = {"timestamp_clock": "monotonic", "timestamp": 100.0,
                "freshness": {"overlay_max_age_seconds": 0.5},
                "tracks": [{"id": "cam/session/1", "state": "observed",
                            "last_seen": 100.0}]}
    pub._encode_tracking = mock.Mock(side_effect=AssertionError("must not encode"))
    with mock.patch("cvti.serving.frame_publisher.time.monotonic", return_value=100.0):
        pub.publish("cam", frame, [overlay], object_tracks=snapshot)
    assert pub._encode_tracking.call_count == 0
    assert pub.frame("cam", tracking=True) == pub.frame("cam")
