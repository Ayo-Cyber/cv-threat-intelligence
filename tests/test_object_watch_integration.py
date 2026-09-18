from __future__ import annotations

import io
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from PIL import Image


def _png(colour=(20, 80, 180)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 24), colour).save(output, "PNG")
    return output.getvalue()


def test_reviewed_enrollment_reaches_rules_and_gate_with_reference_panel(tmp_path):
    """Plumbing test: injected hash semantics, not an embedding-accuracy test."""
    from cvti.object_watch.embeddings import HashEmbeddingBackend, embed_examples
    from cvti.object_watch.matcher import ObjectCandidate
    from cvti.object_watch.runtime import ObjectWatchRuntime, WatchSample
    from cvti.object_watch.runtime_config import ObjectWatchConfig
    from cvti.object_watch.store import (
        ObjectTarget, activate_target, add_example, review_example, save_target,
    )
    from cvti.rules.customization import CustomizationEngine
    from cvti.serving.alert_queue import AlertQueue
    from cvti.serving.camera import _to_queued
    from cvti.serving.event_adapters import object_watch_result_events
    from cvti.serving.gate_pool import GatePool

    backend = HashEmbeddingBackend(16)
    target = ObjectTarget("blue-box", "Blue box", "product", min_similarity=0.0)
    save_target(tmp_path, target)
    example = add_example(tmp_path, target.id, _png(), (0, 0, 1, 1), "upload")
    review_example(tmp_path, target.id, example.id)
    embed_examples(tmp_path, backend)
    activate_target(tmp_path, target.id, backend)

    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"rules": [{
        "name": "watch_blue_box", "priority": "medium",
        "trigger": {"detector": "object_watch", "state": "object_seen",
                    "object_id": "blue-box", "zone": "door"},
    }]}))
    engine = CustomizationEngine(rules)
    runtime = ObjectWatchRuntime(
        ObjectWatchConfig(library_path=tmp_path / "object_library", sample_fps=20),
        backend_factory=lambda _config: backend,
    )
    runtime.start()
    frame = np.asarray(Image.open(io.BytesIO(_png())))[:, :, ::-1].copy()
    candidate = ObjectCandidate((0, 0, 24, 24), "box", 0.9, 7, "door")
    results = []
    deadline = time.monotonic() + 2
    while runtime.status()["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.01)
    for sequence in (1, 2):
        processed = runtime.status()["processed"]
        assert runtime.submit(WatchSample("cam", 0, sequence, time.monotonic(),
                                         float(sequence), frame, (candidate,)))
        while runtime.status()["processed"] == processed and time.monotonic() < deadline:
            time.sleep(0.01)
        results.extend(runtime.drain("cam", 0, time.monotonic()))
    runtime.stop()
    matched = next(row for row in results if row.status == "matched")
    events = object_watch_result_events(matched)
    alerts = engine.evaluate(events)
    assert len(alerts) == 1
    assert len(matched.evidence) == 2
    assert matched.evidence[-1].shape[1] >= frame.shape[1] * 2

    queued = _to_queued("cam", alerts[0], matched.event_timestamp, "door",
                        list(matched.evidence), None)
    token = runtime.reserve(matched, alerts[0].rule_name)
    assert token is not None
    seen = []

    class Gate:
        def verify(self, frames, candidate, scene, examples=None):
            from cvti.contracts import VerificationResult
            assert len(frames) == 2
            assert frames[-1].shape[1] >= frame.shape[1] * 2
            return VerificationResult(True, 0.9, "candidate matches reference",
                                      candidate.priority, str(time.time()))

    queue = AlertQueue(cooldown_seconds=0)
    pool = GatePool(queue, gate_factory=Gate,
                    on_verdict=lambda alert, verdict: seen.append((alert, verdict))).start()
    assert queue.add(queued)
    assert runtime.commit(token)
    assert runtime.reserve(matched, alerts[0].rule_name) is None
    deadline = time.monotonic() + 2
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    pool.stop()
    assert seen and seen[0][1].confirmed


def test_queue_rejection_retries_and_two_rules_reserve_independently(tmp_path):
    from cvti.object_watch.matcher import ObjectMatch
    from cvti.object_watch.runtime import ObjectWatchRuntime, WatchResult
    from cvti.object_watch.runtime_config import ObjectWatchConfig
    from cvti.serving.alert_queue import AlertQueue, QueuedAlert

    runtime = ObjectWatchRuntime(ObjectWatchConfig(library_path=tmp_path / "object_library"),
                                 backend_factory=lambda _config: object())
    runtime.reset_camera("cam", 0)
    runtime._snapshot_signature = (None,)
    runtime._snapshot_revision = 4
    runtime._snapshot_active = {"box": 2}
    match = ObjectMatch("cam", "box", "Box", "product", (0, 0, 4, 4),
                        0.9, 1.0, track_id=7, target_revision=2,
                        library_revision=4)
    result = WatchResult("cam", 0, 2, 2.0, 1.0, 4, 2, "fake", (match,),
                         (np.zeros((2, 2, 3)), np.ones((2, 4, 3))), "matched",
                         _config_signature=(None,))

    rejected = runtime.reserve(result, "rule-a")
    assert rejected is not None, "reservation exists before queue admission"
    full = AlertQueue(max_pending=0)
    alert = QueuedAlert("cam", "rule-a", "medium", "BOX", 1.0)
    assert not full.add(alert)
    # The rejected candidate was not committed and remains eligible.
    retry = runtime.reserve(result, "rule-a")
    sibling = runtime.reserve(result, "rule-b")
    assert retry is not None and sibling is not None
    assert runtime.commit(retry)
    assert runtime.commit(sibling)
    assert runtime.reserve(result, "rule-a") is None
    assert runtime.reserve(result, "rule-b") is None


def test_each_match_result_keeps_its_own_evidence():
    from cvti.object_watch.matcher import ObjectMatch
    from cvti.object_watch.runtime import WatchResult
    from cvti.serving.event_adapters import object_watch_result_events

    red = np.full((4, 8, 3), (0, 0, 255), np.uint8)
    blue = np.full((4, 8, 3), (255, 0, 0), np.uint8)
    rows = []
    for object_id, evidence in (("red-box", red), ("blue-box", blue)):
        match = ObjectMatch("cam", object_id, object_id, "product", (0, 0, 4, 4),
                            0.9, 1.0, target_revision=1, library_revision=3)
        rows.append(WatchResult("cam", 0, 2, 2.0, 1.0, 3, 1, "fake",
                                (match,), (np.zeros((4, 4, 3)), evidence), "matched"))
    events = [object_watch_result_events(row)[0] for row in rows]
    assert [event.extra["object_id"] for event in events] == ["red-box", "blue-box"]
    assert np.array_equal(rows[0].evidence[-1], red)
    assert np.array_equal(rows[1].evidence[-1], blue)
    assert not np.array_equal(rows[0].evidence[-1], rows[1].evidence[-1])


def test_wrong_zone_fails_closed(tmp_path):
    from cvti.contracts import RawEvent
    from cvti.rules.customization import CustomizationEngine

    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"rules": [{
        "name": "watch", "trigger": {"detector": "object_watch",
        "state": "object_seen", "object_id": "box", "zone": "door"},
    }]}))
    engine = CustomizationEngine(rules)
    event = RawEvent("object_watch", True, "BOX SEEN", "medium",
                     state="object_seen", extra={"object_id": "box", "zone": "yard"})
    assert engine.evaluate([event]) == []


def test_camera_zone_snapshot_is_a_fitted_immutable_copy():
    from cvti.serving.camera import PerCameraState

    polygon = np.asarray(((0, 0), (20, 0), (20, 20), (0, 20)), dtype=np.int32)

    class Monitor:
        zones = (SimpleNamespace(name="door", polygon=polygon),)
        fitted = None

        def _fit_to_frame(self, frame_hw):
            self.fitted = frame_hw

    state = PerCameraState.__new__(PerCameraState)
    state.camera_id = "cam"
    state.zone_monitor = Monitor()
    snapshot = state._object_watch_zone_snapshot((100, 200))
    polygon[:] = 99

    assert state.zone_monitor.fitted == (100, 200)
    assert snapshot[0].name == "door"
    assert snapshot[0].polygon == ((0, 0), (20, 0), (20, 20), (0, 20))


def test_queued_alert_guard_rejects_deactivation_and_source_reset(tmp_path, monkeypatch):
    from cvti.contracts import CandidateAlert
    from cvti.object_watch.embeddings import HashEmbeddingBackend, embed_examples
    from cvti.object_watch.store import (
        ObjectTarget, activate_target, add_example, library_revision,
        review_example, save_target,
    )
    from cvti.serving.alert_queue import QueuedAlert
    from cvti.serving.event_adapters import object_watch_alert_current

    backend = HashEmbeddingBackend(8)
    save_target(tmp_path, ObjectTarget("box", "Box", "product", min_similarity=0.0))
    example = add_example(tmp_path, "box", _png(), (0, 0, 1, 1), "upload")
    review_example(tmp_path, "box", example.id)
    embed_examples(tmp_path, backend)
    active = activate_target(tmp_path, "box", backend)
    library = tmp_path / "object_library"
    candidate = CandidateAlert(
        "watch", "medium", "object_watch", "BOX SEEN", None, "Box", 1.0,
        metadata={"object_id": "box", "library_revision": library_revision(library),
                  "target_revision": active.revision, "source_generation": 0,
                  "runtime_config_stamp": ()},
    )
    alert = QueuedAlert("cam", "watch", "medium", "BOX SEEN", 1.0,
                        payload={"candidate": candidate, "frames": []})
    rule = {"name": "watch", "trigger": {"detector": "object_watch",
            "state": "object_seen", "object_id": "box"}}
    from cvti.serving.event_adapters import object_watch_rule_signature
    candidate.metadata["object_watch_rule_signature"] = object_watch_rule_signature(rule)
    state = SimpleNamespace(object_watch=True, _object_watch_generation=0,
                             engine=SimpleNamespace(rules=[rule], baseline_rules=[]))
    assert object_watch_alert_current(alert, state, library)

    state.object_watch = False
    assert not object_watch_alert_current(alert, state, library)
    state.object_watch = True

    state.engine.rules = [{**rule, "priority": "critical"}]
    assert not object_watch_alert_current(alert, state, library)
    state.engine.rules = [rule]

    candidate.metadata["model_fingerprint"] = "old-model"
    monkeypatch.setattr("cvti.object_watch.runtime_config.resolve_config", lambda *_: object())
    monkeypatch.setattr("cvti.object_watch.runtime_config.configured_backend_metadata",
                        lambda *_: SimpleNamespace(fingerprint="new-model"))
    assert not object_watch_alert_current(alert, state, library)
    candidate.metadata.pop("model_fingerprint")

    (library / "runtime.json").write_text(json.dumps({"backend": "siglip"}))
    assert not object_watch_alert_current(alert, state, library)
    (library / "runtime.json").unlink()
    assert object_watch_alert_current(alert, state, library)
    state._object_watch_generation = 1
    assert not object_watch_alert_current(alert, state, library)
    state._object_watch_generation = 0
    save_target(tmp_path, replace(active, review_state="disabled"))
    assert not object_watch_alert_current(alert, state, library)


def test_pipeline_prequeue_guard_is_strictly_in_memory(monkeypatch):
    from cvti.contracts import CandidateAlert
    from cvti.serving.alert_queue import QueuedAlert
    from cvti.serving.event_adapters import object_watch_rule_signature
    from cvti.serving.pipeline import MultiStreamPipeline
    from cvti.serving.streams import Frame

    now = time.monotonic()
    rule = {"name": "watch", "trigger": {"detector": "object_watch",
            "state": "object_seen", "object_id": "box"}}
    candidate = CandidateAlert(
        "watch", "medium", "object_watch", "BOX", None, "Box", now,
        metadata={"object_id": "box", "source_generation": 0,
                  "object_watch_rule_signature": object_watch_rule_signature(rule)},
    )
    result = SimpleNamespace(observed_at_monotonic=now)
    alert = QueuedAlert("cam", "watch", "medium", "BOX", now,
                        payload={"candidate": candidate, "object_watch_result": result})
    admitted = []

    class Runtime:
        config = SimpleNamespace(result_ttl_seconds=200.0)
        def result_current(self, _result): return True
        def commit(self, _token): return True

    class State:
        camera_id = "cam"
        object_watch = True
        object_watch_library = "/must-not-read"
        _object_watch_generation = 0
        _object_watch_runtime = Runtime()
        engine = SimpleNamespace(rules=[rule], baseline_rules=[])
        _health = SimpleNamespace(failed=lambda *_a, **_k: None)
        def ensure_general_object_tracker(self, *_args): pass
        def update_general_object_tracks(self, *_args): pass
        def process(self, *_args, **_kwargs): return [alert]

    pipe = MultiStreamPipeline(
        {"cam": "clip"}, camera_states={"cam": State()},
        alert_queue=SimpleNamespace(add=lambda value: admitted.append(value) or True),
    )
    pipe._names = {}
    pipe._threat_classes = set()
    monkeypatch.setattr("supervision.Detections.from_ultralytics",
                        lambda *_: SimpleNamespace())
    monkeypatch.setattr("cvti.detector.core.extract_detections", lambda *_: [])
    monkeypatch.setattr("cvti.serving.event_adapters.object_watch_alert_current",
                        lambda *_: (_ for _ in ()).throw(AssertionError("disk guard")))
    monkeypatch.setattr("cvti.object_watch.store.library_revision",
                        lambda *_: (_ for _ in ()).throw(AssertionError("disk revision")))
    pipe._route_to_queue(
        Frame("cam", np.zeros((4, 4, 3), np.uint8), 1, now),
        SimpleNamespace(boxes=[]),
    )
    assert admitted == [alert]

    # A result that was already queued by the camera state must still fail the
    # final in-memory admission guard when live configuration disables runtime.
    State.object_watch = False
    pipe._route_to_queue(
        Frame("cam", np.zeros((4, 4, 3), np.uint8), 2, now),
        SimpleNamespace(boxes=[]),
    )
    assert admitted == [alert]
