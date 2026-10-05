import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cvti.serving.registered_objects import RegisteredObjects, source_key


def frame(present=True):
    image = np.random.default_rng(2).integers(70, 90, (100, 100, 3), dtype=np.uint8)
    image[30:70, 30:70] = 80
    if present:
        image[34:66, 34:66] = 220
        image[44:56, 44:56] = 20
    return image


def entry(tmp_path):
    return {"id": "one", "name": "Laptop", "region": [30,30,70,70],
            "frame_hw": [100,100], "confirm_seconds": 2,
            "reference_path": str(tmp_path / "reference.npz"), "source_fingerprint": source_key("video")}


def test_capture_change_evidence_and_restart_revalidation(tmp_path):
    e = entry(tmp_path)
    runtime = RegisteredObjects("cam", "video", [e])
    for t in (0, .5, 1):
        assert runtime.update(frame(), [], t) == []
    assert Path(e["reference_path"]).exists()
    for t in (2, 3):
        assert runtime.update(frame(False), [], t) == []
    events = runtime.update(frame(False), [], 4)
    assert len(events) == 1
    candidate, panel = events[0]
    assert candidate.metadata["object_id"] == "one"
    assert panel.shape == (148,200,3)
    assert runtime.update(frame(False), [], 5) == []
    restarted = RegisteredObjects("cam", "video", [e])
    assert restarted.update(frame(False), [], 0) == []
    assert json.loads(Path(e["reference_path"]).with_suffix(".status.json").read_text())["state"] == "revalidation_required"


def test_capture_waits_for_person_and_valid_inference(tmp_path):
    e = entry(tmp_path)
    runtime = RegisteredObjects("cam", "video", [e])
    person = SimpleNamespace(label="person", bbox=(0,0,100,100))
    for t in range(5):
        runtime.update(frame(), [person], t)
    runtime.update(frame(), None, 5)
    assert not Path(e["reference_path"]).exists()


def test_source_mismatch_never_captures(tmp_path):
    e = entry(tmp_path)
    runtime = RegisteredObjects("cam", "other", [e])
    for t in range(5):
        assert runtime.update(frame(), [], t) == []
    assert not Path(e["reference_path"]).exists()


def test_serving_produces_immutable_evidence_and_stale_guard(tmp_path):
    import supervision as sv
    from cvti.serving.camera import build_camera_states
    from cvti.serving.registered_objects import candidate_current
    e = entry(tmp_path)
    camera = {"id":"cam", "source":"video", "config":"configs/chi_object_state_v1.json", "registered_objects":[e]}
    state = build_camera_states({"cameras":[camera]}, output_dir=tmp_path)["cam"]["state"]
    for t in (0, .5, 1):
        assert not state.process(sv.Detections.empty(), frame(), t, [])
    for t in (2, 3):
        assert not state.process(sv.Detections.empty(), frame(False), t, [])
    events = state.process(sv.Detections.empty(), frame(False), 4, [])
    assert len(events) == 1
    alert = events[0]
    assert alert.zone == "one"
    assert not alert.payload["frames"][0].flags.writeable
    candidate = alert.payload["candidate"]
    assert candidate_current(candidate, {"cameras":[camera]})
    assert not candidate_current(candidate, {"cameras":[{**camera,"registered_objects":[]}]})
    assert not candidate_current(candidate, {"cameras":[{**camera,"source":"changed"}]})
    assert not candidate_current(candidate, {"cameras":[{**camera,"registered_objects":[{**e,"reference_path":"new.npz"}]}]})
    # Plumbing only: deterministic verifier response, not an accuracy claim.
    from unittest.mock import patch
    from cvti.verification.gate import VerificationGate
    from cvti.serving.alert_sink import AlertSink
    import sqlite3
    gate = VerificationGate(provider="ollama")
    with patch.object(gate, "_call_provider", return_value=json.dumps({
        "confirmed": True, "confidence": .9, "reason": "The marked object differs from its reference.", "alert_priority": "medium"
    })):
        verdict = gate.verify(alert.payload["frames"], candidate)
    assert verdict.confirmed
    sink = AlertSink(str(tmp_path / "incidents"), save_evidence=True, routing_path=None)
    try:
        event_id = sink.handle(alert, verdict)
        assert event_id
        with sqlite3.connect(tmp_path / "incidents" / "events.db") as db:
            rule, evidence = db.execute("SELECT rule,evidence_dir FROM events WHERE id=?", (event_id,)).fetchone()
        assert rule == "registered_object_change"
        assert Path(evidence).is_dir()
    finally:
        sink.close()
