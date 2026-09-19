from __future__ import annotations

import json
from pathlib import Path


def test_camera_config_accepts_object_watch_flags(tmp_path):
    from cvti.serving.camera import build_camera_states

    cfg = tmp_path / "object_rules.json"
    cfg.write_text(json.dumps({"use_case_id": "object_test", "rules": []}))
    site = {
        "cameras": [{
            "id": "cam1",
            "source": "data/test_clips/normal_street_01.mp4",
            "config": str(cfg),
            "object_watch": True,
            "object_watch_library": str(tmp_path / "object_library"),
            "object_watch_sample_fps": 1.0,
            "object_watch_max_candidates_per_frame": 12,
            "object_watch_min_similarity": 0.74,
            "object_watch_open_vocab_provider": "disabled",
        }]
    }

    state = build_camera_states(site, output_dir=tmp_path)["cam1"]["state"]

    assert state.object_watch is True
    assert state.object_watch_library == str(tmp_path / "object_library")
    assert state.object_watch_sample_fps == 1.0
    assert state.object_watch_max_candidates_per_frame == 12
    assert state.object_watch_min_similarity == 0.74
    assert state.object_watch_open_vocab_provider == "disabled"
    assert state._object_matcher is None
    assert state._object_state_tracker is None


def test_object_watch_flags_reject_invalid_types(tmp_path):
    from cvti.serving.camera import build_camera_states

    cfg = tmp_path / "object_rules.json"
    cfg.write_text(json.dumps({"use_case_id": "object_test", "rules": []}))
    base = {
        "id": "cam1",
        "source": "clip.mp4",
        "config": str(cfg),
    }

    invalid = (
        {"object_watch": "true"},
        {"object_watch_sample_fps": True},
        {"object_watch_sample_fps": 0.0},
        {"object_watch_max_candidates_per_frame": True},
        {"object_watch_max_candidates_per_frame": 0},
        {"object_watch_min_similarity": True},
        {"object_watch_min_similarity": 1.2},
    )
    for values in invalid:
        site = {"cameras": [{**base, **values}]}
        try:
            build_camera_states(site, output_dir=tmp_path)
        except ValueError as exc:
            assert "cam1" in str(exc)
        else:
            raise AssertionError(f"invalid object watch config was accepted: {values}")


def test_object_watch_rejects_noncanonical_camera_library(tmp_path):
    from cvti.serving.camera import build_camera_states

    cfg = tmp_path / "rules.json"
    cfg.write_text(json.dumps({"rules": []}))
    site = {"cameras": [{
        "id": "cam1", "source": "clip.mp4", "config": str(cfg),
        "object_watch_enabled": True,
        "object_watch_library": str(tmp_path / "some-other-library"),
    }]}
    try:
        build_camera_states(site, output_dir=tmp_path)
    except ValueError as exc:
        assert "canonical site library" in str(exc)
    else:
        raise AssertionError("conflicting object-watch library was accepted")


def test_hot_enable_attaches_shared_runtime_and_uses_canonical_defaults(tmp_path, monkeypatch):
    from cvti.object_watch.runtime_config import ObjectWatchConfig, write_config
    from cvti.serving.camera import build_camera_states
    from cvti.serving.pipeline import MultiStreamPipeline

    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"rules": [{
        "name": "watch_box", "trigger": {"detector": "object_watch",
        "state": "object_seen", "object_id": "box", "zone": "door"},
    }]}))
    write_config(tmp_path, ObjectWatchConfig(
        model_path=tmp_path / "model", sample_fps=3.0, max_candidates=9,
    ))
    camera = {"id": "cam1", "source": "clip.mp4", "config": str(rules),
              "object_watch": False}
    built = build_camera_states({"cameras": [camera]}, output_dir=tmp_path)
    state = built["cam1"]["state"]

    class Runtime:
        def __init__(self, config):
            self.config = config
            self.started = False
            self.resets = []
        def start(self): self.started = True
        def stop(self): self.started = False
        def reset_camera(self, *args): self.resets.append(args)

    monkeypatch.setattr("cvti.object_watch.runtime.ObjectWatchRuntime", Runtime)
    pipe = MultiStreamPipeline({"cam1": "clip.mp4"}, camera_states={"cam1": state})
    pipe._started = True
    pipe._refresh_camera_state({**camera, "object_watch": True,
                                "object_watch_max_candidates_per_frame": 12})

    assert state.object_watch
    assert state.object_watch_sample_fps == 3.0
    assert state.object_watch_max_candidates_per_frame == 12
    assert state._object_watch_runtime is pipe._object_watch_runtime
    assert pipe._object_watch_runtime.started
    assert pipe._object_watch_runtime.resets == [("cam1", 0)]
