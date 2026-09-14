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
