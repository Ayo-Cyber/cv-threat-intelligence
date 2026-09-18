from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvti.object_watch.runtime_config import (
    ObjectWatchConfig, advisory_inference_lock, configured_backend_metadata, preflight,
    resolve_config, write_config,
)


def fake_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "siglip",
        "vision_config": {
            "model_type": "siglip_vision_model",
            "patch_size": 16,
        },
    }))
    (model / "preprocessor_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"local-weights")
    return model


def test_resolver_defaults_to_local_siglip_and_fails_missing_assets(tmp_path: Path):
    config = resolve_config(tmp_path)
    assert config.backend == "siglip"
    assert config.device == "cpu"
    assert config.sample_fps == 1.0
    assert config.max_candidates == 24
    assert config.result_ttl_seconds == 2.0
    readiness = preflight(config)
    assert readiness.status == "unavailable"
    assert readiness.model_fingerprint is None


def test_config_round_trip_and_artifact_fingerprint(tmp_path: Path):
    model = fake_model(tmp_path)
    write_config(tmp_path, ObjectWatchConfig(model_path=model, sample_fps=2.0,
                                             max_candidates=12, result_ttl_seconds=3.0))
    config = resolve_config(tmp_path)
    readiness = preflight(config)
    assert readiness.status == "structurally_available"
    assert not readiness.ready
    assert readiness.model_fingerprint.startswith("siglip-")
    assert readiness.dimensions == 768
    assert readiness.structurally_available
    assert not readiness.executable_verified
    metadata = configured_backend_metadata(config)
    assert metadata.dimensions == 768
    assert metadata.fingerprint == readiness.model_fingerprint
    assert config.model_path == model.resolve()
    assert config.sample_fps == 2.0
    (model / "model.safetensors").write_bytes(b"changed-local-weights")
    assert preflight(config).model_fingerprint != readiness.model_fingerprint


def test_optional_missing_proposer_is_degraded_not_unavailable(tmp_path: Path):
    config = ObjectWatchConfig(model_path=fake_model(tmp_path),
                               proposal_provider="yolo_world",
                               world_weights=tmp_path / "missing.pt")
    assert preflight(config).status == "degraded"


def test_siglip_dimension_uses_config_semantics_and_honours_nondefault(tmp_path: Path):
    model = fake_model(tmp_path)
    assert preflight(ObjectWatchConfig(model_path=model)).dimensions == 768
    (model / "config.json").write_text(json.dumps({
        "model_type": "siglip",
        "vision_config": {
            "model_type": "siglip_vision_model",
            "patch_size": 16,
            "hidden_size": 1024,
        },
    }))
    assert preflight(ObjectWatchConfig(model_path=model)).dimensions == 1024


@pytest.mark.parametrize("model_type", ["siglip2", "clip", None, 123])
def test_siglip_dimension_rejects_unsupported_or_malformed_type(
    tmp_path: Path, model_type: object
):
    model = fake_model(tmp_path)
    (model / "config.json").write_text(json.dumps({
        "model_type": model_type,
        "vision_config": {"model_type": "siglip_vision_model", "patch_size": 16},
    }))
    readiness = preflight(ObjectWatchConfig(model_path=model))
    assert readiness.status == "unavailable"
    assert readiness.dimensions is None
    assert any("model_type" in reason or "SigLIP2" in reason for reason in readiness.reasons)


def test_production_resolver_rejects_hash_and_remote_paths(tmp_path: Path):
    library = tmp_path / "object_library"
    library.mkdir()
    (library / "runtime.json").write_text(json.dumps({"backend": "hash"}))
    with pytest.raises(ValueError, match="test-only"):
        resolve_config(tmp_path)
    (library / "runtime.json").write_text(json.dumps({
        "backend": "siglip", "model_path": "https://example.invalid/model"
    }))
    with pytest.raises(ValueError, match="local filesystem"):
        resolve_config(tmp_path)


def test_nonblocking_advisory_lock_reports_busy(tmp_path: Path):
    with advisory_inference_lock(tmp_path, blocking=True) as acquired:
        assert acquired
        with advisory_inference_lock(tmp_path, blocking=False) as second:
            assert not second
