from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from cvti.scene import SceneContextStore as PublicSceneContextStore
from cvti.scene.agent_mapper import MappingResult
from cvti.scene.context_store import (
    SceneContextStore,
    normalize_environment_type,
    source_fingerprint,
    validate_scene_context,
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _context(environment: str = "retail_shop") -> dict:
    return {
        "camera_id": "cam_1",
        "source_type": "video_file",
        "environment_type": environment,
        "scene_description": "A monitored customer area.",
        "expected_actors": ["customers", "staff"],
        "zones": [
            {
                "id": "checkout",
                "label": "Checkout",
                "role": "checkout",
                "bbox": [10, 20, 100, 80],
            }
        ],
        "confidence": 0.82,
        "generated_at": _now(),
        "source_frame_path": "unused/source_frame.jpg",
        "notes": "",
    }


def _mapping_result(environment: str = "retail_shop") -> MappingResult:
    frame = np.full((60, 80, 3), 127, dtype=np.uint8)
    return MappingResult(_context(environment), frame, '{"environment_type":"retail_shop"}')


def _source(tmp_path, content: bytes = b"clip"):
    path = tmp_path / "clip.mp4"
    path.write_bytes(content)
    return path


def test_scene_package_exports_context_store() -> None:
    assert PublicSceneContextStore is SceneContextStore


def test_store_rejects_camera_path_traversal(tmp_path) -> None:
    with pytest.raises(ValueError, match="camera_id"):
        SceneContextStore(tmp_path, "../outside")


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("retail", "retail_shop"),
        ("parking", "parking_lot"),
        ("warehouse", "warehouse_floor"),
        ("street", "estate_street"),
    ],
)
def test_environment_aliases_normalize(raw: str, canonical: str) -> None:
    assert normalize_environment_type(raw) == canonical


def test_schema_validation_rejects_missing_required_field() -> None:
    context = _context()
    context.pop("expected_actors")

    with pytest.raises(ValueError, match="expected_actors"):
        validate_scene_context(context)


def test_schema_validation_rejects_extra_fields() -> None:
    context = _context()
    context["risk_hints"] = ["theft"]

    with pytest.raises(ValueError, match="risk_hints"):
        validate_scene_context(context)


def test_rtsp_fingerprint_never_contains_credentials() -> None:
    value = source_fingerprint("rtsp://alice:secret@10.0.0.8:554/live?profile=1")

    assert value.startswith("sha256:")
    assert "alice" not in value
    assert "secret" not in value


def test_save_mapping_writes_site_scoped_artifacts_atomically(tmp_path) -> None:
    source = _source(tmp_path)
    store = SceneContextStore(tmp_path / "context", "cam_1")

    resolution = store.save_mapping(_mapping_result(), source)

    context = json.loads(store.context_path.read_text())
    status = json.loads(store.status_path.read_text())
    assert context["camera_id"] == "cam_1"
    assert context["source_frame_path"] == str(store.frame_path)
    assert status["status"] == "ready_unreviewed"
    assert store.frame_path.read_bytes().startswith(b"\xff\xd8")
    assert resolution.usable is True
    assert not list(store.directory.glob("*.tmp"))
    assert not store.raw_response_path.exists()


def test_raw_response_is_debug_only(tmp_path) -> None:
    store = SceneContextStore(tmp_path / "context", "cam_1")
    store.save_mapping(_mapping_result(), _source(tmp_path), dump_raw_response=True)

    assert store.raw_response_path.read_text().startswith("{")


def test_manual_context_outranks_reviewed_cache(tmp_path) -> None:
    source = _source(tmp_path)
    store = SceneContextStore(tmp_path / "context", "cam_1")
    store.approve(_context("parking_lot"), "owner", source)

    result = store.resolve(
        source, "require_reviewed", manual_context=_context("retail_shop")
    )

    assert result.provenance == "manual"
    assert result.usable is True
    assert result.context["environment_type"] == "retail_shop"


def test_reviewed_matching_cache_is_usable_under_require_reviewed(tmp_path) -> None:
    source = _source(tmp_path)
    store = SceneContextStore(tmp_path / "context", "cam_1")
    store.approve(_context(), "owner", source)

    result = store.resolve(source, "require_reviewed")

    assert result.usable is True
    assert result.provenance == "cache"
    assert result.status.status == "ready_reviewed"


def test_unreviewed_cache_is_usable_only_under_auto(tmp_path) -> None:
    source = _source(tmp_path)
    store = SceneContextStore(tmp_path / "context", "cam_1")
    store.save_mapping(_mapping_result(), source)

    assert store.resolve(source, "auto").usable is True
    assert store.resolve(source, "require_reviewed").usable is False


def test_source_change_marks_cache_stale(tmp_path) -> None:
    source = _source(tmp_path, b"first")
    store = SceneContextStore(tmp_path / "context", "cam_1")
    store.save_mapping(_mapping_result(), source)
    source.write_bytes(b"changed source identity")

    result = store.resolve(source, "auto")

    assert result.usable is False
    assert result.status.status == "stale"


def test_approve_records_reviewer_and_corrected_context(tmp_path) -> None:
    source = _source(tmp_path)
    store = SceneContextStore(tmp_path / "context", "cam_1")

    result = store.approve(_context("parking_lot"), "ayo", source)

    assert result.status.reviewed_by == "ayo"
    assert result.status.reviewed_at
    assert result.context["environment_type"] == "parking_lot"


def test_legacy_two_field_context_imports_as_unreviewed_canonical(tmp_path) -> None:
    source = _source(tmp_path)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        '{"environment_type":"retail","scene_description":"A shop."}'
    )
    store = SceneContextStore(tmp_path / "site-context", "cam_1")

    result = store.resolve(source, "auto", legacy_context_path=legacy)

    assert result.context["environment_type"] == "retail_shop"
    assert result.context["expected_actors"] == []
    assert result.status.status == "ready_unreviewed"
    assert result.usable is True


def test_manual_policy_does_not_use_unreviewed_cache(tmp_path) -> None:
    source = _source(tmp_path)
    store = SceneContextStore(tmp_path / "context", "cam_1")
    store.save_mapping(_mapping_result(), source)

    result = store.resolve(source, "manual")

    assert result.usable is False
    assert result.status.status == "ready_unreviewed"
