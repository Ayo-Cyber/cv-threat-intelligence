from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from cvti.scene import SceneContextStore as PublicSceneContextStore
from cvti.scene.agent_mapper import MappingResult
from cvti.scene.context_store import (
    SceneContextStore,
    _atomic_bytes_write,
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


def test_store_contains_camera_path_traversal(tmp_path) -> None:
    """Repinned 31 Aug: rejecting ids crashed the engine on every feed with a
    human-named camera. Hostile ids are now CONTAINED — slugged into a
    directory that cannot leave the context root — instead of rejected."""
    store = SceneContextStore(tmp_path, "../outside")
    assert tmp_path in store.directory.parents
    with pytest.raises(ValueError, match="camera_id"):
        SceneContextStore(tmp_path, "   ")


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


def test_late_mapping_becomes_proposal_instead_of_overwriting_review(tmp_path) -> None:
    source = _source(tmp_path)
    store = SceneContextStore(tmp_path / "context", "cam_1")
    reviewed = store.approve(_context("parking_lot"), "owner", source)

    resolution = store.save_mapping(
        _mapping_result("retail_shop"), source, dump_raw_response=True
    )

    assert resolution.context == reviewed.context
    assert resolution.status.status == "ready_reviewed"
    assert resolution.provenance == "reviewed_during_mapping"
    assert json.loads(store.context_path.read_text())["environment_type"] == "parking_lot"
    assert (
        json.loads(store.proposal_context_path.read_text())["environment_type"]
        == "retail_shop"
    )
    assert store.proposal_frame_path.read_bytes().startswith(b"\xff\xd8")
    assert store.proposal_raw_response_path.read_text().startswith("{")


def test_windows_atomic_replace_retries_transient_permission_errors(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "context.json"
    real_replace = Path.replace
    attempts = 0

    def flaky_replace(path: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("file is being read")
        return real_replace(path, target)

    monkeypatch.setattr("cvti.scene.context_store.sys.platform", "win32")
    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr("cvti.scene.context_store.time.sleep", lambda _: None)

    _atomic_bytes_write(destination, b"complete")

    assert destination.read_bytes() == b"complete"
    assert attempts == 3


def test_atomic_replace_does_not_retry_permission_errors_off_windows(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "context.json"
    attempts = 0

    def denied_replace(path: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        raise PermissionError("denied")

    monkeypatch.setattr("cvti.scene.context_store.sys.platform", "darwin")
    monkeypatch.setattr(Path, "replace", denied_replace)

    with pytest.raises(PermissionError, match="denied"):
        _atomic_bytes_write(destination, b"never committed")

    assert attempts == 1


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


def test_unsafe_ids_get_safe_distinct_directories(tmp_path):
    """'Dublin Street' must work (v1.5.0 refused it and the engine died), a
    slug collision must not share state, and a hostile id must stay inside
    the context root."""
    from cvti.scene.context_store import SceneContextStore

    spaced = SceneContextStore(tmp_path, "Dublin Street")
    assert spaced.camera_id == "Dublin Street"
    assert spaced.directory.parent == tmp_path

    a = SceneContextStore(tmp_path, "Aisle 4")
    b = SceneContextStore(tmp_path, "Aisle_4")
    assert a.directory != b.directory

    hostile = SceneContextStore(tmp_path, "../../etc/passwd")
    assert tmp_path in hostile.directory.parents
