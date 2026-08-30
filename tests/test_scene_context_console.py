from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from _backend_helper import signed_in
from cvti.scene.agent_mapper import MappingResult
from cvti.scene.context_store import SceneContextStore
from cvti.security.permissions import PermissionDenied


def _context(environment: str = "parking_lot") -> dict:
    return {
        "camera_id": "cam_1",
        "source_type": "video_file",
        "environment_type": environment,
        "scene_description": "A monitored parking and delivery area.",
        "expected_actors": ["drivers", "security staff"],
        "zones": [
            {
                "id": "delivery_entry",
                "label": "Delivery entry",
                "role": "entry",
                "bbox": [10, 20, 70, 80],
            }
        ],
        "confidence": 0.8,
        "generated_at": "2026-08-30T10:00:00Z",
        "source_frame_path": "unused.jpg",
        "notes": "",
    }


def _backend(tmp_path, role: str = "owner"):
    site = tmp_path / "site.json"
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"clip")
    site.write_text(
        json.dumps({"cameras": [{"id": "cam_1", "source": str(source)}]})
    )
    backend = signed_in(
        role,
        site_path=str(site),
        db_path=str(tmp_path / "events.db"),
        enable_demo=False,
    )
    return backend, source


def _seed_mapping(tmp_path, source: Path) -> SceneContextStore:
    store = SceneContextStore(tmp_path / "context", "cam_1")
    result = MappingResult(
        _context(), np.zeros((100, 100, 3), dtype=np.uint8), "{}"
    )
    store.save_mapping(result, source)
    return store


def test_scene_context_reads_db_parent_context_not_cwd(tmp_path, monkeypatch) -> None:
    backend, source = _backend(tmp_path)
    _seed_mapping(tmp_path, source)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = backend.scene_context("cam_1")

    assert result["environment_type"] == "parking_lot"
    assert result["mapping"]["status"] == "ready_unreviewed"
    assert result["source_frame_uri"].startswith("data:image/jpeg;base64,")


def test_scene_context_returns_failure_status_before_context_exists(tmp_path) -> None:
    backend, source = _backend(tmp_path)
    store = SceneContextStore(tmp_path / "context", "cam_1")
    store.mark_failed(source, "ollama unavailable")

    result = backend.scene_context("cam_1")

    assert result["mapping"]["status"] == "failed"
    assert result["mapping"]["error"] == "ollama unavailable"
    assert result["scene_description"] == ""


def test_operator_cannot_edit_or_approve_scene_context(tmp_path) -> None:
    backend, source = _backend(tmp_path, role="operator")
    _seed_mapping(tmp_path, source)

    with pytest.raises(PermissionDenied):
        backend.update_scene_context("cam_1", _context("retail_shop"))
    with pytest.raises(PermissionDenied):
        backend.approve_scene_context("cam_1", _context())


def test_installer_can_edit_and_approve_scene_context(tmp_path) -> None:
    backend, source = _backend(tmp_path, role="installer")
    _seed_mapping(tmp_path, source)

    updated = backend.update_scene_context("cam_1", _context("retail_shop"))
    approved = backend.approve_scene_context("cam_1", _context("retail_shop"))

    assert updated["ok"] is True
    assert approved["ok"] is True
    assert approved["context"]["environment_type"] == "retail_shop"
    assert approved["context"]["mapping"]["status"] == "ready_reviewed"


def test_owner_approval_records_identity_and_audit_entry(tmp_path) -> None:
    backend, source = _backend(tmp_path)
    store = _seed_mapping(tmp_path, source)

    backend.approve_scene_context("cam_1", _context())

    assert store.load_status().reviewed_by == "owner"
    entries = backend.audit.entries(limit=20)
    assert any(
        entry.action == "config_change"
        and entry.target == "camera:cam_1"
        and entry.detail.get("scene_context") == "approved"
        for entry in entries
    )


def test_remap_marks_context_stale_without_deleting_it(tmp_path) -> None:
    backend, source = _backend(tmp_path, role="installer")
    store = _seed_mapping(tmp_path, source)

    result = backend.request_scene_remap("cam_1")

    assert result["ok"] is True
    assert store.context_path.exists()
    assert store.frame_path.exists()
    assert store.load_status().status == "stale"


def test_accept_suggested_zone_converts_bbox_to_active_polygon(tmp_path) -> None:
    backend, source = _backend(tmp_path, role="installer")
    _seed_mapping(tmp_path, source)
    backend._zones_file = lambda camera_id: tmp_path / f"{camera_id}-zones.json"

    result = backend.accept_suggested_zone(
        "cam_1", "delivery_entry", dwell_seconds=5.0
    )

    assert result["ok"] is True
    assert result["zone"]["polygon"] == [
        [10, 20],
        [70, 20],
        [70, 80],
        [10, 80],
    ]
    assert result["zone"]["context_role"] == "entry"
    camera = backend.list_cameras()[0]
    assert camera["accepted_zone_roles"] == ["entry"]


def test_qt_bridge_exposes_all_scene_review_actions() -> None:
    from cvti.app.bridge import Backend

    source = inspect.getsource(Backend)
    for method in (
        "updateSceneContext",
        "approveSceneContext",
        "requestSceneRemap",
        "acceptSuggestedZone",
    ):
        assert f"def {method}" in source
