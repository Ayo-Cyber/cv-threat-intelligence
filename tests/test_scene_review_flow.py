from __future__ import annotations

import pytest
import numpy as np

from _backend_helper import signed_in
from _scene_hierarchy_fixtures import area_context, camera_context, write_site
from cvti.scene.agent_mapper import MappingResult
from cvti.scene.hierarchy import HierarchyContextStore
from cvti.security.permissions import PermissionDenied
from cvti.scene.context_store import SceneContextStore


def _backend(tmp_path, role="owner"):
    site = write_site(tmp_path, ["north", "south", "exit"])
    return signed_in(
        role,
        site_path=str(site),
        db_path=str(tmp_path / "out/events.db"),
        enable_demo=False,
    )


def _map_camera(backend, tmp_path, camera_id: str, confidence: float = 0.9) -> None:
    camera = next(item for item in backend.list_cameras() if item["id"] == camera_id)
    context = camera_context(camera_id, confidence=confidence)
    context["environment_type"] = "warehouse_floor"
    SceneContextStore(tmp_path / "out/context", camera_id).save_mapping(
        MappingResult(
            context,
            np.zeros((20, 20, 3), dtype=np.uint8),
            "{}",
        ),
        camera["source"],
    )


def _save_area_proposal(tmp_path, camera_ids: list[str]) -> dict:
    context = area_context()
    context["evidence_camera_ids"] = camera_ids
    HierarchyContextStore(tmp_path / "out/context").save_area_proposal(context)
    return context


def test_scene_review_summary_groups_three_cameras_into_one_area(tmp_path) -> None:
    backend = _backend(tmp_path)

    summary = backend.scene_review_summary()

    assert summary["areas"][0]["camera_ids"] == ["north", "south", "exit"]
    assert summary["counts"]["total_cameras"] == 3
    assert "clip.mp4" not in str(summary)
    assert str(tmp_path) not in str(summary)


def test_bulk_review_requires_context_and_frame_for_every_camera(tmp_path) -> None:
    backend = _backend(tmp_path)
    for camera_id in ("north", "south", "exit"):
        _map_camera(backend, tmp_path, camera_id)
    _save_area_proposal(tmp_path, ["north", "south", "exit"])

    area = backend.scene_review_summary()["areas"][0]

    assert area["bulk_reviewable"] is True
    assert all(camera["evidence_ready"] for camera in area["cameras"])
    assert {camera["environment_type"] for camera in area["cameras"]} == {
        "warehouse_floor"
    }
    assert {camera["confidence"] for camera in area["cameras"]} == {0.9}


def test_bulk_review_is_blocked_when_a_camera_is_missing_from_evidence(tmp_path) -> None:
    backend = _backend(tmp_path)
    for camera_id in ("north", "south", "exit"):
        _map_camera(backend, tmp_path, camera_id)
    context = _save_area_proposal(tmp_path, ["north", "south"])

    area = backend.scene_review_summary()["areas"][0]

    assert area["bulk_reviewable"] is False
    assert area["missing_evidence_camera_ids"] == ["exit"]
    with pytest.raises(ValueError, match="complete camera evidence"):
        backend.approve_area_context("production", context)


def test_bulk_review_is_blocked_when_a_camera_frame_is_missing(tmp_path) -> None:
    backend = _backend(tmp_path)
    for camera_id in ("north", "south", "exit"):
        _map_camera(backend, tmp_path, camera_id)
    _save_area_proposal(tmp_path, ["north", "south", "exit"])
    SceneContextStore(tmp_path / "out/context", "exit").frame_path.unlink()

    area = backend.scene_review_summary()["areas"][0]

    assert area["bulk_reviewable"] is False
    assert area["missing_evidence_camera_ids"] == ["exit"]


def test_operator_can_view_but_cannot_assign_or_approve(tmp_path) -> None:
    backend = _backend(tmp_path, role="operator")
    assert backend.scene_review_summary()["areas"]

    with pytest.raises(PermissionDenied):
        backend.assign_camera_area("north", "production")
    with pytest.raises(PermissionDenied):
        backend.approve_area_context("production", area_context())


def test_owner_can_create_assign_and_approve_area(tmp_path) -> None:
    backend = _backend(tmp_path)
    backend.create_area({"id": "exit_area", "name": "Exit"})
    backend.assign_camera_area("exit", "exit_area")
    exit_camera = next(camera for camera in backend.list_cameras() if camera["id"] == "exit")
    SceneContextStore(tmp_path / "out/context", "exit").save_mapping(
        MappingResult(
            camera_context("exit", area_id="exit_area", area_type="entrance"),
            np.zeros((20, 20, 3), dtype=np.uint8),
            "{}",
        ),
        exit_camera["source"],
    )

    context = area_context("entrance", "exit_area")
    context["evidence_camera_ids"] = ["exit"]
    result = backend.approve_area_context("exit_area", context)

    assert result["ok"] is True
    assert backend.area_context("exit_area")["area_type"] == "entrance"
    assert SceneContextStore(tmp_path / "out/context", "exit").load_status().status == "ready_reviewed"
    assert any(entry.target == "area:exit_area" for entry in backend.audit.entries(20))


def test_bridge_declares_grouped_scene_review_slots() -> None:
    source = open("cvti/app/bridge.py").read()
    for method in (
        "sceneReviewSummary",
        "listAreas",
        "createArea",
        "assignCameraArea",
        "approveAreaContext",
        "updateSiteContext",
        "approveSiteContext",
        "enqueueSceneMapping",
        "sceneMappingProgress",
    ):
        assert f"def {method}" in source
