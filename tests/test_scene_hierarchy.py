from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvti.serving import onboarding
from cvti.scene.context_store import validate_scene_context
from cvti.scene.hierarchy import (
    AREA_TYPES,
    HierarchyContextStore,
    normalize_site_type,
    validate_area_context,
    validate_site_context,
)
from _scene_hierarchy_fixtures import (
    area_context,
    camera_context,
    site_context,
    write_site,
)


def test_manufacturing_and_supermarket_are_distinct_site_types() -> None:
    assert normalize_site_type("manufacturing plant") == "manufacturing_plant"
    assert normalize_site_type("supermarket") == "supermarket"


def test_json_schemas_match_runtime_area_vocabulary() -> None:
    root = Path(__file__).resolve().parents[1]
    area_schema = json.loads((root / "schemas/area_context.schema.json").read_text())
    camera_schema = json.loads((root / "schemas/scene_context.schema.json").read_text())

    assert set(area_schema["properties"]["area_type"]["enum"]) == AREA_TYPES
    assert set(camera_schema["properties"]["area_type_candidate"]["enum"]) == AREA_TYPES


def test_camera_context_accepts_additive_hierarchy_fields() -> None:
    context = camera_context()

    validated = validate_scene_context(context)

    assert validated["area_id"] == "production"
    assert validated["site_type_candidate"] == "manufacturing_plant"


def test_legacy_camera_context_remains_valid() -> None:
    context = camera_context()
    for field in (
        "area_id",
        "site_type_candidate",
        "area_type_candidate",
        "view_description",
    ):
        context.pop(field)

    assert validate_scene_context(context)["camera_id"] == "cam1"


def test_site_context_rejects_unknown_fields() -> None:
    context = site_context()
    context["risk_hints"] = ["theft"]

    with pytest.raises(ValueError, match="risk_hints"):
        validate_site_context(context)


def test_area_context_rejects_duplicate_evidence_camera_ids() -> None:
    context = area_context()
    context["evidence_camera_ids"] = ["cam1", "cam1"]

    with pytest.raises(ValueError, match="evidence_camera_ids"):
        validate_area_context(context)


def test_hierarchy_store_preserves_existing_camera_namespace(tmp_path) -> None:
    store = HierarchyContextStore(tmp_path / "context")

    assert store.site_dir == tmp_path / "context/_site"
    assert store.area_dir("Production Floor").parent == tmp_path / "context/_areas"


def test_reviewed_area_outranks_new_proposal(tmp_path) -> None:
    store = HierarchyContextStore(tmp_path / "context")
    store.approve_area(area_context("production_floor"), "owner")
    store.save_area_proposal(area_context("warehouse_floor"))

    assert store.load_area("production")["area_type"] == "production_floor"


def test_reviewed_site_outranks_new_proposal(tmp_path) -> None:
    store = HierarchyContextStore(tmp_path / "context")
    store.approve_site(site_context("manufacturing_plant"), "owner")
    store.save_site_proposal(site_context("warehouse"))

    assert store.load_site()["site_type"] == "manufacturing_plant"


def test_loading_legacy_site_does_not_rewrite_it(tmp_path) -> None:
    site = tmp_path / "site.json"
    original = '{"configured":true,"cameras":[{"id":"Front Gate","source":"x"}]}'
    site.write_text(original)

    areas = onboarding.normalized_areas(site)

    assert areas == [{
        "id": "camera--Front_Gate-6bbd15cc",
        "name": "Front Gate",
        "implicit": True,
        "camera_ids": ["Front Gate"],
    }]
    assert site.read_text() == original


def test_three_cameras_can_share_one_area(tmp_path) -> None:
    site = write_site(tmp_path, ["north", "south", "exit"], grouped=False)
    onboarding.upsert_area(site, {"id": "production", "name": "Production"})

    for camera_id in ("north", "south", "exit"):
        onboarding.assign_camera_area(site, camera_id, "production")

    assert {c["area_id"] for c in onboarding.list_cameras(site)} == {"production"}


def test_assignment_rejects_unknown_explicit_area(tmp_path) -> None:
    site = write_site(tmp_path, ["north"], grouped=False)

    with pytest.raises(ValueError, match="unknown area"):
        onboarding.assign_camera_area(site, "north", "missing")


def test_removing_area_makes_member_cameras_implicit(tmp_path) -> None:
    site = write_site(tmp_path, ["north", "south"])

    onboarding.remove_area(site, "production")

    assert all("area_id" not in camera for camera in onboarding.list_cameras(site))
    assert len(onboarding.normalized_areas(site)) == 2
