from __future__ import annotations

from cvti.scene.aggregation import aggregate_area, aggregate_site
from _scene_hierarchy_fixtures import area, camera_observation


def test_three_agreeing_camera_views_produce_one_area_proposal() -> None:
    proposal = aggregate_area(
        {"id": "production", "name": "Production"},
        [
            camera_observation("cam1", "manufacturing_plant", "production_floor", 0.9),
            camera_observation("cam2", "manufacturing_plant", "production_floor", 0.8),
            camera_observation("cam3", "manufacturing_plant", "production_floor", 0.7),
        ],
    )

    assert proposal.context["area_type"] == "production_floor"
    assert proposal.context["site_type"] == "manufacturing_plant"
    assert proposal.bulk_reviewable is True


def test_high_confidence_disagreement_is_visible_not_averaged_away() -> None:
    proposal = aggregate_area(
        {"id": "front", "name": "Front"},
        [
            camera_observation("cam1", "bank", "banking_hall", 0.9),
            camera_observation("cam2", "supermarket", "checkout", 0.9),
        ],
    )

    assert proposal.context["area_type"] == "unknown"
    assert proposal.bulk_reviewable is False
    assert {conflict.field for conflict in proposal.conflicts} == {
        "site_type",
        "area_type",
    }


def test_low_confidence_outlier_does_not_block_agreement() -> None:
    proposal = aggregate_area(
        {"id": "loading", "name": "Loading"},
        [
            camera_observation("cam1", "warehouse", "loading_bay", 0.9),
            camera_observation("cam2", "supermarket", "checkout", 0.3),
        ],
    )

    assert proposal.context["area_type"] == "loading_bay"
    assert proposal.bulk_reviewable is True


def test_reviewed_area_is_authoritative() -> None:
    reviewed = area("production_floor", "production")
    reviewed["site_type"] = "manufacturing_plant"

    proposal = aggregate_area(
        {"id": "production", "name": "Production"},
        [camera_observation("cam1", "warehouse", "warehouse_floor", 0.99)],
        reviewed=reviewed,
    )

    assert proposal.context == reviewed
    assert proposal.bulk_reviewable is True


def test_production_areas_infer_manufacturing_site() -> None:
    result = aggregate_site(
        {"site_id": "factory"},
        [area("production_floor"), area("assembly_line"), area("loading_bay")],
    )

    assert result["site_type"] == "manufacturing_plant"


def test_parking_alone_does_not_claim_a_site_type() -> None:
    result = aggregate_site({"site_id": "site"}, [area("parking_lot")])

    assert result["site_type"] == "unknown"
