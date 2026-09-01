"""Deterministic camera-to-area and area-to-site context aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cvti.scene.agent_mapper import utc_now_iso
from cvti.scene.hierarchy import validate_area_context, validate_site_context


MIN_BULK_CONFIDENCE = 0.60


@dataclass(frozen=True)
class AggregationConflict:
    field: str
    camera_ids: list[str]
    values: list[str]
    reason: str


@dataclass(frozen=True)
class AreaProposal:
    context: dict[str, Any]
    conflicts: list[AggregationConflict]
    bulk_reviewable: bool


def _consensus(
    contexts: list[dict[str, Any]], field: str, output_field: str
) -> tuple[str, AggregationConflict | None]:
    evidence = [
        context for context in contexts
        if float(context.get("confidence", 0)) >= MIN_BULK_CONFIDENCE
        and str(context.get(field, "unknown")) != "unknown"
    ]
    values = sorted({str(context[field]) for context in evidence})
    if not values:
        return "unknown", None
    if len(values) == 1:
        return values[0], None
    return "unknown", AggregationConflict(
        field=output_field,
        camera_ids=[str(context.get("camera_id", "")) for context in evidence],
        values=values,
        reason="high-confidence camera observations disagree",
    )


def aggregate_area(
    area: dict[str, Any],
    camera_contexts: list[dict[str, Any]],
    reviewed: dict[str, Any] | None = None,
) -> AreaProposal:
    if reviewed is not None:
        return AreaProposal(validate_area_context(reviewed), [], True)

    site_type, site_conflict = _consensus(
        camera_contexts, "site_type_candidate", "site_type"
    )
    area_type, area_conflict = _consensus(
        camera_contexts, "area_type_candidate", "area_type"
    )
    conflicts = [item for item in (site_conflict, area_conflict) if item]
    evidence = [
        context for context in camera_contexts
        if float(context.get("confidence", 0)) >= MIN_BULK_CONFIDENCE
    ]
    actors: list[str] = []
    for context in evidence:
        for actor in context.get("expected_actors", []):
            actor = str(actor).strip()
            if actor and actor not in actors:
                actors.append(actor)
    descriptions = [
        str(context.get("view_description") or context.get("scene_description") or "").strip()
        for context in evidence
    ]
    descriptions = [description for description in descriptions if description]
    confidence = (
        sum(float(context.get("confidence", 0)) for context in evidence) / len(evidence)
        if evidence and not conflicts else 0.0
    )
    context = validate_area_context({
        "area_id": str(area["id"]),
        "name": str(area.get("name") or area["id"]),
        "site_type": site_type,
        "area_type": area_type,
        "area_description": " ".join(descriptions) or "Insufficient visual evidence.",
        "expected_actors": actors,
        "confidence": confidence,
        "evidence_camera_ids": [str(item.get("camera_id", "")) for item in evidence],
        "conflicts": [asdict(conflict) for conflict in conflicts],
        "generated_at": utc_now_iso(),
    })
    return AreaProposal(context, conflicts, bool(evidence) and not conflicts)


_AREA_SITE_COMPATIBILITY = {
    "production_floor": "manufacturing_plant",
    "assembly_line": "manufacturing_plant",
    "chemical_store": "manufacturing_plant",
    "machine_room": "manufacturing_plant",
    "retail_floor": "supermarket",
    "checkout": "supermarket",
    "banking_hall": "bank",
    "vault_approach": "bank",
    "warehouse_floor": "warehouse",
    "storage_aisle": "warehouse",
}


def aggregate_site(
    site: dict[str, Any],
    area_contexts: list[dict[str, Any]],
    reviewed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if reviewed is not None:
        return validate_site_context(reviewed)

    direct = {
        str(area.get("site_type")) for area in area_contexts
        if str(area.get("site_type", "unknown")) != "unknown"
        and float(area.get("confidence", 0)) >= MIN_BULK_CONFIDENCE
    }
    inferred = {
        _AREA_SITE_COMPATIBILITY[str(area.get("area_type"))]
        for area in area_contexts
        if str(area.get("area_type")) in _AREA_SITE_COMPATIBILITY
    }
    candidates = direct or inferred
    site_type = next(iter(candidates)) if len(candidates) == 1 else "unknown"
    evidence_ids = list(dict.fromkeys(
        str(area.get("area_id", "")) for area in area_contexts
    ))
    descriptions = [str(area.get("area_description", "")).strip() for area in area_contexts]
    descriptions = [description for description in descriptions if description]
    return validate_site_context({
        "site_id": str(site.get("site_id") or site.get("id") or "site"),
        "site_type": site_type,
        "site_description": " ".join(descriptions) or "Insufficient area evidence.",
        "confidence": (
            sum(float(area.get("confidence", 0)) for area in area_contexts)
            / len(area_contexts)
            if area_contexts and site_type != "unknown" else 0.0
        ),
        "evidence_area_ids": evidence_ids,
        "generated_at": utc_now_iso(),
    })
