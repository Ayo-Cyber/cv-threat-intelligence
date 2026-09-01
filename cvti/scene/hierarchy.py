"""Validated site and area context contracts and their artifact store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from cvti.scene.agent_mapper import ALLOWED_ENVIRONMENT_TYPES
from cvti.scene.context_store import (
    _atomic_json_write,
    _directory_name_for,
    utc_now_iso,
)


SITE_TYPES = {
    "manufacturing_plant",
    "supermarket",
    "retail_store",
    "bank",
    "warehouse",
    "office_building",
    "residential_estate",
    "shopping_mall",
    "school",
    "hospital",
    "hotel",
    "transport_hub",
    "mixed_use",
    "unknown",
}

AREA_TYPES = {
    "production_floor",
    "assembly_line",
    "loading_bay",
    "storage_aisle",
    "chemical_store",
    "machine_room",
    "reception",
    "retail_floor",
    "checkout",
    "banking_hall",
    "vault_approach",
    "office_floor",
    "parking_lot",
    "perimeter",
    "entrance",
    "walkway",
    "unknown",
    *ALLOWED_ENVIRONMENT_TYPES,
}

SITE_TYPE_ALIASES = {
    "manufacturing": "manufacturing_plant",
    "factory": "manufacturing_plant",
    "retail": "retail_store",
    "office": "office_building",
    "mall": "shopping_mall",
}
AREA_TYPE_ALIASES = {
    "warehouse": "warehouse_floor",
    "parking": "parking_lot",
    "lobby": "reception",
    "retail_shop": "retail_floor",
    "perimeter_fence": "perimeter",
    "estate_gate": "entrance",
}

_SITE_KEYS = {
    "site_id",
    "site_type",
    "site_description",
    "confidence",
    "evidence_area_ids",
    "generated_at",
}
_AREA_KEYS = {
    "area_id",
    "name",
    "site_type",
    "area_type",
    "area_description",
    "expected_actors",
    "confidence",
    "evidence_camera_ids",
    "conflicts",
    "generated_at",
}


def _canonical(value: Any) -> str:
    return str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_site_type(value: Any) -> str:
    value = SITE_TYPE_ALIASES.get(_canonical(value), _canonical(value))
    return value if value in SITE_TYPES else "unknown"


def normalize_area_type(value: Any) -> str:
    value = AREA_TYPE_ALIASES.get(_canonical(value), _canonical(value))
    return value if value in AREA_TYPES else "unknown"


def _exact_object(payload: dict[str, Any], keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} context must be an object")
    missing = sorted(keys - set(payload))
    if missing:
        raise ValueError(f"{label} context missing required field(s): {', '.join(missing)}")
    extras = sorted(set(payload) - keys)
    if extras:
        raise ValueError(f"{label} context contains unsupported field(s): {', '.join(extras)}")
    return dict(payload)


def _nonempty(value: Any, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("confidence must be a number between 0 and 1")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number between 0 and 1") from exc
    if not 0 <= result <= 1:
        raise ValueError("confidence must be a number between 0 and 1")
    return result


def _timestamp(value: Any) -> str:
    result = str(value)
    try:
        datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generated_at must be an ISO date-time") from exc
    return result


def _unique_strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result = [_nonempty(item, f"{field} entry") for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{field} entries must be unique")
    return result


def validate_site_context(payload: dict[str, Any]) -> dict[str, Any]:
    result = _exact_object(payload, _SITE_KEYS, "site")
    result["site_id"] = _nonempty(result["site_id"], "site_id")
    result["site_type"] = normalize_site_type(result["site_type"])
    result["site_description"] = _nonempty(
        result["site_description"], "site_description"
    )
    result["confidence"] = _confidence(result["confidence"])
    result["evidence_area_ids"] = _unique_strings(
        result["evidence_area_ids"], "evidence_area_ids"
    )
    result["generated_at"] = _timestamp(result["generated_at"])
    return result


def validate_area_context(payload: dict[str, Any]) -> dict[str, Any]:
    result = _exact_object(payload, _AREA_KEYS, "area")
    result["area_id"] = _nonempty(result["area_id"], "area_id")
    result["name"] = _nonempty(result["name"], "name")
    result["site_type"] = normalize_site_type(result["site_type"])
    result["area_type"] = normalize_area_type(result["area_type"])
    result["area_description"] = _nonempty(
        result["area_description"], "area_description"
    )
    result["expected_actors"] = _unique_strings(
        result["expected_actors"], "expected_actors"
    )
    result["confidence"] = _confidence(result["confidence"])
    result["evidence_camera_ids"] = _unique_strings(
        result["evidence_camera_ids"], "evidence_camera_ids"
    )
    if not isinstance(result["conflicts"], list):
        raise ValueError("conflicts must be an array")
    result["generated_at"] = _timestamp(result["generated_at"])
    return result


class HierarchyContextStore:
    """Keep machine proposals separate from operator-approved hierarchy."""

    def __init__(self, context_root: str | Path) -> None:
        self.context_root = Path(context_root)
        self.site_dir = self.context_root / "_site"
        self.areas_dir = self.context_root / "_areas"

    def area_dir(self, area_id: str) -> Path:
        return self.areas_dir / _directory_name_for(_nonempty(area_id, "area_id"))

    def _load(self, reviewed: Path, proposal: Path, validator) -> dict[str, Any] | None:
        import json

        for path in (reviewed, proposal):
            try:
                return validator(json.loads(path.read_text()))
            except (OSError, TypeError, ValueError):
                continue
        return None

    def load_site(self) -> dict[str, Any] | None:
        return self._load(
            self.site_dir / "site_context.json",
            self.site_dir / "site_context.proposal.json",
            validate_site_context,
        )

    def save_site_proposal(self, context: dict[str, Any]) -> dict[str, Any]:
        context = validate_site_context(context)
        _atomic_json_write(self.site_dir / "site_context.proposal.json", context)
        return context

    def approve_site(self, context: dict[str, Any], reviewer: str) -> dict[str, Any]:
        context = validate_site_context(context)
        _atomic_json_write(self.site_dir / "site_context.json", context)
        _atomic_json_write(
            self.site_dir / "review.json",
            {"reviewed_at": utc_now_iso(), "reviewed_by": _nonempty(reviewer, "reviewer")},
        )
        return context

    def load_area(self, area_id: str) -> dict[str, Any] | None:
        directory = self.area_dir(area_id)
        return self._load(
            directory / "area_context.json",
            directory / "area_context.proposal.json",
            validate_area_context,
        )

    def save_area_proposal(self, context: dict[str, Any]) -> dict[str, Any]:
        context = validate_area_context(context)
        _atomic_json_write(
            self.area_dir(context["area_id"]) / "area_context.proposal.json", context
        )
        return context

    def approve_area(self, context: dict[str, Any], reviewer: str) -> dict[str, Any]:
        context = validate_area_context(context)
        directory = self.area_dir(context["area_id"])
        _atomic_json_write(directory / "area_context.json", context)
        _atomic_json_write(
            directory / "review.json",
            {"reviewed_at": utc_now_iso(), "reviewed_by": _nonempty(reviewer, "reviewer")},
        )
        return context
