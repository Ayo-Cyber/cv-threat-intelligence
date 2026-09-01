"""Site-scoped persistence and lifecycle rules for canonical scene context."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2

from cvti.scene.agent_mapper import (
    ALLOWED_ENVIRONMENT_TYPES,
    ALLOWED_SOURCE_TYPES,
    ALLOWED_ZONE_ROLES,
    MappingResult,
    detect_source_type,
    utc_now_iso,
)


MAPPING_STATUSES = {
    "pending",
    "ready_unreviewed",
    "ready_reviewed",
    "stale",
    "failed",
}
SCENE_CONTEXT_POLICIES = {"auto", "require_reviewed", "manual"}
ENVIRONMENT_ALIASES = {
    "retail": "retail_shop",
    "shop": "retail_shop",
    "parking": "parking_lot",
    "warehouse": "warehouse_floor",
    "street": "estate_street",
    "entrance": "estate_gate",
    "office": "office_floor",
    "home": "residential_interior",
    "other": "unknown",
}
_REQUIRED_CONTEXT_KEYS = {
    "camera_id",
    "source_type",
    "environment_type",
    "scene_description",
    "expected_actors",
    "zones",
    "confidence",
    "generated_at",
    "source_frame_path",
}
_OPTIONAL_CONTEXT_KEYS = {
    "notes",
    "area_id",
    "site_type_candidate",
    "area_type_candidate",
    "view_description",
}
_SAFE_CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class MappingStatus:
    status: str
    source_fingerprint: str = ""
    mapped_at: str = ""
    reviewed_at: str = ""
    reviewed_by: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MappingStatus":
        status = str(payload.get("status", "failed"))
        if status not in MAPPING_STATUSES:
            status = "failed"
        return cls(
            status=status,
            source_fingerprint=str(payload.get("source_fingerprint", "")),
            mapped_at=str(payload.get("mapped_at", "")),
            reviewed_at=str(payload.get("reviewed_at", "")),
            reviewed_by=str(payload.get("reviewed_by", "")),
            error=str(payload.get("error", "")),
        )


@dataclass(frozen=True)
class ContextResolution:
    context: dict[str, Any] | None
    status: MappingStatus
    provenance: str
    usable: bool


def normalize_scene_context_policy(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "auto"
    policy = str(value).strip().lower()
    if policy not in SCENE_CONTEXT_POLICIES:
        raise ValueError(
            "scene_context_policy must be one of: auto, require_reviewed, manual"
        )
    return policy


def normalize_environment_type(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower().replace(" ", "_")
    normalized = ENVIRONMENT_ALIASES.get(normalized, normalized)
    return normalized if normalized in ALLOWED_ENVIRONMENT_TYPES else "unknown"


def _parse_iso(value: Any, field: str) -> str:
    text = str(value or "")
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date-time") from exc
    return text


def validate_scene_context(context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(context, dict):
        raise ValueError("scene context must be an object")
    missing = sorted(_REQUIRED_CONTEXT_KEYS - set(context))
    if missing:
        raise ValueError(f"scene context missing required field(s): {', '.join(missing)}")
    extras = sorted(set(context) - _REQUIRED_CONTEXT_KEYS - _OPTIONAL_CONTEXT_KEYS)
    if extras:
        raise ValueError(f"scene context contains unsupported field(s): {', '.join(extras)}")

    result = dict(context)
    camera_id = str(result["camera_id"]).strip()
    if not camera_id:
        raise ValueError("camera_id must not be empty")
    result["camera_id"] = camera_id

    source_type = str(result["source_type"])
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError(f"invalid source_type: {source_type}")
    result["source_type"] = source_type
    result["environment_type"] = normalize_environment_type(
        result["environment_type"]
    )

    description = str(result["scene_description"]).strip()
    if not description:
        raise ValueError("scene_description must not be empty")
    result["scene_description"] = description

    actors = result["expected_actors"]
    if not isinstance(actors, list):
        raise ValueError("expected_actors must be an array")
    cleaned_actors: list[str] = []
    for actor in actors:
        text = str(actor).strip()
        if not text:
            raise ValueError("expected_actors entries must not be empty")
        if text not in cleaned_actors:
            cleaned_actors.append(text)
    result["expected_actors"] = cleaned_actors

    zones = result["zones"]
    if not isinstance(zones, list):
        raise ValueError("zones must be an array")
    cleaned_zones: list[dict[str, Any]] = []
    for zone in zones:
        if not isinstance(zone, dict):
            raise ValueError("each zone must be an object")
        allowed_zone_keys = {"id", "label", "role", "bbox"}
        if set(zone) != allowed_zone_keys:
            raise ValueError("each zone must contain only id, label, role, and bbox")
        zone_id = str(zone["id"]).strip()
        label = str(zone["label"]).strip()
        role = str(zone["role"]).strip()
        bbox = zone["bbox"]
        if not zone_id or not label:
            raise ValueError("zone id and label must not be empty")
        if role not in ALLOWED_ZONE_ROLES:
            raise ValueError(f"invalid zone role: {role}")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
            or any(value < 0 for value in bbox)
        ):
            raise ValueError("zone bbox must contain four non-negative integers")
        cleaned_zones.append(
            {"id": zone_id, "label": label, "role": role, "bbox": list(bbox)}
        )
    result["zones"] = cleaned_zones

    if isinstance(result["confidence"], bool):
        raise ValueError("confidence must be a number between 0 and 1")
    try:
        confidence = float(result["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number between 0 and 1") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be a number between 0 and 1")
    result["confidence"] = confidence
    result["generated_at"] = _parse_iso(result["generated_at"], "generated_at")

    source_frame_path = str(result["source_frame_path"]).strip()
    if not source_frame_path:
        raise ValueError("source_frame_path must not be empty")
    result["source_frame_path"] = source_frame_path
    if "notes" in result:
        result["notes"] = str(result["notes"])
    if "area_id" in result:
        result["area_id"] = str(result["area_id"]).strip()
    if "site_type_candidate" in result:
        from cvti.scene.hierarchy import normalize_site_type

        result["site_type_candidate"] = normalize_site_type(
            result["site_type_candidate"]
        )
    if "area_type_candidate" in result:
        from cvti.scene.hierarchy import normalize_area_type

        result["area_type_candidate"] = normalize_area_type(
            result["area_type_candidate"]
        )
    if "view_description" in result:
        result["view_description"] = str(result["view_description"]).strip()
    return result


def _source_identity(source: int | str | Path) -> str:
    if isinstance(source, int) or str(source).isdigit():
        return f"webcam:{int(source)}"
    raw = str(source)
    parts = urlsplit(raw)
    if parts.scheme.lower() in {"rtsp", "rtsps"}:
        host = parts.hostname or ""
        netloc = host
        if parts.port is not None:
            netloc = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme.lower(), netloc, parts.path, parts.query, ""))
    path = Path(raw).expanduser().resolve(strict=False)
    identity = f"file:{path}"
    try:
        stat = path.stat()
    except OSError:
        return identity
    return f"{identity}|size:{stat.st_size}|mtime_ns:{stat.st_mtime_ns}"


def source_fingerprint(source: int | str | Path) -> str:
    digest = hashlib.sha256(_source_identity(source).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def render_scene_context(context: dict[str, Any] | None) -> str:
    """Render descriptive context without asserting an actor's identity."""
    if not isinstance(context, dict):
        return "A monitored area with no reviewed scene description."
    description = str(context.get("scene_description", "")).strip()
    environment = str(context.get("environment_type", "unknown")).replace("_", " ")
    parts = [description or "A monitored area.", f"Environment: {environment}."]
    actors = [str(actor).strip() for actor in context.get("expected_actors", [])]
    actors = [actor for actor in actors if actor]
    if actors:
        parts.append(f"Expected actors may include {', '.join(actors)}.")
    return " ".join(parts)


def _atomic_bytes_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes_write(
        path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    )


def _directory_name_for(camera_id: str) -> str:
    """A safe directory name for ANY camera id, not a gate on which ids exist.

    Real sites name cameras for humans — 'Dublin Street', 'Forecourt ATM' —
    and rejecting those crashed the engine at startup on every feed that had
    one (field report, 31 Aug: crash-loop before the first frame). The id is
    the site's business; only the DIRECTORY must be filesystem-safe, so
    unsafe ids are slugged, with a short digest so 'Aisle 4' and 'Aisle_4'
    can never share a directory."""
    raw = str(camera_id)
    if _SAFE_CAMERA_ID.fullmatch(raw):
        return raw
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}" if slug else f"camera-{digest}"


class SceneContextStore:
    def __init__(self, context_root: str | Path, camera_id: str) -> None:
        if not str(camera_id).strip():
            raise ValueError("camera_id is empty")
        self.camera_id = str(camera_id)
        self.directory = Path(context_root) / _directory_name_for(camera_id)
        self.context_path = self.directory / "scene_context.json"
        self.status_path = self.directory / "mapping_status.json"
        self.frame_path = self.directory / "source_frame.jpg"
        self.raw_response_path = self.directory / "raw_response.txt"

    def load_status(self) -> MappingStatus:
        try:
            return MappingStatus.from_dict(json.loads(self.status_path.read_text()))
        except (OSError, ValueError, TypeError):
            return MappingStatus(status="pending")

    def _load_context(self) -> dict[str, Any] | None:
        try:
            return validate_scene_context(json.loads(self.context_path.read_text()))
        except (OSError, ValueError, TypeError):
            return None

    def _write_status(self, status: MappingStatus) -> MappingStatus:
        _atomic_json_write(self.status_path, status.to_dict())
        return status

    def _prepare_context(
        self, context: dict[str, Any], source: int | str | Path
    ) -> dict[str, Any]:
        payload = dict(context)
        payload["camera_id"] = self.camera_id
        payload["source_type"] = detect_source_type(
            int(source) if str(source).isdigit() else str(source)
        )
        payload["source_frame_path"] = str(self.frame_path)
        payload.setdefault("generated_at", utc_now_iso())
        payload.setdefault("expected_actors", [])
        payload.setdefault("zones", [])
        payload.setdefault("confidence", 0.0)
        payload.setdefault("notes", "")
        payload["environment_type"] = normalize_environment_type(
            payload.get("environment_type")
        )
        return validate_scene_context(payload)

    def mark_pending(self, source: int | str | Path) -> MappingStatus:
        return self._write_status(
            MappingStatus(
                status="pending", source_fingerprint=source_fingerprint(source)
            )
        )

    def mark_failed(self, source: int | str | Path, error: str) -> MappingStatus:
        previous = self.load_status()
        return self._write_status(
            MappingStatus(
                status="failed",
                source_fingerprint=source_fingerprint(source),
                mapped_at=previous.mapped_at,
                reviewed_at=previous.reviewed_at,
                reviewed_by=previous.reviewed_by,
                error=str(error)[:240],
            )
        )

    def mark_stale(self, source: int | str | Path) -> MappingStatus:
        previous = self.load_status()
        return self._write_status(
            MappingStatus(
                status="stale",
                source_fingerprint=source_fingerprint(source),
                mapped_at=previous.mapped_at,
                reviewed_at=previous.reviewed_at,
                reviewed_by=previous.reviewed_by,
                error="source identity changed or remap was requested",
            )
        )

    def save_mapping(
        self,
        result: MappingResult,
        source: int | str | Path,
        dump_raw_response: bool = False,
    ) -> ContextResolution:
        context = self._prepare_context(result.context, source)
        ok, encoded = cv2.imencode(".jpg", result.selected_frame)
        if not ok:
            raise ValueError("unable to encode representative source frame")
        _atomic_bytes_write(self.frame_path, encoded.tobytes())
        _atomic_json_write(self.context_path, context)
        if dump_raw_response:
            _atomic_bytes_write(
                self.raw_response_path, result.raw_response.encode("utf-8")
            )
        status = self._write_status(
            MappingStatus(
                status="ready_unreviewed",
                source_fingerprint=source_fingerprint(source),
                mapped_at=utc_now_iso(),
            )
        )
        return ContextResolution(context, status, "mapper", True)

    def save_unreviewed(
        self,
        context: dict[str, Any],
        source: int | str | Path,
        provenance: str = "manual_edit",
    ) -> ContextResolution:
        prepared = self._prepare_context(context, source)
        previous = self.load_status()
        _atomic_json_write(self.context_path, prepared)
        status = self._write_status(
            MappingStatus(
                status="ready_unreviewed",
                source_fingerprint=source_fingerprint(source),
                mapped_at=previous.mapped_at or utc_now_iso(),
            )
        )
        return ContextResolution(prepared, status, provenance, True)

    def approve(
        self,
        context: dict[str, Any],
        reviewer: str,
        source: int | str | Path,
    ) -> ContextResolution:
        prepared = self._prepare_context(context, source)
        previous = self.load_status()
        _atomic_json_write(self.context_path, prepared)
        status = self._write_status(
            MappingStatus(
                status="ready_reviewed",
                source_fingerprint=source_fingerprint(source),
                mapped_at=previous.mapped_at or utc_now_iso(),
                reviewed_at=utc_now_iso(),
                reviewed_by=str(reviewer).strip(),
            )
        )
        return ContextResolution(prepared, status, "manual", True)

    def resolve(
        self,
        source: int | str | Path,
        policy: str,
        manual_context: dict[str, Any] | None = None,
        legacy_context_path: str | Path | None = None,
    ) -> ContextResolution:
        normalized_policy = normalize_scene_context_policy(policy)
        if manual_context is not None:
            prepared = self._prepare_context(manual_context, source)
            status = MappingStatus(
                status="ready_reviewed",
                source_fingerprint=source_fingerprint(source),
                mapped_at=prepared["generated_at"],
                reviewed_at=prepared["generated_at"],
                reviewed_by="site_config",
            )
            return ContextResolution(prepared, status, "manual", True)

        context = self._load_context()
        status = self.load_status()
        current_fingerprint = source_fingerprint(source)
        if context is not None:
            if status.source_fingerprint and status.source_fingerprint != current_fingerprint:
                stale = self.mark_stale(source)
                return ContextResolution(context, stale, "cache", False)
            if status.status == "ready_reviewed":
                return ContextResolution(context, status, "cache", True)
            if status.status == "ready_unreviewed":
                return ContextResolution(
                    context, status, "cache", normalized_policy == "auto"
                )
            return ContextResolution(context, status, "cache", False)

        if legacy_context_path is not None:
            try:
                legacy = json.loads(Path(legacy_context_path).read_text())
            except (OSError, ValueError, TypeError):
                legacy = None
            if isinstance(legacy, dict) and legacy.get("scene_description"):
                imported = self.save_unreviewed(legacy, source, provenance="legacy")
                return ContextResolution(
                    imported.context,
                    imported.status,
                    "legacy",
                    normalized_policy == "auto",
                )

        pending = MappingStatus(
            status="pending",
            source_fingerprint=current_fingerprint,
            error=(
                "explicit scene context is required"
                if normalized_policy == "manual"
                else "scene context has not been mapped"
            ),
        )
        return ContextResolution(None, pending, "none", False)
