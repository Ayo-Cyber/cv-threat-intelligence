"""Production serving adapter for canonical per-camera scene mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cvti.logging_setup import get_logger
from cvti.scene.agent_mapper import AgentMapper
from cvti.scene.context_store import (
    ContextResolution,
    SceneContextStore,
    normalize_scene_context_policy,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class CameraMappingResult:
    camera_id: str
    resolution: ContextResolution
    mapped: bool


@dataclass
class SceneMappingPreflight:
    contexts: dict[str, dict[str, Any]] = field(default_factory=dict)
    statuses: dict[str, dict[str, Any]] = field(default_factory=dict)
    blocked_camera_ids: set[str] = field(default_factory=set)


def _manual_context(camera: dict[str, Any]) -> dict[str, Any] | None:
    description = str(camera.get("scene_description", "")).strip()
    if not description:
        return None
    return {
        "camera_id": str(camera["id"]),
        "source_type": "video_file",
        "environment_type": camera.get("environment_type", "unknown"),
        "scene_description": description,
        "expected_actors": list(camera.get("expected_actors") or []),
        "zones": [],
        "confidence": 1.0,
        "generated_at": str(
            camera.get("scene_context_updated_at") or "2026-01-01T00:00:00Z"
        ),
        "source_frame_path": "site-config",
        "notes": "Human-authored site configuration.",
    }


def _status_document(
    resolution: ContextResolution, policy: str
) -> dict[str, Any]:
    document: dict[str, Any] = resolution.status.to_dict()
    document.update(
        {
            "provenance": resolution.provenance,
            "usable": resolution.usable,
            "review_required": policy == "require_reviewed",
            "environment_type": (
                resolution.context.get("environment_type", "unknown")
                if resolution.context
                else "unknown"
            ),
        }
    )
    return document


class FullAgentMapperService:
    def __init__(
        self,
        output_dir: str | Path,
        mapper: AgentMapper,
        dump_raw_response: bool = False,
        legacy_root: str | Path = Path("runs/context"),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.context_root = self.output_dir / "context"
        self.mapper = mapper
        self.dump_raw_response = dump_raw_response
        self.legacy_root = Path(legacy_root)

    def prepare(
        self, cameras: list[dict[str, Any]], policy: str = "auto"
    ) -> SceneMappingPreflight:
        normalized_policy = normalize_scene_context_policy(policy)
        preflight = SceneMappingPreflight()
        for camera in cameras:
            result = self._prepare_camera(camera, normalized_policy)
            camera_id = result.camera_id
            resolution = result.resolution
            if resolution.usable and resolution.context is not None:
                preflight.contexts[camera_id] = resolution.context
            else:
                preflight.blocked_camera_ids.add(camera_id)
            preflight.statuses[camera_id] = _status_document(
                resolution, normalized_policy
            )
        return preflight

    def _prepare_camera(
        self, camera: dict[str, Any], policy: str
    ) -> CameraMappingResult:
        camera_id = str(camera["id"])
        source = camera["source"]
        store = SceneContextStore(self.context_root, camera_id)
        manual = _manual_context(camera)
        legacy_path = self.legacy_root / camera_id / "scene_context.json"
        resolution = store.resolve(
            source,
            policy,
            manual_context=manual,
            legacy_context_path=legacy_path,
        )

        if manual is not None and resolution.context is not None:
            persisted = store.approve(resolution.context, "site_config", source)
            return CameraMappingResult(camera_id, persisted, False)
        if resolution.usable:
            return CameraMappingResult(camera_id, resolution, False)
        if resolution.status.status == "ready_unreviewed":
            return CameraMappingResult(camera_id, resolution, False)
        if policy == "manual":
            status = store.mark_failed(source, "explicit scene context is required")
            blocked = ContextResolution(None, status, "none", False)
            return CameraMappingResult(camera_id, blocked, False)

        try:
            store.mark_pending(source)
            mapped = self.mapper.map_result(
                source,
                camera_id,
                sample_count=3,
                source_frame_path=str(store.frame_path),
            )
            saved = store.save_mapping(
                mapped, source, dump_raw_response=self.dump_raw_response
            )
            if policy == "require_reviewed":
                saved = ContextResolution(
                    saved.context, saved.status, saved.provenance, False
                )
            log.info(
                "[agent-map] %s: %s (%s)",
                camera_id,
                saved.context.get("environment_type", "unknown")
                if saved.context
                else "unknown",
                saved.status.status,
            )
            return CameraMappingResult(camera_id, saved, True)
        except Exception as exc:  # noqa: BLE001 - one camera must not block the site
            status = store.mark_failed(source, str(exc))
            log.warning("[agent-map] %s failed: %s", camera_id, str(exc)[:120])
            failed = ContextResolution(None, status, "mapper", False)
            return CameraMappingResult(camera_id, failed, True)
