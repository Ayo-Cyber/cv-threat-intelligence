"""Production serving adapter for canonical per-camera scene mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cvti.logging_setup import get_logger
from cvti.scene.agent_mapper import AgentMapper
from cvti.scene.context_store import (
    ContextResolution,
    MappingStatus,
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


def _operator_hints(camera: dict[str, Any]) -> dict[str, Any] | None:
    """The operator's onboarding answers, when they gave any.

    Distinct from _manual_context on purpose: a full scene_description is
    human-AUTHORED context (mapper skipped, auto-approved), while these are
    human HINTS — priors the mapper refines instead of guessing from one
    frame. Collected by the connect-your-cameras wizard; the discord this
    heals (co-engineer, 1 Sep) was onboarding knowledge the backend never
    saw."""
    environment = str(camera.get("environment_type") or "").strip()
    actors = [str(a).strip() for a in camera.get("expected_actors") or []]
    actors = [a for a in actors if a]
    note = str(camera.get("scene_hint") or "").strip()
    hints: dict[str, Any] = {}
    site_type = str(camera.get("site_type") or "").strip()
    area_id = str(camera.get("area_id") or "").strip()
    area_type = str(camera.get("area_type") or "").strip()
    if site_type and site_type != "unknown":
        hints["site_type"] = site_type
    if area_id:
        hints["area_id"] = area_id
    if area_type and area_type != "unknown":
        hints["area_type"] = area_type
    if environment and environment != "unknown":
        hints["environment_type"] = environment
    if actors:
        hints["expected_actors"] = actors
    if note:
        hints["note"] = note
    return hints or None


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

    def inspect(
        self, cameras: list[dict[str, Any]], policy: str = "auto"
    ) -> SceneMappingPreflight:
        """Resolve existing/manual context without performing VLM inference."""
        normalized_policy = normalize_scene_context_policy(policy)
        preflight = SceneMappingPreflight()
        for camera in cameras:
            camera_id = str(camera.get("id", "?"))
            try:
                source = camera["source"]
                store = SceneContextStore(self.context_root, camera_id)
                manual = _manual_context(camera)
                resolution = store.resolve(
                    source,
                    normalized_policy,
                    manual_context=manual,
                    legacy_context_path=(
                        self.legacy_root / camera_id / "scene_context.json"),
                )
                if manual is not None and resolution.context is not None:
                    resolution = store.approve(
                        resolution.context, "site_config", source
                    )
            except Exception as exc:  # noqa: BLE001 — same guarantee as
                # prepare(): inspect() runs at engine startup, and a malformed
                # hand-edited scene_context in site.json used to raise out of
                # approve() and kill the engine — the exact #67 bug class,
                # reopened through a different door. Degrade to failed, loud.
                log.exception("[agent-map] inspect failed for %s", camera_id)
                resolution = ContextResolution(
                    context=None,
                    status=MappingStatus(status="failed", error=str(exc)),
                    provenance="inspect_error", usable=False)
            if resolution.context is not None:
                preflight.contexts[camera_id] = resolution.context
            if not resolution.usable:
                preflight.blocked_camera_ids.add(camera_id)
            preflight.statuses[camera_id] = _status_document(
                resolution, normalized_policy
            )
        return preflight

    def prepare(
        self, cameras: list[dict[str, Any]], policy: str = "auto"
    ) -> SceneMappingPreflight:
        normalized_policy = normalize_scene_context_policy(policy)
        preflight = SceneMappingPreflight()
        for camera in cameras:
            try:
                result = self._prepare_camera(camera, normalized_policy)
            except Exception as exc:  # noqa: BLE001 — the mapper must never
                # take down monitoring. The store's id validation raised here
                # and killed the WHOLE engine before the first frame (field
                # report, 31 Aug: crash-loop on every feed with a
                # human-named camera). Any preflight failure degrades to a
                # failed mapping status: policy auto runs the camera generic
                # and loud, the strict policies block that camera only.
                camera_id = str(camera.get("id", "?"))
                log.exception("[agent-map] preflight failed for %s", camera_id)
                failed = MappingStatus(status="failed", error=str(exc))
                result = CameraMappingResult(
                    camera_id=camera_id,
                    resolution=ContextResolution(
                        context=None, status=failed,
                        provenance="preflight_error", usable=False),
                    mapped=False,
                )
            camera_id = result.camera_id
            resolution = result.resolution
            if resolution.usable and resolution.context is not None:
                preflight.contexts[camera_id] = resolution.context
            elif normalized_policy == "auto":
                # FAIL VISIBLE, NEVER CLOSED. Under the default policy a
                # mapper failure used to BLOCK the camera — the E2E harness
                # caught the consequence before any customer did (30 Aug):
                # one missing prompt file and the engine started with ZERO
                # cameras watching. A security product must never trade
                # 'monitoring without scene context' for 'no monitoring at
                # all' by default. The camera runs generic; the failure
                # stays loud in the mapping health rows. The strict policies
                # (require_reviewed / manual) still block — there the
                # operator explicitly chose certainty over coverage.
                log.warning("[agent-map] %s has no usable scene context — "
                            "starting WITHOUT it (policy=auto); reason: %s",
                            camera_id, resolution.status.error or resolution.status.status)
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
                operator_hints=_operator_hints(camera),
            )
            saved = store.save_mapping(
                mapped, source, dump_raw_response=self.dump_raw_response
            )
            if policy == "require_reviewed" and saved.status.status != "ready_reviewed":
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
