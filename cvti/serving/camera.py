"""Per-camera state for the multi-stream pipeline.

The detector model is shared and batched across all cameras (stateless). The
STATEFUL work is per camera and lives here: each camera keeps its own ByteTrack
tracker, zone monitor, rules engine, and scene context. Detections for a camera
are associated to that camera's tracks, turned into RawEvents, evaluated against
that camera's threat policy, and emitted as QueuedAlerts for the gate.

Concealment/violence (which need the pose model) are a documented seam: add a
batched pose pass in the pipeline and feed pose_people here the same way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cvti.event_adapters import zone_states_to_events
from cvti.rules.customization import CustomizationEngine
from cvti.serving.alert_queue import QueuedAlert


def _to_queued(camera_id: str, alert: Any, timestamp: float, zone: str | None,
               frames: list, scene: dict | None) -> QueuedAlert:
    # Evidence frames are captured NOW because the async gate verifies later,
    # by which point the live frame is gone.
    return QueuedAlert(
        camera_id=camera_id,
        rule_name=alert.rule_name,
        priority=alert.priority,
        title=alert.title,
        timestamp=timestamp,
        track_id=alert.person_id,
        zone=zone,
        object_label=alert.object_label,
        payload={"candidate": alert, "frames": frames, "scene": scene},
    )


@dataclass
class PerCameraState:
    camera_id: str
    engine: CustomizationEngine
    zone_monitor: Any = None          # RetailZoneMonitor | None
    scene_context: dict | None = None
    person_filter: bool = True
    _tracker: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import warnings
        import supervision as sv
        # sv.ByteTrack is deprecation-proxied in supervision 0.28 (removed in
        # 0.30). It still works; silence the per-camera warning spam for now.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack()

    def process(self, detections: Any, image: Any, timestamp: float) -> list[QueuedAlert]:
        """Associate detections to this camera's tracks, run zones + rules,
        and return the candidate alerts (already mapped to QueuedAlert)."""
        from cvti.retail.zones import filter_person_detections

        frame_hw = image.shape[:2]
        if self.person_filter and self.zone_monitor is not None:
            detections = filter_person_detections(detections, frame_hw)
        tracked = self._tracker.update_with_detections(detections)

        raw_events = []
        zone_by_event: list[str | None] = []
        if self.zone_monitor is not None:
            states = self.zone_monitor.update(tracked, timestamp)
            zone_events = zone_states_to_events(states, timestamp=timestamp)
            raw_events += zone_events
            zone_by_event = [e.extra.get("zone") for e in zone_events]

        if not raw_events:
            return []

        alerts = self.engine.evaluate(raw_events, scene_context=self.scene_context)
        # Best-effort: attach the zone of the first matching presence event for dedup.
        zone_hint = zone_by_event[0] if zone_by_event else None
        return [_to_queued(self.camera_id, a, timestamp, zone_hint, [image], self.scene_context)
                for a in alerts]


def build_camera_states(site_config: dict) -> dict[str, dict]:
    """Parse a site config into {camera_id: {"source": ..., "state": PerCameraState}}.

    Site config shape:
        {"cameras": [{"id", "source", "config", "zones"?, "scene_description"?}]}
    """
    from cvti.retail.zones import RetailZoneMonitor, load_zone_config

    out: dict[str, dict] = {}
    for cam in site_config["cameras"]:
        cam_id = cam["id"]
        engine = CustomizationEngine(cam["config"])
        zone_monitor = None
        if cam.get("zones"):
            zone_monitor = RetailZoneMonitor(load_zone_config(cam["zones"]))
        scene = None
        if cam.get("scene_description"):
            scene = {"environment_type": cam.get("environment_type", "unknown"),
                     "scene_description": cam["scene_description"]}
        out[cam_id] = {
            "source": cam["source"],
            "state": PerCameraState(cam_id, engine, zone_monitor=zone_monitor, scene_context=scene),
        }
    return out


def load_site_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
