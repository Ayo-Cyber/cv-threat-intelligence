"""Per-camera state for the multi-stream pipeline.

The detector model is shared and batched across all cameras (stateless). The
STATEFUL work is per camera and lives here: each camera keeps its own ByteTrack
tracker, zone monitor, rules engine, and scene context. Detections for a camera
are associated to that camera's tracks, turned into RawEvents, evaluated against
that camera's threat policy, and emitted as QueuedAlerts for the gate.

Pose-based CONCEALMENT (theft) is wired in via a shared pose model (opt-in per
camera with "concealment": true). Still on the seam for a later pass: violence
(needs weapon validation + the temporal gate), the theft state machine, and the
video-action model per camera — all of which the single-stream detector runs.
"""
from __future__ import annotations

import json
from collections import deque
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
    # Pose-based concealment (theft) — needs a SHARED pose model instance. The
    # tracker/zones are the always-on path; concealment is opt-in per camera.
    pose_model: Any = None            # LoadedModel | None (shared across cameras)
    concealment: bool = False
    pose_conf: float = 0.35
    imgsz: int = 640
    _tracker: Any = field(default=None, init=False, repr=False)
    _conceal: Any = field(default=None, init=False, repr=False)
    _prev_pose: list = field(default_factory=list, init=False, repr=False)
    _next_pose_id: int = field(default=1, init=False, repr=False)
    _pose_history: dict = field(default_factory=dict, init=False, repr=False)
    # Rolling recent frames (~2s at 5 FPS) so the gate gets per-rule evidence
    # (motion-peak span for violence, sharpest single frame for weapons) instead
    # of just the flagged frame.
    _frame_buffer: deque = field(default_factory=lambda: deque(maxlen=10), init=False, repr=False)

    def __post_init__(self) -> None:
        import warnings
        import supervision as sv
        # sv.ByteTrack is deprecation-proxied in supervision 0.28 (removed in
        # 0.30). It still works; silence the per-camera warning spam for now.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack()
        if self.concealment and self.pose_model is not None:
            from cvti.retail.concealment import ConcealmentDetector
            self._conceal = ConcealmentDetector()

    def _concealment_events(self, image: Any, timestamp: float) -> list:
        """Run the shared pose model on this camera's frame and produce
        concealment RawEvents (pocket/waist/bag). Per-camera pose track state is
        kept here so dwell/motion history is not mixed across cameras."""
        from cvti.detector.core import (
            assign_pose_tracks, enrich_pose_people_with_history,
            extract_pose_people, pose_people_to_concealment_frames,
        )
        from cvti.event_adapters import concealment_to_events

        pose_people = extract_pose_people(self.pose_model, image, self.pose_conf, self.imgsz)
        pose_people, self._next_pose_id = assign_pose_tracks(
            pose_people, previous_people=self._prev_pose, next_track_id=self._next_pose_id)
        pose_people = enrich_pose_people_with_history(pose_people, self._pose_history)
        self._prev_pose = list(pose_people)
        assessments = self._conceal.update(
            pose_people_to_concealment_frames(pose_people, timestamp), timestamp)
        return concealment_to_events(assessments, timestamp)

    def process(self, detections: Any, image: Any, timestamp: float) -> list[QueuedAlert]:
        """Associate detections to this camera's tracks, run zones (+ optional
        pose-based concealment) + rules, and return candidate alerts."""
        from cvti.retail.zones import filter_person_detections

        frame_hw = image.shape[:2]
        self._frame_buffer.append(image)
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

        if self._conceal is not None:
            try:
                raw_events += self._concealment_events(image, timestamp)
            except Exception as exc:  # noqa: BLE001 - a pose hiccup must not kill the camera
                print(f"[{self.camera_id}] concealment error: {str(exc)[:120]}")

        if not raw_events:
            return []

        alerts = self.engine.evaluate(raw_events, scene_context=self.scene_context)
        if not alerts:
            return []
        from cvti.verification.frame_select import select_evidence_frames

        # Best-effort: attach the zone of the first matching presence event for dedup.
        zone_hint = zone_by_event[0] if zone_by_event else None
        recent = list(self._frame_buffer)
        out = []
        for a in alerts:
            frames, _ = select_evidence_frames(recent, a.rule_name)
            out.append(_to_queued(self.camera_id, a, timestamp, zone_hint,
                                  frames or [image], self.scene_context))
        return out


def build_camera_states(site_config: dict, *, pose_model: Any = None,
                        baseline_config: str | None = None) -> dict[str, dict]:
    """Parse a site config into {camera_id: {"source": ..., "state": PerCameraState}}.

    Site config shape:
        {"cameras": [{"id", "source", "config", "zones"?, "concealment"?,
                      "scene_description"?}]}

    `pose_model` is a shared LoadedModel; cameras with "concealment": true reuse
    it for pose-based theft detection. `baseline_config` (if given) is merged into
    every camera's engine so the always-on critical rules apply per camera too.
    """
    from cvti.retail.zones import RetailZoneMonitor, load_zone_config

    out: dict[str, dict] = {}
    for cam in site_config["cameras"]:
        cam_id = cam["id"]
        engine = CustomizationEngine(cam["config"], baseline_path=baseline_config)
        zone_monitor = None
        if cam.get("zones"):
            zone_monitor = RetailZoneMonitor(load_zone_config(cam["zones"]))
        scene = None
        if cam.get("scene_description"):
            scene = {"environment_type": cam.get("environment_type", "unknown"),
                     "scene_description": cam["scene_description"]}
        out[cam_id] = {
            "source": cam["source"],
            "state": PerCameraState(
                cam_id, engine, zone_monitor=zone_monitor, scene_context=scene,
                pose_model=pose_model, concealment=bool(cam.get("concealment")),
            ),
        }
    return out


def load_site_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
