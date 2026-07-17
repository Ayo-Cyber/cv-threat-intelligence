"""Per-camera state for the multi-stream pipeline.

The detector model is shared and batched across all cameras (stateless). The
STATEFUL work is per camera and lives here: each camera keeps its own ByteTrack
tracker, pose-track state, violence/theft detectors, zone monitor, rules engine,
and scene context. Detections for a camera are associated to that camera's
tracks, turned into RawEvents, evaluated against that camera's threat policy,
and emitted as QueuedAlerts for the gate.

This runs the full single-stream detector's signals per camera (opt-in per
camera via the site config): zones, pose-based concealment, violence, weapons
(needs a shared weapon model), and the theft state machine. It reuses the exact
`cvti/detector/core.py` functions so behaviour matches single-stream. The one
piece still on the seam is the video-action model per camera.
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

# Tuned defaults mirrored from cvti/detector/core.py argparse so the multi-stream
# path behaves like single-stream. Keep in sync if those defaults change.
_VIOLENCE_DISTANCE_RATIO = 1.1
_VIOLENCE_WRIST_SPEED = 120.0
_VIOLENCE_ARM_EXTENSION_RATIO = 0.35
_VIOLENCE_WRIST_ACCEL = 800.0
_WEAPON_HAND_DISTANCE_RATIO = 0.20
_ASSAULT_DISTANCE_RATIO = 1.2
_WEAPON_MIN_AREA_RATIO = 0.002
_WEAPON_MAX_AREA_RATIO = 0.18
_WEAPON_BORDER_MARGIN_RATIO = 0.03
_MIN_THREAT_FRAMES = 3


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
    # Shared (stateless) models, injected by the pipeline.
    pose_model: Any = None            # LoadedModel | None — needed by concealment/violence/theft
    weapon_model: Any = None          # LoadedModel | None — needed by weapons
    # Per-camera opt-in signals (from the site config).
    concealment: bool = False
    violence: bool = False
    weapons: bool = False
    theft: bool = False
    pose_conf: float = 0.35
    weapon_conf: float = 0.40
    imgsz: int = 640
    # --- per-camera stateful bits (not constructor args) ---
    _tracker: Any = field(default=None, init=False, repr=False)
    _conceal: Any = field(default=None, init=False, repr=False)
    _violence_gate: Any = field(default=None, init=False, repr=False)
    _theft: Any = field(default=None, init=False, repr=False)
    _prev_pose: list = field(default_factory=list, init=False, repr=False)
    _next_pose_id: int = field(default=1, init=False, repr=False)
    _pose_history: dict = field(default_factory=dict, init=False, repr=False)
    _object_threat_frames: int = field(default=0, init=False, repr=False)
    _weapon_classes: Any = field(default=None, init=False, repr=False)
    _person_classes: Any = field(default=None, init=False, repr=False)
    # Rolling recent frames (~2s at 5 FPS) so the gate gets per-rule evidence
    # (motion-peak span for violence, sharpest single frame for weapons).
    _frame_buffer: deque = field(default_factory=lambda: deque(maxlen=10), init=False, repr=False)

    def __post_init__(self) -> None:
        import warnings
        import supervision as sv
        from cvti.detector.core import normalize_threat_classes
        # sv.ByteTrack is deprecation-proxied in supervision 0.28 (removed in
        # 0.30). It still works; silence the per-camera warning spam for now.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack()
        self._weapon_classes = normalize_threat_classes("gun,knife")
        self._person_classes = normalize_threat_classes("person")
        if self.concealment and self.pose_model is not None:
            from cvti.retail.concealment import ConcealmentDetector
            self._conceal = ConcealmentDetector()
        if self.violence and self.pose_model is not None:
            from cvti.detector.core import ViolenceTemporalGate
            self._violence_gate = ViolenceTemporalGate()
        if self.theft and self.pose_model is not None:
            from cvti.detector.core import TheftDetector
            self._theft = TheftDetector()

    def _needs_pose(self) -> bool:
        return self.pose_model is not None and (self.concealment or self.violence or self.theft)

    def _compute_pose(self, image: Any, timestamp: float) -> list:
        """Shared pose model on this camera's frame + per-camera track state, so
        wrist speed/dwell history is not mixed across cameras."""
        from cvti.detector.core import (
            assign_pose_tracks, enrich_pose_people_with_history, extract_pose_people,
        )
        pose_people = extract_pose_people(self.pose_model, image, self.pose_conf, self.imgsz)
        pose_people, self._next_pose_id = assign_pose_tracks(
            pose_people, previous_people=self._prev_pose, next_track_id=self._next_pose_id)
        pose_people = enrich_pose_people_with_history(pose_people, self._pose_history)
        self._prev_pose = list(pose_people)
        return pose_people

    def _merged_detections(self, object_detections: list | None, image: Any) -> list:
        """Shared object detections + (optional) per-camera weapon-model detections."""
        merged = list(object_detections or [])
        if self.weapon_model is not None:
            from cvti.detector.core import merge_detections, predict_with_model
            weap = predict_with_model(self.weapon_model, image, self.weapon_conf, self.imgsz,
                                      self._weapon_classes, source_model="weapon")
            merged = merge_detections(merged, weap)
        return merged

    def _assessment_events(self, pose_people: list, merged: list, image: Any,
                           timestamp: float) -> list:
        """Run weapons / violence / theft assessments and adapt to RawEvents,
        reusing the single-stream core.py logic."""
        from cvti.detector.core import (
            ThreatAssessment, assess_threat, assess_violence, gate_assessment,
            validate_weapon_detections,
        )
        from cvti.event_adapters import assessments_to_events

        validated_weapons: list = []
        if (self.violence or self.weapons):
            validated_weapons = validate_weapon_detections(
                detections=merged, weapon_classes=self._weapon_classes,
                person_classes=self._person_classes, pose_people=pose_people,
                frame_shape=image.shape, weapon_min_area_ratio=_WEAPON_MIN_AREA_RATIO,
                weapon_max_area_ratio=_WEAPON_MAX_AREA_RATIO,
                weapon_border_margin_ratio=_WEAPON_BORDER_MARGIN_RATIO,
                weapon_hand_distance_ratio=_WEAPON_HAND_DISTANCE_RATIO,
                allow_unattached_weapons=False)

        object_assessment = violence_assessment = theft_assessment = None
        if self.weapons:
            raw = assess_threat(merged, self._weapon_classes, self._person_classes,
                                validated_weapons, _ASSAULT_DISTANCE_RATIO)
            self._object_threat_frames = self._object_threat_frames + 1 if raw.active else 0
            object_assessment = gate_assessment(raw, self._object_threat_frames, _MIN_THREAT_FRAMES)
        if self.violence:
            raw_v = assess_violence(
                pose_people=pose_people, validated_weapon_detections=validated_weapons,
                violence_distance_ratio=_VIOLENCE_DISTANCE_RATIO,
                violence_wrist_speed=_VIOLENCE_WRIST_SPEED,
                violence_arm_extension_ratio=_VIOLENCE_ARM_EXTENSION_RATIO,
                weapon_hand_distance_ratio=_WEAPON_HAND_DISTANCE_RATIO,
                violence_wrist_accel=_VIOLENCE_WRIST_ACCEL)
            violence_assessment = self._violence_gate.update(raw_v)
        if self.theft:
            theft_assessment = self._theft.update(pose_people, merged, timestamp)

        if object_assessment is None and violence_assessment is None and theft_assessment is None:
            return []
        return assessments_to_events(object_assessment, violence_assessment, theft_assessment,
                                     timestamp=timestamp, theft_detector=self._theft)

    def process(self, detections: Any, image: Any, timestamp: float,
                object_detections: list | None = None) -> list[QueuedAlert]:
        """Track + run all enabled signals (zones, concealment, violence, weapons,
        theft) + rules; return candidate alerts with per-rule evidence frames.

        `detections` is sv.Detections (tracking/zones); `object_detections` is the
        core.py Detection list from the same frame (weapons/violence/theft)."""
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

        try:
            pose_people = self._compute_pose(image, timestamp) if self._needs_pose() else []
            if self._conceal is not None:
                from cvti.detector.core import pose_people_to_concealment_frames
                from cvti.event_adapters import concealment_to_events
                assessments = self._conceal.update(
                    pose_people_to_concealment_frames(pose_people, timestamp), timestamp)
                raw_events += concealment_to_events(assessments, timestamp)
            if self.violence or self.weapons or self.theft:
                merged = self._merged_detections(object_detections, image)
                raw_events += self._assessment_events(pose_people, merged, image, timestamp)
        except Exception as exc:  # noqa: BLE001 - a detector hiccup must not kill the camera
            print(f"[{self.camera_id}] detector error: {str(exc)[:140]}")

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


def build_camera_states(site_config: dict, *, pose_model: Any = None, weapon_model: Any = None,
                        baseline_config: str | None = None) -> dict[str, dict]:
    """Parse a site config into {camera_id: {"source": ..., "state": PerCameraState}}.

    Site config per-camera keys: id, source, config, plus optional zones,
    scene_description, and the signal toggles concealment / violence / weapons /
    theft (all default false). `pose_model` / `weapon_model` are shared instances;
    cameras that enable a pose/weapon signal reuse them. `baseline_config` (if
    given) is merged into every camera's engine (always-on critical rules).
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
                pose_model=pose_model, weapon_model=weapon_model,
                concealment=bool(cam.get("concealment")), violence=bool(cam.get("violence")),
                weapons=bool(cam.get("weapons")), theft=bool(cam.get("theft")),
            ),
        }
    return out


def load_site_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
