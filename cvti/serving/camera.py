"""Per-camera state for the multi-stream pipeline.

The detector model is shared and batched across all cameras (stateless). The
STATEFUL work is per camera and lives here: each camera keeps its own ByteTrack
tracker, pose-track state, violence/theft detectors, zone monitor, rules engine,
and scene context. Detections for a camera are associated to that camera's
tracks, turned into RawEvents, evaluated against that camera's threat policy,
and emitted as QueuedAlerts for the gate.

This runs the FULL single-stream detector per camera (opt-in via the site
config): zones, pose-based concealment, violence, weapons (needs a shared weapon
model), the theft state machine, and the fine-tuned video-action model (gated,
shared instance). It reuses the exact `cvti/detector/core.py` functions so
behaviour matches single-stream. Shared stateless models (object/pose/weapon/
video) are injected by the pipeline; everything stateful is per camera.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from cvti.contracts import RawEvent
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


def _person_boxes(tracked: Any) -> list:
    """Extract (track_id, x1, y1, x2, y2) person boxes from an sv.Detections.

    COCO person class is 0; if no class_id is present (already person-filtered) we
    keep every box. track_id falls back to the row index when the tracker hasn't
    assigned one yet."""
    boxes: list = []
    xyxy = getattr(tracked, "xyxy", None)
    if xyxy is None:
        return boxes
    cls = getattr(tracked, "class_id", None)
    tids = getattr(tracked, "tracker_id", None)
    for i in range(len(xyxy)):
        if cls is not None and cls[i] is not None and int(cls[i]) != 0:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
        tid = int(tids[i]) if tids is not None and tids[i] is not None else i
        boxes.append((tid, x1, y1, x2, y2))
    return boxes


def _to_queued(camera_id: str, alert: Any, timestamp: float, zone: str | None,
               frames: list, scene: dict | None,
               clip_frames: list | None = None, clip_fps: float = 0.0,
               bbox: tuple | None = None) -> QueuedAlert:
    # Evidence frames are captured NOW because the async gate verifies later,
    # by which point the live frame is gone.
    #  * `frames`      — a few sharp stills, for the VLM gate + thumbnails.
    #  * `clip_frames` — the continuous JPEG-encoded window (~last N seconds) so the
    #                    sink can write a REAL video of the event, not a slideshow.
    return QueuedAlert(
        camera_id=camera_id,
        rule_name=alert.rule_name,
        priority=alert.priority,
        title=alert.title,
        timestamp=timestamp,
        track_id=alert.person_id,
        zone=zone,
        object_label=alert.object_label,
        payload={"candidate": alert, "frames": frames, "scene": scene,
                 "clip_frames": clip_frames or [], "clip_fps": clip_fps,
                 "enqueued_at": time.time(),    # wall-clock, for verify-latency
                 # where the subject was when this fired, so evidence can point at
                 # WHO — an alert with no box makes the operator hunt the frame.
                 "bbox": bbox},
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
    tamper: bool = False              # camera block/tamper detection (pure CV)
    fall: bool = False                # person collapsed / on the ground (person boxes)
    # HSE situational detectors (cvti.detector.situational) — cheap candidate
    # generators the VLM gate then confirms.
    fire_smoke: bool = False          # fire / smoke visual candidate
    running: bool = False             # sustained fast person movement (panic)
    crowd_formation: bool = False     # tight group formation
    running_min_speed_ratio: float = 0.18
    running_min_frames: int = 3
    crowd_min_people: int = 4
    crowd_min_frames: int = 3
    crowd_max_cluster_ratio: float = 0.24
    fire_min_frames: int = 3
    fire_min_hot_area_ratio: float = 0.012
    video_action: bool = False
    video_action_model: Any = None    # shared VideoMAEActionModel instance
    pose_conf: float = 0.35
    weapon_conf: float = 0.40
    imgsz: int = 640
    va_fps: float = 5.0
    va_window_seconds: float = 4.0
    va_frames: int = 16
    va_cooldown: float = 2.0
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
    _video_runtime: Any = field(default=None, init=False, repr=False)
    _va_index: int = field(default=0, init=False, repr=False)
    _tamper_det: Any = field(default=None, init=False, repr=False)
    _fall_det: Any = field(default=None, init=False, repr=False)
    _fire_det: Any = field(default=None, init=False, repr=False)
    _running_det: Any = field(default=None, init=False, repr=False)
    _crowd_det: Any = field(default=None, init=False, repr=False)
    # Rolling recent frames (~2s at 5 FPS) so the gate gets per-rule evidence
    # (motion-peak span for violence, sharpest single frame for weapons).
    _frame_buffer: deque = field(default_factory=lambda: deque(maxlen=10), init=False, repr=False)
    # Longer CONTINUOUS window, JPEG-encoded (kept light for RAM), so a confirmed
    # alert can be replayed as a real video of the event lead-up, not a slideshow.
    # ~48 frames ≈ 8-12s at typical pipeline fps. Holds (timestamp, jpeg_bytes).
    _clip_buffer: deque = field(default_factory=lambda: deque(maxlen=48), init=False, repr=False)

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
        if self.video_action and self.video_action_model is not None:
            from cvti.video_action_runtime import VideoActionRuntime
            self._video_runtime = VideoActionRuntime(
                model=self.video_action_model, backend="videomae",
                model_name=getattr(self.video_action_model, "model_name", "videomae"),
                fps=self.va_fps, window_seconds=self.va_window_seconds,
                frame_count=self.va_frames, cooldown_seconds=self.va_cooldown)

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
        # Continuous replay buffer: encode this frame to JPEG (cheap, ~1ms) and keep
        # a rolling window with timestamps so a confirmed alert replays as real video.
        ok_enc, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok_enc:
            self._clip_buffer.append((timestamp, enc.tobytes()))

        # Camera tamper/block runs on the raw frame — independent of any person,
        # since a covered camera shows nothing. Cheap CV, every frame.
        raw_events: list = []
        if self.tamper:
            if self._tamper_det is None:
                from cvti.detector.tamper import TamperDetector
                self._tamper_det = TamperDetector()
            t = self._tamper_det.update(image)
            if t is not None:
                raw_events.append(RawEvent(
                    detector="camera_tampering", active=True,
                    title=f"CAMERA BLOCKED ({t['kind']})", level="high",
                    timestamp=timestamp, extra=t))

        # Fire + smoke — pure CV pre-filter on the raw frame (no person needed).
        if self.fire_smoke:
            if self._fire_det is None:
                from cvti.detector.situational import FireSmokeCandidateDetector
                self._fire_det = FireSmokeCandidateDetector(
                    min_frames=self.fire_min_frames,
                    min_hot_area_ratio=self.fire_min_hot_area_ratio)
            f = self._fire_det.update(image, timestamp)
            if f is not None:
                raw_events.append(RawEvent(
                    detector="fire", active=True, title="POSSIBLE FIRE OR SMOKE",
                    level="critical", timestamp=timestamp, extra=f))

        if self._video_runtime is not None:
            self._video_runtime.add_frame(image, frame_index=self._va_index)
            self._va_index += 1
        if self.person_filter and self.zone_monitor is not None:
            detections = filter_person_detections(detections, frame_hw)
        tracked = self._tracker.update_with_detections(detections)

        # Track -> box for EVERY frame (cheap): lets an alert record where its
        # subject was, so evidence can point at who rather than at a whole frame.
        self._box_by_track = {tid: (int(x1), int(y1), int(x2), int(y2))
                              for (tid, x1, y1, x2, y2) in _person_boxes(tracked)}

        # Fall / panic-running / crowd-formation work off tracked PERSON boxes.
        if self.fall or self.running or self.crowd_formation:
            person_boxes = _person_boxes(tracked)   # [(tid, x1, y1, x2, y2), ...]
            frame_area = float(frame_hw[0] * frame_hw[1])
            people = [{"track_id": tid, "bbox": (int(x1), int(y1), int(x2), int(y2))}
                      for (tid, x1, y1, x2, y2) in person_boxes]
            if self.fall:
                if self._fall_det is None:
                    from cvti.detector.fall import FallDetector
                    self._fall_det = FallDetector()
                fl = self._fall_det.update(person_boxes, frame_area, timestamp)
                if fl is not None:
                    raw_events.append(RawEvent(
                        detector="person_fall", active=True, title="PERSON COLLAPSED",
                        person_id=fl.get("track_id"), level="critical",
                        timestamp=timestamp, extra=fl))
            if self.running:
                if self._running_det is None:
                    from cvti.detector.situational import RunningPanicDetector
                    self._running_det = RunningPanicDetector(
                        min_speed_ratio=self.running_min_speed_ratio,
                        min_frames=self.running_min_frames)
                for p in people:
                    r = self._running_det.update(p["track_id"], p["bbox"], timestamp, image.shape)
                    if r is not None:
                        raw_events.append(RawEvent(
                            detector="running", active=True, title="PANIC RUNNING DETECTED",
                            person_id=p["track_id"], level="high", timestamp=timestamp, extra=r))
            if self.crowd_formation:
                if self._crowd_det is None:
                    from cvti.detector.situational import CrowdFormationDetector
                    self._crowd_det = CrowdFormationDetector(
                        min_people=self.crowd_min_people, min_frames=self.crowd_min_frames,
                        max_cluster_ratio=self.crowd_max_cluster_ratio)
                c = self._crowd_det.update(people, timestamp, image.shape)
                if c is not None:
                    raw_events.append(RawEvent(
                        detector="crowd_formation", active=True, title="UNSAFE CROWD FORMATION",
                        level="medium", timestamp=timestamp, extra=c))

        zone_by_pid: dict[Any, str | None] = {}   # person_id -> zone, for presence alerts
        if self.zone_monitor is not None:
            states = self.zone_monitor.update(tracked, timestamp)
            zone_events = zone_states_to_events(states, timestamp=timestamp)
            raw_events += zone_events
            zone_by_pid = {e.person_id: e.extra.get("zone") for e in zone_events}

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
            # Video-action runs on a CADENCE, not only when another signal fired.
            # Gating it behind concealment meant theft that isn't concealment-
            # shaped (e.g. a grab-and-go) was never seen by the fine-tuned model —
            # even when it would catch it at high confidence. The runtime's own
            # cooldown_seconds throttles how often the model actually runs, so
            # calling every frame is cheap; it only infers ~every cooldown window.
            if self._video_runtime is not None:
                raw_events += self._video_runtime.analyze_event(
                    center_frame_index=self._va_index - 1, timestamp=timestamp)
        except Exception as exc:  # noqa: BLE001 - a detector hiccup must not kill the camera
            print(f"[{self.camera_id}] detector error: {str(exc)[:140]}")

        if not raw_events:
            return []

        alerts = self.engine.evaluate(raw_events, scene_context=self.scene_context)
        if not alerts:
            return []
        from cvti.verification.frame_select import select_evidence_frames

        recent = list(self._frame_buffer)
        # Snapshot the continuous replay window once (shared by all alerts this frame).
        clip_snap = list(self._clip_buffer)
        clip_frames = [j for _, j in clip_snap]
        clip_fps = 0.0
        if len(clip_snap) >= 2:
            span = clip_snap[-1][0] - clip_snap[0][0]
            if span > 0:
                clip_fps = (len(clip_snap) - 1) / span
        out = []
        for a in alerts:
            # Zone is only meaningful for presence (zone) alerts; for other
            # detectors leave it None so the dedup key isn't polluted.
            zone = zone_by_pid.get(a.person_id) if a.detector == "presence" else None
            frames, _ = select_evidence_frames(recent, a.rule_name)
            # Whole-frame detectors (video-action, fire) carry no person_id, so an
            # alert would arrive with nothing to point at. If exactly one person is
            # tracked, that's who it's about; with several, box the most prominent
            # (largest) one. An empty scene stays unboxed — correct for fire.
            # Pose-based detectors (concealment) number people in their own ID
            # space, so a person_id may not exist in the ByteTrack map either —
            # fall back for any unresolved id, not just a missing one.
            boxes = getattr(self, "_box_by_track", {}) or {}
            bbox = boxes.get(a.person_id)
            if bbox is None and boxes:
                bbox = max(boxes.values(),
                           key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            out.append(_to_queued(self.camera_id, a, timestamp, zone,
                                  frames or [image], self.scene_context,
                                  clip_frames=clip_frames, clip_fps=clip_fps,
                                  bbox=bbox))
        return out


def build_camera_states(site_config: dict, *, pose_model: Any = None, weapon_model: Any = None,
                        video_action_model: Any = None,
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
                video_action_model=video_action_model,
                concealment=bool(cam.get("concealment")), violence=bool(cam.get("violence")),
                weapons=bool(cam.get("weapons")), theft=bool(cam.get("theft")),
                tamper=bool(cam.get("tamper")), fall=bool(cam.get("fall")),
                fire_smoke=bool(cam.get("fire_smoke")), running=bool(cam.get("running")),
                crowd_formation=bool(cam.get("crowd_formation")),
                running_min_speed_ratio=float(cam.get("running_min_speed_ratio", 0.18)),
                running_min_frames=int(cam.get("running_min_frames", 3)),
                crowd_min_people=int(cam.get("crowd_min_people", 4)),
                crowd_min_frames=int(cam.get("crowd_min_frames", 3)),
                crowd_max_cluster_ratio=float(cam.get("crowd_max_cluster_ratio", 0.24)),
                fire_min_frames=int(cam.get("fire_min_frames", 3)),
                fire_min_hot_area_ratio=float(cam.get("fire_min_hot_area_ratio", 0.012)),
                video_action=bool(cam.get("video_action")),
            ),
        }
    return out


def load_site_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
