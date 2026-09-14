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
from math import isfinite
from pathlib import Path
from typing import Any

import cv2

from cvti.contracts import RawEvent
from cvti.event_adapters import zone_states_to_events
from cvti.rules.customization import CustomizationEngine
from cvti.serving.alert_queue import QueuedAlert
from cvti.logging_setup import get_logger

log = get_logger(__name__)

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


# Replay clips are context, not forensic evidence — the gate's frames and the
# saved evidence bundle stay full-resolution elsewhere. 640w is plenty to see
# what led up to an alert, and it is the difference between the per-camera
# replay buffer costing ~7 MB + a full-1080p JPEG encode EVERY frame, and
# ~1.5 MB + an encode a fraction of that size (audit 1 Sep, D4).
CLIP_BUFFER_WIDTH = 640


def encode_clip_frame(image: Any, max_width: int = CLIP_BUFFER_WIDTH) -> bytes | None:
    """One replay-buffer frame: downscaled to <=max_width, JPEG q80.

    Returns None when encoding fails — the replay window simply misses that
    frame, which is the old behaviour for a failed encode."""
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        image = cv2.resize(image, (max_width, max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return enc.tobytes() if ok else None


@dataclass
class PerCameraState:
    camera_id: str
    engine: CustomizationEngine
    zone_monitor: Any = None          # RetailZoneMonitor | None
    scene_context: dict | None = None
    monitoring_scope: str = "full"
    # Human-reviewed context is the ONLY context allowed to suppress alerts
    # (coverage-first hybrid, 1 Sep): an unreviewed AI scene guess still
    # informs the gate and the English rules, but never silences a rule.
    # field(default=...) on purpose: this is state, not a detector toggle —
    # the toggle-coverage scanner matches the `: bool = False` shape.
    scene_reviewed: bool = field(default=False)
    active_zone_roles: set[str] = field(default_factory=set)
    person_filter: bool = True
    # The zone lane's plausibility filter was born in retail (mannequin heads,
    # reflections) and demanded a person be >=1.2% of the frame — on a wide
    # outdoor camera a real person at distance is ~0.3%, so nobody could EVER
    # enter a zone (12 Sep, VIRAT campus demo: raw=1 person, filtered=0,
    # zero presence events). None = derive from the scene: retail keeps the
    # tight gate, everything else gets a far-field one. Site key:
    # "zone_min_person_area_ratio".
    zone_min_person_area_ratio: float | None = None
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
    normal_movement: bool = False     # telemetry-only moving-person state
    multiple_people_moving: bool = False
    object_watch: bool = False
    running_min_speed_ratio: float = 0.18
    running_min_frames: int = 3
    crowd_min_people: int = 4
    crowd_min_frames: int = 3
    crowd_max_cluster_ratio: float = 0.24
    fire_min_frames: int = 3
    fire_min_hot_area_ratio: float = 0.012
    movement_enter_speed_ratio: float = 0.05
    movement_exit_speed_ratio: float = 0.02
    movement_min_track_seconds: float = 0.4
    movement_min_people: int = 2
    movement_persistence_seconds: float = 0.5
    permitted_movement_zones: tuple[str, ...] | None = None
    object_watch_library: str | None = None
    object_watch_sample_fps: float = 1.0
    object_watch_max_candidates_per_frame: int = 24
    object_watch_min_similarity: float | None = None
    object_watch_open_vocab_provider: str = "disabled"
    video_action: bool = False
    video_action_model: Any = None    # shared VideoMAEActionModel instance
    # Shared AsyncVideoActionRunner (one worker thread per site). When set,
    # inference leaves the frame loop entirely: this camera submits clips and
    # drains verdicts. None = the synchronous legacy path (single-stream CLI).
    va_runner: Any = None
    pose_conf: float = 0.35
    weapon_conf: float = 0.40
    # Inference size for the per-camera pose/weapon passes. run_site plumbs
    # its own --imgsz here (audit 1 Sep, D3): the engine spawns with 512, and
    # until 3 Sep this default silently made pose and weapon the only models
    # still paying for 640 — a leak, not a choice.
    imgsz: int = 640
    # The heavy per-camera models (pose, weapon) run every Nth frame instead
    # of every frame — their consumers are temporal accumulators plus the VLM
    # gate, and half the samples still trigger them; the model cost halves.
    heavy_stride: int = 2
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
    _heavy_tick: int = field(default=0, init=False, repr=False)
    _person_seen_last_frame: bool = field(default=False, init=False, repr=False)
    _tamper_det: Any = field(default=None, init=False, repr=False)
    _fall_det: Any = field(default=None, init=False, repr=False)
    _fire_det: Any = field(default=None, init=False, repr=False)
    _first_seen_ts: float = field(default=-1.0, init=False, repr=False)
    # Cameras auto-adjust exposure/IR for the first seconds after a stream
    # opens; the fire detector's hot-area heuristic reads that white-out as
    # flame, so EVERY engine start opened with a critical fire alert on the
    # pilot's night camera ('anytime the software starts i get this alert of
    # fire', 30 Aug). Situational detectors sit out the settle window.
    settle_seconds: float = 8.0
    _running_det: Any = field(default=None, init=False, repr=False)
    _crowd_det: Any = field(default=None, init=False, repr=False)
    _motion_tracker: Any = field(default=None, init=False, repr=False)
    _simultaneous_movement_det: Any = field(default=None, init=False, repr=False)
    _object_matcher: Any = field(default=None, init=False, repr=False)
    _object_state_tracker: Any = field(default=None, init=False, repr=False)
    _next_object_watch_sample_ts: float = field(default=-1.0, init=False, repr=False)
    object_watch_skipped_over_budget: int = field(default=0, init=False)
    _motion_overlays: list[dict] = field(default_factory=list, init=False, repr=False)
    context_decisions: list[dict] = field(default_factory=list, init=False)
    context_suppression_count: int = field(default=0, init=False)
    # Rolling recent frames (~2s at 5 FPS) so the gate gets per-rule evidence
    # (motion-peak span for violence, sharpest single frame for weapons).
    _frame_buffer: deque = field(default_factory=lambda: deque(maxlen=10), init=False, repr=False)
    # Longer CONTINUOUS window, JPEG-encoded (kept light for RAM), so a confirmed
    # alert can be replayed as a real video of the event lead-up, not a slideshow.
    # ~48 frames ≈ 8-12s at typical pipeline fps. Holds (timestamp, jpeg_bytes).
    _clip_buffer: deque = field(default_factory=lambda: deque(maxlen=48), init=False, repr=False)

    def activate_scene_context(self, context: dict) -> None:
        """Apply reviewed semantic context without rebuilding detector state."""
        self.scene_context = dict(context)
        self.monitoring_scope = "full"
        self.scene_reviewed = True

    def __post_init__(self) -> None:
        import warnings
        import supervision as sv
        from cvti.health import component
        # Per camera, not per detector class: an operator asks "is camera 4 ok?",
        # and a camera whose detectors throw on every frame looks exactly like a
        # camera watching a quiet corridor.
        self._health = component(f"detector.{self.camera_id}")
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
        if self.normal_movement or self.multiple_people_moving:
            from cvti.detector.person_motion import (
                PersonMotionTracker,
                SimultaneousMovementDetector,
            )
            self._motion_tracker = PersonMotionTracker(
                enter_speed_ratio=self.movement_enter_speed_ratio,
                exit_speed_ratio=self.movement_exit_speed_ratio,
                min_track_seconds=self.movement_min_track_seconds,
            )
            self._simultaneous_movement_det = SimultaneousMovementDetector(
                min_people=self.movement_min_people,
                persistence_seconds=self.movement_persistence_seconds,
            )
        if self.object_watch:
            from cvti.object_watch.embeddings import load_embedding_backend
            from cvti.object_watch.matcher import ObjectMatcher
            from cvti.object_watch.tracker import ObjectStateTracker

            backend = load_embedding_backend("hash")
            root = self.object_watch_library or "."
            self._object_matcher = ObjectMatcher(
                root,
                backend,
                max_candidates_per_frame=self.object_watch_max_candidates_per_frame,
            )
            self._object_state_tracker = ObjectStateTracker()

    def _movement_zone_allows(self, zone_names: tuple[str, ...]) -> bool:
        if self.permitted_movement_zones is None:
            return True
        return bool(set(zone_names).intersection(self.permitted_movement_zones))

    def _needs_pose(self) -> bool:
        return self.pose_model is not None and (self.concealment or self.violence or self.theft)

    def _compute_pose(self, image: Any, timestamp: float) -> list:
        """Shared pose model on this camera's frame + per-camera track state, so
        wrist speed/dwell history is not mixed across cameras."""
        from cvti.detector.core import (
            assign_pose_tracks, enrich_pose_people_with_history, extract_pose_people,
        )
        from cvti.serving.perf import BOARD
        started = time.perf_counter()
        try:
            pose_people = extract_pose_people(
                self.pose_model, image, self.pose_conf, self.imgsz
            )
        finally:
            BOARD.observe(
                "pose_infer", self.camera_id,
                (time.perf_counter() - started) * 1000.0,
            )
        pose_people, self._next_pose_id = assign_pose_tracks(
            pose_people, previous_people=self._prev_pose, next_track_id=self._next_pose_id)
        pose_people = enrich_pose_people_with_history(pose_people, self._pose_history)
        self._prev_pose = list(pose_people)
        return pose_people

    def _merged_detections(self, object_detections: list | None, image: Any,
                           include_weapon: bool = True) -> list:
        """Shared object detections + (optional) per-camera weapon-model detections.

        `include_weapon=False` skips the weapon model's forward pass (the
        person-gate/cadence below decided this frame doesn't earn one) while
        still handing the assessments the shared detections."""
        merged = list(object_detections or [])
        if include_weapon and self.weapon_model is not None:
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

    def _object_watch_due(self, timestamp: float) -> bool:
        if self._next_object_watch_sample_ts < 0:
            self._next_object_watch_sample_ts = timestamp + (1.0 / self.object_watch_sample_fps)
            return True
        if timestamp + 1e-9 < self._next_object_watch_sample_ts:
            return False
        self._next_object_watch_sample_ts = timestamp + (1.0 / self.object_watch_sample_fps)
        return True

    def _object_candidates(
        self,
        object_detections: list | None,
        frame_hw: tuple[int, int],
    ) -> list:
        from cvti.object_watch.matcher import ObjectCandidate

        candidates = []
        for detection in object_detections or []:
            label = str(getattr(detection, "label", "") or "")
            if label.lower() == "person":
                continue
            bbox = tuple(int(v) for v in getattr(detection, "bbox", ()))
            if len(bbox) != 4:
                continue
            candidates.append(ObjectCandidate(
                bbox=bbox,
                label_hint=label,
                confidence=float(getattr(detection, "confidence", 0.0) or 0.0),
                track_id=getattr(detection, "track_id", None),
                zone_id=self._zone_for_bbox(bbox, frame_hw),
            ))
        return candidates

    def _zone_for_bbox(self, bbox: tuple[int, int, int, int],
                       frame_hw: tuple[int, int]) -> str | None:
        if self.zone_monitor is None:
            return None
        try:
            import cv2
            self.zone_monitor._fit_to_frame(frame_hw)
            x1, y1, x2, y2 = bbox
            point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            for spec in self.zone_monitor.zones:
                if cv2.pointPolygonTest(spec.polygon, point, False) >= 0:
                    return spec.name
        except Exception:  # noqa: BLE001 - zone hinting must not block object matching
            return None
        return None

    def process(self, detections: Any, image: Any, timestamp: float,
                object_detections: list | None = None) -> list[QueuedAlert]:
        """Track + run all enabled signals (zones, concealment, violence, weapons,
        theft) + rules; return candidate alerts with per-rule evidence frames.

        `detections` is sv.Detections (tracking/zones); `object_detections` is the
        core.py Detection list from the same frame (weapons/violence/theft)."""
        from cvti.retail.concealment import personal_bag_boxes
        from cvti.retail.zones import filter_person_detections

        bag_boxes = personal_bag_boxes(detections) if self._conceal is not None else []
        frame_hw = image.shape[:2]
        self._frame_buffer.append(image)
        # Continuous replay buffer: a rolling JPEG window with timestamps so a
        # confirmed alert replays as real video of the lead-up.
        clip_jpeg = encode_clip_frame(image)
        if clip_jpeg is not None:
            self._clip_buffer.append((timestamp, clip_jpeg))

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

        if self._first_seen_ts < 0:
            self._first_seen_ts = timestamp
        settled = (timestamp - self._first_seen_ts) >= self.settle_seconds

        # Fire + smoke — pure CV pre-filter on the raw frame (no person needed).
        if self.fire_smoke and settled:
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
            ratio = self.zone_min_person_area_ratio
            if ratio is None:
                env = (self.scene_context or {}).get("environment_type", "")
                ratio = 0.012 if env in ("retail_shop", "mall_corridor") else 0.002
            detections = filter_person_detections(detections, frame_hw,
                                                  min_area_ratio=ratio)
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
            if self.running and settled:
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
            if self.crowd_formation and settled:
                # Count from the RAW detections, not the tracked ones: ByteTrack
                # exists to give people stable identities, and it drops
                # low-confidence boxes to do that. In a packed scene those dropped
                # boxes ARE the crowd, so tracking starved the count to zero.
                raw_people = [{"track_id": i, "bbox": (int(x1), int(y1), int(x2), int(y2))}
                              for i, (_t, x1, y1, x2, y2) in enumerate(_person_boxes(detections))]
                if self._crowd_det is None:
                    from cvti.detector.situational import CrowdFormationDetector
                    self._crowd_det = CrowdFormationDetector(
                        min_people=self.crowd_min_people, min_frames=self.crowd_min_frames,
                        max_cluster_ratio=self.crowd_max_cluster_ratio)
                c = self._crowd_det.update(raw_people or people, timestamp, image.shape)
                if c is not None:
                    raw_events.append(RawEvent(
                        detector="crowd_formation", active=True, title="UNSAFE CROWD FORMATION",
                        level="medium", timestamp=timestamp, extra=c))

        zone_by_pid: dict[Any, str | None] = {}   # person_id -> zone, for presence alerts
        zones_by_track: dict[int, tuple[str, ...]] = {}
        if self.zone_monitor is not None:
            # frame_hw lets normalized (0..1) zone polygons fit THIS camera's
            # resolution, and tells the monitor where the frame edges are.
            states = self.zone_monitor.update(tracked, timestamp, frame_hw=frame_hw)
            zone_events = zone_states_to_events(states, timestamp=timestamp)
            # Exits are debounced inside the monitor; drain them each frame so a
            # person leaving a restricted zone is its own event, not silence.
            exits = self.zone_monitor.drain_exits()
            if exits:
                from cvti.event_adapters import zone_exits_to_events
                zone_events = zone_events + zone_exits_to_events(exits, timestamp=timestamp)
            raw_events += zone_events
            zone_by_pid = {e.person_id: e.extra.get("zone") for e in zone_events
                           if e.extra.get("zone")}
            zones_by_track = {
                int(state.tracker_id): tuple(state.zones)
                for state in states
                if state.tracker_id is not None
            }

        if self._motion_tracker is not None:
            motions = self._motion_tracker.update(
                _person_boxes(tracked), timestamp, frame_hw,
                zones_by_track=zones_by_track,
            )
            permitted_motions = [
                motion for motion in motions
                if self._movement_zone_allows(motion.zone_names)
            ]
            if self.multiple_people_moving:
                movement_event = self._simultaneous_movement_det.update(
                    permitted_motions, timestamp
                )
                if movement_event is not None:
                    from cvti.event_adapters import simultaneous_movement_to_event
                    raw_events.append(
                        simultaneous_movement_to_event(movement_event, timestamp)
                    )
            active_track_ids = (
                set(self._simultaneous_movement_det.active_track_ids)
                if self.multiple_people_moving else set()
            )
            self._motion_overlays = [
                {
                    "track_id": motion.track_id,
                    "bbox": tuple(int(v) for v in motion.bbox),
                    "label": f"#{motion.track_id} MOVING",
                    "zone_names": list(motion.zone_names),
                    "speed_ratio": motion.speed_ratio,
                    "colour": ((0, 200, 255) if motion.track_id in active_track_ids
                               else (0, 200, 0)),
                }
                for motion in permitted_motions
                if motion.observed and motion.moving
                and (self.normal_movement or motion.track_id in active_track_ids)
            ]

        try:
            if self._conceal is not None:
                self._conceal.expire(timestamp)
            # Person-gate + cadence for the HEAVY per-camera models (audit
            # 1 Sep, D3). Pose and weapon are full forward passes per camera
            # per frame, and they ran unconditionally — on empty corridors,
            # on parked cars, all night. Everything they feed is about a
            # person doing something, so: no person in the shared detector's
            # frame, no heavy models. With people present they run on a
            # stride (every Nth frame — their consumers are temporal
            # accumulators and the VLM gate, which survive half the samples),
            # except the first frame a person APPEARS, which always runs so a
            # threat's opening moment is never the one we skipped. The gate
            # reads the RAW detections, not the tracked ones — ByteTrack
            # drops low-confidence boxes, and a half-seen person is exactly
            # who the weapon model should look at.
            person_present = bool(_person_boxes(detections))
            first_appearance = person_present and not self._person_seen_last_frame
            self._person_seen_last_frame = person_present
            self._heavy_tick += 1
            run_heavy = person_present and (
                first_appearance or self._heavy_tick % max(1, self.heavy_stride) == 0)
            pose_ran = self._needs_pose() and run_heavy
            pose_people = self._compute_pose(image, timestamp) if pose_ran else []
            if self._conceal is not None:
                from cvti.detector.core import pose_people_to_concealment_frames
                from cvti.event_adapters import concealment_to_events
                if pose_ran:
                    pose_frames = pose_people_to_concealment_frames(pose_people, timestamp)
                    assessments = self._conceal.update_with_bag_detections(
                        pose_frames, timestamp, bag_boxes
                    )
                    raw_events += concealment_to_events(assessments, timestamp)
            if self.violence or self.weapons or self.theft:
                merged = self._merged_detections(object_detections, image,
                                                 include_weapon=run_heavy)
                raw_events += self._assessment_events(pose_people, merged, image, timestamp)
            # Video-action runs on a CADENCE, not only when another signal fired.
            # Gating it behind concealment meant theft that isn't concealment-
            # shaped (e.g. a grab-and-go) was never seen by the fine-tuned model —
            # even when it would catch it at high confidence. The runtime's own
            # cooldown_seconds throttles how often clips are queued, so calling
            # every frame is cheap.
            if self._video_runtime is not None:
                if self.va_runner is not None:
                    # Off the frame loop (audit 1 Sep, D1): the 2–6s forward
                    # pass used to run HERE, stalling every camera on the
                    # site. Now this frame only collects verdicts that
                    # finished since last frame and queues a clip snapshot;
                    # the shared worker does the waiting.
                    raw_events += self.va_runner.drain(self.camera_id)
                    clip = self._video_runtime.prepare_analysis(
                        center_frame_index=self._va_index - 1, timestamp=timestamp)
                    if clip is not None:
                        self.va_runner.submit(self.camera_id, clip)
                else:
                    raw_events += self._video_runtime.analyze_event(
                        center_frame_index=self._va_index - 1, timestamp=timestamp)
            self._health.ok()
        except Exception as exc:  # noqa: BLE001 - a detector hiccup must not kill the camera
            # Rate-limited: a detector throwing on every frame must not fill the
            # disk with its own traceback. The counter carries the true scale.
            self._health.failed(exc, log, "processing a frame")

        if self.object_watch and self._object_matcher is not None \
                and self._object_state_tracker is not None \
                and self._object_watch_due(timestamp):
            try:
                from cvti.event_adapters import object_observations_to_events
                candidates = self._object_candidates(object_detections, frame_hw)
                vehicle_boxes = [
                    tuple(int(v) for v in getattr(detection, "bbox", ()))
                    for detection in object_detections or []
                    if str(getattr(detection, "label", "") or "").lower()
                    in {"car", "truck", "bus", "vehicle", "van", "lorry"}
                    and len(tuple(getattr(detection, "bbox", ()))) == 4
                ]
                matches = self._object_matcher.match(
                    self.camera_id, image, candidates, timestamp
                )
                self.object_watch_skipped_over_budget += int(
                    self._object_matcher.last_stats.get("skipped_over_budget", 0)
                )
                raw_events += object_observations_to_events(
                    self._object_state_tracker.update(matches, timestamp, vehicle_boxes),
                    timestamp,
                )
            except Exception as exc:  # noqa: BLE001 - object watch is optional
                self._health.failed(exc, log, "processing object watch")

        if not raw_events:
            return []

        alerts = self.engine.evaluate(
            raw_events,
            scene_context=self.scene_context,
            active_zone_roles=self.active_zone_roles,
            monitoring_scope=self.monitoring_scope,
            scene_reviewed=self.scene_reviewed,
        )
        self.context_decisions = list(self.engine.context_decisions)
        self.context_suppression_count += sum(
            decision.get("decision") == "context_incompatible"
            for decision in self.context_decisions
        )
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
            zone = (
                a.metadata.get("zone")
                if a.detector == "object_watch"
                else zone_by_pid.get(a.person_id) if a.detector == "presence" else None
            )
            frames, _ = select_evidence_frames(recent, a.rule_name)
            # Whole-frame detectors (video-action, fire) carry no person_id, so an
            # alert would arrive with nothing to point at. If exactly one person is
            # tracked, that's who it's about; with several, box the most prominent
            # (largest) one. An empty scene stays unboxed — correct for fire.
            # Pose-based detectors (concealment) number people in their own ID
            # space, so a person_id may not exist in the ByteTrack map either —
            # fall back for any unresolved id, not just a missing one.
            boxes = getattr(self, "_box_by_track", {}) or {}
            bbox = (a.metadata.get("group_bbox")
                    if a.detector == "multiple_people_moving" else None)
            if bbox is None and a.detector == "object_watch":
                bbox = a.metadata.get("bbox")
            if bbox is None:
                bbox = boxes.get(a.person_id)
            if bbox is None and boxes:
                bbox = max(boxes.values(),
                           key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            # Evidence upgrade: full frames give the gate context; a zoomed
            # crop of the flagged subject gives it hands and held objects —
            # its own rejections say "no people visible" on full CCTV frames.
            from cvti.verification.frame_select import append_subject_crop
            if a.detector == "multiple_people_moving":
                evidence = frames or [image]
            else:
                evidence = append_subject_crop(frames or [image], image, bbox)
            out.append(_to_queued(self.camera_id, a, timestamp, zone,
                                  evidence, self.scene_context,
                                  clip_frames=clip_frames, clip_fps=clip_fps,
                                  bbox=bbox))
        return out


def build_camera_states(site_config: dict, *, pose_model: Any = None, weapon_model: Any = None,
                        video_action_model: Any = None,
                        va_runner: Any = None,
                        baseline_config: str | None = None,
                        scene_contexts: dict[str, dict] | None = None,
                        monitoring_scopes: dict[str, str] | None = None,
                        reviewed_camera_ids: set | None = None,
                        imgsz: int = 640,
                        output_dir: str | Path | None = None) -> dict[str, dict]:
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
        detector_flags = (
            "concealment", "violence", "weapons", "theft", "tamper", "fall",
            "fire_smoke", "running", "crowd_formation", "normal_movement",
            "multiple_people_moving", "video_action", "object_watch",
        )
        for flag in detector_flags:
            if flag in cam and not isinstance(cam[flag], bool):
                raise ValueError(f"camera {cam_id}: {flag} must be a boolean")
        if cam.get("view_only"):
            # A view-only camera is glass, not a detector (4 Sep, pilot): it
            # streams to the wall and runs nothing — no state, no rules, no
            # baseline. The pipeline still decodes and publishes it.
            continue
        float_movement_fields = (
            "movement_enter_speed_ratio",
            "movement_exit_speed_ratio",
            "movement_min_track_seconds",
            "movement_persistence_seconds",
        )
        for field_name in float_movement_fields:
            if isinstance(cam.get(field_name), bool):
                raise ValueError(
                    f"camera {cam_id}: {field_name} must be a number, not boolean"
                )
        try:
            movement_enter = float(cam.get("movement_enter_speed_ratio", 0.05))
            movement_exit = float(cam.get("movement_exit_speed_ratio", 0.02))
            movement_min_track = float(cam.get("movement_min_track_seconds", 0.4))
            movement_persistence = float(cam.get("movement_persistence_seconds", 0.5))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera {cam_id}: invalid movement configuration: {exc}") from exc
        movement_min_people_raw = cam.get("movement_min_people", 2)
        if isinstance(movement_min_people_raw, bool):
            raise ValueError(f"camera {cam_id}: movement_min_people must be an integer")
        try:
            movement_min_people_number = float(movement_min_people_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera {cam_id}: invalid movement configuration: {exc}") from exc
        if not isfinite(movement_min_people_number) or not movement_min_people_number.is_integer():
            raise ValueError(f"camera {cam_id}: movement_min_people must be an integer")
        movement_min_people = int(movement_min_people_number)
        if not all(isfinite(value) for value in (
            movement_enter, movement_exit, movement_min_track, movement_persistence,
        )):
            raise ValueError(f"camera {cam_id}: movement thresholds must be finite")
        if movement_enter <= 0:
            raise ValueError(f"camera {cam_id}: movement_enter_speed_ratio must be positive")
        if movement_exit <= 0 or movement_exit >= movement_enter:
            raise ValueError(
                f"camera {cam_id}: movement_exit_speed_ratio must be positive and lower "
                "than movement_enter_speed_ratio"
            )
        if movement_min_track <= 0:
            raise ValueError(f"camera {cam_id}: movement_min_track_seconds must be positive")
        if movement_min_people < 2:
            raise ValueError(f"camera {cam_id}: movement_min_people must be at least 2")
        if movement_persistence <= 0:
            raise ValueError(f"camera {cam_id}: movement_persistence_seconds must be positive")
        for field_name in ("object_watch_sample_fps", "object_watch_min_similarity"):
            if isinstance(cam.get(field_name), bool):
                raise ValueError(f"camera {cam_id}: {field_name} must be a number, not boolean")
        max_candidates_raw = cam.get("object_watch_max_candidates_per_frame", 24)
        if isinstance(max_candidates_raw, bool):
            raise ValueError(
                f"camera {cam_id}: object_watch_max_candidates_per_frame must be an integer"
            )
        try:
            object_watch_sample_fps = float(cam.get("object_watch_sample_fps", 1.0))
            object_watch_min_similarity = (
                None if cam.get("object_watch_min_similarity") is None
                else float(cam.get("object_watch_min_similarity"))
            )
            max_candidates_number = float(max_candidates_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera {cam_id}: invalid object watch configuration: {exc}") from exc
        if not isfinite(object_watch_sample_fps) or object_watch_sample_fps <= 0:
            raise ValueError(f"camera {cam_id}: object_watch_sample_fps must be positive")
        if not isfinite(max_candidates_number) or not max_candidates_number.is_integer() \
                or max_candidates_number < 1:
            raise ValueError(
                f"camera {cam_id}: object_watch_max_candidates_per_frame must be a positive integer"
            )
        if object_watch_min_similarity is not None and (
            not isfinite(object_watch_min_similarity) or not 0.0 <= object_watch_min_similarity <= 1.0
        ):
            raise ValueError(f"camera {cam_id}: object_watch_min_similarity must be between 0 and 1")
        permitted_raw = cam.get("permitted_movement_zones")
        permitted_zones: tuple[str, ...] | None = None
        if permitted_raw is not None:
            if not isinstance(permitted_raw, (list, tuple)):
                raise ValueError(
                    f"camera {cam_id}: permitted_movement_zones must be a list or tuple"
                )
            if any(not isinstance(zone, str) or not zone.strip() for zone in permitted_raw):
                raise ValueError(
                    f"camera {cam_id}: permitted_movement_zones must contain non-empty strings"
                )
            if not cam.get("zones"):
                raise ValueError(
                    f"camera {cam_id}: permitted_movement_zones requires a zone config"
                )
            permitted_zones = tuple(permitted_raw)
        engine = CustomizationEngine(cam["config"], baseline_path=baseline_config)
        zone_monitor = None
        if cam.get("zones"):
            zone_specs = load_zone_config(cam["zones"])
            if permitted_zones is not None:
                configured_names = {spec.name for spec in zone_specs}
                unknown_names = sorted(set(permitted_zones) - configured_names)
                if unknown_names:
                    raise ValueError(
                        f"camera {cam_id}: unknown permitted movement zones: {unknown_names}"
                    )
            zone_monitor = RetailZoneMonitor(zone_specs)
        scene = (scene_contexts or {}).get(cam_id)
        if scene is None and cam.get("scene_description"):
            scene = {"environment_type": cam.get("environment_type", "unknown"),
                     "scene_description": cam["scene_description"]}
        active_zone_roles = set(cam.get("accepted_zone_roles") or [])
        if cam.get("zones"):
            try:
                zone_data = json.loads(Path(cam["zones"]).read_text())
                kind_roles = {"shelf": "merchandise", "checkout": "checkout"}
                for zone in zone_data.get("zones", []):
                    role = zone.get("context_role") or kind_roles.get(zone.get("kind"))
                    if role:
                        active_zone_roles.add(str(role))
            except (OSError, ValueError, TypeError):
                log.warning("unable to read accepted zone roles for %s", cam_id)
        out[cam_id] = {
            "source": cam["source"],
            "state": PerCameraState(
                cam_id, engine, zone_monitor=zone_monitor, scene_context=scene,
                monitoring_scope=(monitoring_scopes or {}).get(cam_id, "full"),
                scene_reviewed=cam_id in (reviewed_camera_ids or set()),
                active_zone_roles=active_zone_roles,
                pose_model=pose_model, weapon_model=weapon_model,
                video_action_model=video_action_model,
                va_runner=va_runner,
                # The engine's one inference size: pose/weapon run at the
                # same imgsz as the shared detector, not a leaked default.
                imgsz=imgsz,
                heavy_stride=int(cam.get("heavy_stride", 2)),
                concealment=cam.get("concealment", False),
                violence=cam.get("violence", False),
                weapons=cam.get("weapons", False), theft=cam.get("theft", False),
                tamper=cam.get("tamper", False), fall=cam.get("fall", False),
                fire_smoke=cam.get("fire_smoke", False),
                running=cam.get("running", False),
                crowd_formation=cam.get("crowd_formation", False),
                normal_movement=cam.get("normal_movement", False),
                multiple_people_moving=cam.get("multiple_people_moving", False),
                running_min_speed_ratio=float(cam.get("running_min_speed_ratio", 0.18)),
                running_min_frames=int(cam.get("running_min_frames", 3)),
                crowd_min_people=int(cam.get("crowd_min_people", 4)),
                crowd_min_frames=int(cam.get("crowd_min_frames", 3)),
                crowd_max_cluster_ratio=float(cam.get("crowd_max_cluster_ratio", 0.24)),
                fire_min_frames=int(cam.get("fire_min_frames", 3)),
                fire_min_hot_area_ratio=float(cam.get("fire_min_hot_area_ratio", 0.012)),
                movement_enter_speed_ratio=movement_enter,
                movement_exit_speed_ratio=movement_exit,
                movement_min_track_seconds=movement_min_track,
                movement_min_people=movement_min_people,
                movement_persistence_seconds=movement_persistence,
                permitted_movement_zones=permitted_zones,
                object_watch=cam.get("object_watch", False),
                object_watch_library=cam.get("object_watch_library") or (
                    str(Path(output_dir) / "object_library") if output_dir else None
                ),
                object_watch_sample_fps=object_watch_sample_fps,
                object_watch_max_candidates_per_frame=int(max_candidates_number),
                object_watch_min_similarity=object_watch_min_similarity,
                object_watch_open_vocab_provider=str(
                    cam.get("object_watch_open_vocab_provider", "disabled")
                ),
                video_action=cam.get("video_action", False),
                zone_min_person_area_ratio=cam.get("zone_min_person_area_ratio"),
            ),
        }
    return out


def refresh_camera_rules(state: "PerCameraState", cam: dict,
                         baseline_config: str | None = None) -> None:
    """Hot-swap a camera's rules engine and zone monitor from its config files.

    Rules and zones are JSON — reloading them is cheap and touches no model,
    so a plain-English rule typed in the app takes effect on the RUNNING
    engine within seconds instead of waiting for a restart nobody was told to
    do. Attribute swaps are atomic in CPython: a frame in flight sees either
    the old engine or the new one, never a half-built one.
    """
    from cvti.retail.zones import RetailZoneMonitor, load_zone_config
    state.engine = CustomizationEngine(cam["config"], baseline_path=baseline_config)
    if cam.get("zones"):
        state.zone_monitor = RetailZoneMonitor(load_zone_config(cam["zones"]))


def load_site_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
