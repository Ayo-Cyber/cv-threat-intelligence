from __future__ import annotations

import argparse
from collections import deque
import importlib
import json
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cvti.event_adapters import (
    assessments_to_events,
    concealment_to_events,
    zone_states_to_events,
)
from cvti.rules.customization import CustomizationEngine
from cvti.serving.alert_queue import AlertQueue, QueuedAlert
from cvti.verification.frame_select import select_evidence_frames
from cvti.verification.gate import VerificationGate
from cvti.video_action_runtime import build_video_action_runtime

import cv2
import torch
from ultralytics import YOLO
from cvti.logging_setup import get_logger

log = get_logger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*deprecated.*",
    category=FutureWarning,
)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    is_threat: bool
    source_model: str


@dataclass
class ThreatAssessment:
    active: bool
    title: str
    level: str
    reasons: list[str]
    weapon_labels: list[str]
    explicit_labels: list[str]


@dataclass
class LoadedModel:
    runner: Any
    kind: str
    names: dict[int, str]
    source_path: str


@dataclass
class PosePersonState:
    track_id: int
    bbox: tuple[int, int, int, int]
    timestamp: float
    left_shoulder: tuple[float, float] | None
    right_shoulder: tuple[float, float] | None
    left_elbow: tuple[float, float] | None
    right_elbow: tuple[float, float] | None
    left_wrist: tuple[float, float] | None
    right_wrist: tuple[float, float] | None
    max_wrist_speed: float
    max_wrist_accel: float
    max_arm_extension_ratio: float
    weapon_labels: list[str]
    # Hips: used by the concealment (action) layer for the hand-to-waist signal.
    # Optional/defaulted so existing violence-path constructions are unaffected.
    left_hip: tuple[float, float] | None = None
    right_hip: tuple[float, float] | None = None


class EventRecorder:
    def __init__(self, output_root: Path, clip_seconds: int, fps_fallback: float) -> None:
        self.output_root = output_root
        self.clip_seconds = clip_seconds
        self.fps_fallback = fps_fallback
        self.writer: cv2.VideoWriter | None = None
        self.event_dir: Path | None = None
        self.clip_path: Path | None = None
        self.event_deadline = 0.0
        self.event_count = 0

    def start(
        self,
        frame: Any,
        detections: list[Detection],
        assessment: ThreatAssessment,
        source: str,
        fps: float,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_dir = self.output_root / f"event_{timestamp}_{self.event_count:03d}"
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.event_count += 1

        image_path = self.event_dir / "frame.jpg"
        cv2.imwrite(str(image_path), frame)

        metadata = {
            "timestamp": timestamp,
            "source": source,
            "threat_assessment": {
                "active": assessment.active,
                "title": assessment.title,
                "level": assessment.level,
                "reasons": assessment.reasons,
                "weapon_labels": assessment.weapon_labels,
                "explicit_labels": assessment.explicit_labels,
            },
            "detections": [
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 4),
                    "bbox": list(detection.bbox),
                    "is_threat": detection.is_threat,
                }
                for detection in detections
            ],
        }
        metadata_path = self.event_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.clip_path = self.event_dir / "clip.mp4"
        self.writer = cv2.VideoWriter(
            str(self.clip_path),
            fourcc,
            fps if fps > 0 else self.fps_fallback,
            (width, height),
        )
        self.event_deadline = time.time() + self.clip_seconds
        return self.event_dir

    def write(self, frame: Any) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def should_stop(self) -> bool:
        return self.writer is not None and time.time() >= self.event_deadline

    def stop(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.event_dir = None
        self.clip_path = None
        self.event_deadline = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threat detection POC for webcam, RTSP, or video input."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, RTSP URL, or video file path. Example: 0 or rtsp://...",
    )
    parser.add_argument(
        "--weights",
        default="models/yolov8n.pt",
        help="Path to default YOLO weights. Ultralytics model names also work.",
    )
    parser.add_argument(
        "--person-weights",
        default="",
        help="Optional YOLO weights dedicated to person detection.",
    )
    parser.add_argument(
        "--weapon-weights",
        default="",
        help="Optional YOLO weights dedicated to dangerous-object detection.",
    )
    parser.add_argument(
        "--yolov5-repo",
        default="external/yolov5",
        help="Local YOLOv5 repo path used when loading legacy YOLOv5 checkpoints.",
    )
    parser.add_argument(
        "--weapon-loader",
        default="auto",
        choices=("auto", "ultralytics", "yolov5"),
        help="How to load `--weapon-weights`. Use `yolov5` for legacy YOLOv5 checkpoints like `models\\weapon_best.pt`.",
    )
    parser.add_argument(
        "--pose-weights",
        default="models/yolov8n-pose.pt",
        help="Pose model weights used for violence heuristics. Set to empty to disable pose-based violence logic.",
    )
    parser.add_argument(
        "--threat-classes",
        default="gun,knife",
        help="Comma-separated explicit class names that should trigger an alert.",
    )
    parser.add_argument(
        "--person-classes",
        default="person",
        help="Comma-separated labels treated as person classes by the threat rules.",
    )
    parser.add_argument(
        "--weapon-classes",
        default="knife,gun",
        help="Comma-separated dangerous-object class labels used by the threat rules.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Default confidence threshold for detections.",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for the person model.",
    )
    parser.add_argument(
        "--weapon-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for the weapon model.",
    )
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for the pose model.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Minimum seconds between new threat events.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=int,
        default=5,
        help="How long to keep recording after a threat event starts.",
    )
    parser.add_argument(
        "--save-dir",
        default="runs/detect",
        help="Directory for saved evidence.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the live window. Recommended for local demos.",
    )
    parser.add_argument(
        "--save-video",
        default="",
        help="Write the full annotated output to this .mp4 path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional limit for debugging. 0 means unlimited.",
    )
    parser.add_argument(
        "--assault-distance-ratio",
        type=float,
        default=1.2,
        help="How close two people must be to flag a possible assault.",
    )
    parser.add_argument(
        "--min-threat-frames",
        type=int,
        default=3,
        help="Minimum consecutive frames a threat must persist before it becomes active.",
    )
    parser.add_argument(
        "--debug-weapon",
        action="store_true",
        help="Print exact weapon detections and confidences to the terminal when they change.",
    )
    parser.add_argument(
        "--show-all-detections",
        action="store_true",
        help="Draw every raw detection. By default the overlay focuses on people and validated threat-related detections only.",
    )
    parser.add_argument(
        "--violence-distance-ratio",
        type=float,
        default=1.1,
        help="How close two people must be before violence heuristics consider them interacting.",
    )
    parser.add_argument(
        "--violence-wrist-speed",
        type=float,
        default=120.0,
        help="Minimum wrist speed in pixels per second used by the violence heuristics.",
    )
    parser.add_argument(
        "--violence-arm-extension-ratio",
        type=float,
        default=0.35,
        help="Minimum normalized arm extension used for armed-assault heuristics.",
    )
    parser.add_argument(
        "--violence-wrist-accel",
        type=float,
        default=800.0,
        help="Minimum wrist acceleration (px/s²) that counts as an aggressive motion signature.",
    )
    parser.add_argument(
        "--violence-gate-window",
        type=int,
        default=8,
        help="Rolling frame window for the violence temporal gate.",
    )
    parser.add_argument(
        "--violence-gate-votes",
        type=int,
        default=3,
        help="Minimum active-frame votes within the window before violence alert is confirmed.",
    )
    parser.add_argument(
        "--theft-acquire-frames",
        type=int,
        default=8,
        help="Frames a wrist must stay near an object to count as an acquisition.",
    )
    parser.add_argument(
        "--theft-depart-frames",
        type=int,
        default=6,
        help="Frames person must stay in DEPART state before a theft alert fires.",
    )
    parser.add_argument(
        "--theft-approach-ratio",
        type=float,
        default=2.0,
        help="Max center-distance ratio between person and object to enter APPROACH state.",
    )
    parser.add_argument(
        "--debug-theft",
        action="store_true",
        help="Print theft state machine transitions (IDLE→APPROACH→ACQUIRE→DEPART) as they happen.",
    )
    parser.add_argument(
        "--weapon-hand-distance-ratio",
        type=float,
        default=0.20,
        help="Margin ratio used when deciding whether a wrist is close enough to a weapon bbox.",
    )
    parser.add_argument(
        "--weapon-min-area-ratio",
        type=float,
        default=0.002,
        help="Reject weapon boxes smaller than this fraction of the frame area.",
    )
    parser.add_argument(
        "--weapon-max-area-ratio",
        type=float,
        default=0.18,
        help="Reject weapon boxes larger than this fraction of the frame area.",
    )
    parser.add_argument(
        "--weapon-border-margin-ratio",
        type=float,
        default=0.03,
        help="Reject weapon boxes that hug the frame edge more than this margin ratio.",
    )
    parser.add_argument(
        "--allow-unattached-weapons",
        action="store_true",
        help="Allow weapon detections that are not attached to a person or hand. Default behavior is stricter for live demos.",
    )
    parser.add_argument(
        "--violence-min-frames",
        type=int,
        default=4,
        help="Minimum consecutive frames a violence heuristic must persist before it becomes active.",
    )
    parser.add_argument(
        "--debug-violence",
        action="store_true",
        help="Print exact violence heuristic signals when they change.",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable ByteTrack person tracking and use plain predict instead. Use this if tracking causes crashes.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=("all", "theft", "violence", "weapons"),
        help=(
            "Detection mode: 'all' runs everything, 'theft' only runs the theft state machine, "
            "'violence' only runs pose heuristics, 'weapons' only runs object/weapon detection."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to user_config.json (Customization Engine rules). Example: configs/retail_v1.json",
    )
    parser.add_argument(
        "--max-alerts-per-frame",
        type=int,
        default=2,
        help="Max queued candidate alerts verified by the gate per frame (Phase 1 "
             "throttle). Extra concurrent alerts wait in the queue for later frames.",
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        default="configs/baseline_critical_v1.json",
        help="Always-on critical safety rules merged with --config (Phase 2). "
             "Fires weapon/violence/etc. even if the customer config omits them.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Disable the always-on critical baseline (dev/testing only).",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default="",
        help="Zone-geometry JSON (e.g. configs/retail_zones.theft_shop_01.json). Enables shelf/"
             "restricted zones + dwell -> 'presence' events (loitering, after-hours). Needs --pose-weights.",
    )
    parser.add_argument(
        "--concealment",
        action="store_true",
        help="Enable the pose-based concealment (action) detector -> 'concealment' events. Needs --pose-weights.",
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default="",
        help="Path to fine-tuned YOLOv8s-cls weights for violence/theft/normal classification. "
             "Example: runs/classify/runs/classify/robbery_v18/weights/best.pt",
    )
    parser.add_argument(
        "--classifier-conf",
        type=float,
        default=0.55,
        help="Minimum confidence threshold for classifier to fire an alert. Default: 0.55",
    )
    parser.add_argument(
        "--context-file",
        type=str,
        default=None,
        help="Path to scene_context.json produced by agent_mapper.py. Passed to the Customization Engine and Verification Gate.",
    )
    parser.add_argument(
        "--video-action-backend",
        default="none",
        choices=("none", "videomae", "x3d"),
        help="Optional weak temporal classifier run around detector-triggered moments.",
    )
    parser.add_argument(
        "--video-action-model",
        default="",
        help="Override video action model. Defaults: VideoMAE HF checkpoint or x3d_s.",
    )
    parser.add_argument(
        "--video-action-window-seconds",
        type=float,
        default=4.0,
        help="Seconds of recent video analyzed around a detector-triggered moment.",
    )
    parser.add_argument(
        "--video-action-frames",
        type=int,
        default=16,
        help="Number of sampled frames sent to the video action model.",
    )
    parser.add_argument(
        "--video-action-top-k",
        type=int,
        default=5,
        help="Number of action labels considered from the video action model.",
    )
    parser.add_argument(
        "--video-action-cooldown",
        type=float,
        default=2.0,
        help="Minimum seconds between video action model runs.",
    )
    parser.add_argument(
        "--video-action-device",
        default=None,
        help="Force video action device: cpu, mps, or cuda. X3D defaults to cpu on Mac.",
    )
    parser.add_argument(
        "--gate-provider",
        default="ollama",
        choices=("ollama", "local", "anthropic", "openrouter", "openai_compatible", "mock"),
        help="VLM provider for the Verification Gate. Default 'ollama' = the on-device "
             "model, offline and free. 'anthropic'/'openrouter' send frames off-device. "
             "'mock' confirms EVERYTHING without looking and needs ARGUS_ALLOW_MOCK_GATE=1.",
    )
    parser.add_argument(
        "--gate-model",
        default="",
        help="Override the gate model (empty = the provider default, e.g. gemma3:4b for ollama).",
    )
    parser.add_argument(
        "--gate-base-url",
        default="",
        help="Override the gate endpoint (e.g. a non-default Ollama host).",
    )
    parser.add_argument(
        "--gate-api-key-env",
        default="ANTHROPIC_API_KEY",
        help="Env var holding the gate API key (auto-switches to OPENROUTER_API_KEY for --gate-provider openrouter).",
    )
    return parser.parse_args()


def normalize_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ").replace("-", " ")


def normalize_threat_classes(raw_value: str) -> set[str]:
    return {normalize_label(item) for item in raw_value.split(",") if item.strip()}


def get_label_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def load_ultralytics_model(weights: str) -> LoadedModel:
    runner = YOLO(weights)
    return LoadedModel(
        runner=runner,
        kind="ultralytics",
        names=get_label_map(runner.names),
        source_path=weights,
    )


def load_yolov5_model(weights: str, yolov5_repo: str) -> LoadedModel:
    repo_path = str(Path(yolov5_repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    # Clear stale imports from failed Ultralytics legacy-load attempts.
    for module_name in list(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]
    importlib.invalidate_caches()
    hubconf = importlib.import_module("hubconf")

    runner = hubconf.custom(path=weights, autoshape=True, _verbose=False)
    return LoadedModel(
        runner=runner,
        kind="yolov5",
        names=get_label_map(runner.names),
        source_path=weights,
    )


def load_detection_model(weights: str, yolov5_repo: str, preferred_kind: str = "auto") -> LoadedModel:
    if preferred_kind == "yolov5":
        return load_yolov5_model(weights, yolov5_repo)
    if preferred_kind == "ultralytics":
        return load_ultralytics_model(weights)
    try:
        return load_ultralytics_model(weights)
    except ModuleNotFoundError as exc:
        if "models.yolo" not in str(exc):
            raise
        return load_yolov5_model(weights, yolov5_repo)


POSE_KEYPOINT_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}

# Personal-carry bag classes that count as a concealment destination (mirrors
# concealment.BAG_CLASSES). A shopping trolley is NOT one of these, so it stays safe.
CONCEALMENT_BAG_CLASSES = frozenset({"backpack", "handbag", "suitcase"})


def label_matches_any(label: str, configured_labels: set[str]) -> bool:
    normalized = normalize_label(label)
    if not configured_labels:
        return False

    default_aliases = {
        "gun": {"gun", "handgun", "pistol", "firearm", "rifle", "shotgun", "revolver"},
        "knife": {"knife", "blade", "dagger", "machete"},
        "person": {"person", "people", "man", "woman"},
    }
    expanded_terms: set[str] = set()
    for item in configured_labels:
        expanded_terms.add(item)
        expanded_terms.update(default_aliases.get(item, set()))

    return any(term == normalized or term in normalized or normalized in term for term in expanded_terms)


def extract_detections(result: Any, label_map: dict[int, str], threat_classes: set[str]) -> list[Detection]:
    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        label = label_map.get(class_id, str(class_id))
        is_threat = label_matches_any(label, threat_classes)
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                is_threat=is_threat,
                source_model="default",
            )
        )
    return detections


def extract_yolov5_detections(
    result: Any,
    label_map: dict[int, str],
    threat_classes: set[str],
) -> list[Detection]:
    detections: list[Detection] = []
    predictions = result.xyxy[0]
    if predictions is None:
        return detections

    for prediction in predictions.tolist():
        x1, y1, x2, y2, confidence, class_id = prediction[:6]
        label = label_map.get(int(class_id), str(int(class_id)))
        detections.append(
            Detection(
                label=label,
                confidence=float(confidence),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                is_threat=label_matches_any(label, threat_classes),
                source_model="default",
            )
        )
    return detections


def safe_keypoint(
    xy_points: Any,
    conf_points: Any,
    index: int,
    min_conf: float = 0.20,
) -> tuple[float, float] | None:
    if xy_points is None or conf_points is None:
        return None
    if index >= len(xy_points) or index >= len(conf_points):
        return None
    confidence = float(conf_points[index])
    if confidence < min_conf:
        return None
    x, y = xy_points[index]
    return float(x), float(y)


def point_distance(point_a: tuple[float, float] | None, point_b: tuple[float, float] | None) -> float | None:
    if point_a is None or point_b is None:
        return None
    ax, ay = point_a
    bx, by = point_b
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def compute_arm_extension_ratio(
    shoulder: tuple[float, float] | None,
    wrist: tuple[float, float] | None,
    bbox: tuple[int, int, int, int],
) -> float:
    distance = point_distance(shoulder, wrist)
    if distance is None:
        return 0.0
    return distance / max(bbox_diagonal(bbox), 1.0)


def extract_pose_people(model: LoadedModel, frame: Any, conf: float, imgsz: int) -> list[PosePersonState]:
    if model.kind != "ultralytics":
        raise RuntimeError("Pose inference requires an Ultralytics pose model.")

    results = model.runner.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
    result = results[0]
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or keypoints.xy is None:
        return []

    xy_batches = keypoints.xy.tolist()
    conf_batches = keypoints.conf.tolist() if keypoints.conf is not None else None
    pose_people: list[PosePersonState] = []
    timestamp = time.time()

    for index, box in enumerate(boxes):
        class_id = int(box.cls[0].item())
        label = model.names.get(class_id, str(class_id))
        if normalize_label(label) != "person":
            continue

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        xy_points = xy_batches[index] if index < len(xy_batches) else None
        conf_points = conf_batches[index] if conf_batches is not None and index < len(conf_batches) else None
        left_shoulder = safe_keypoint(xy_points, conf_points, POSE_KEYPOINT_INDEX["left_shoulder"])
        right_shoulder = safe_keypoint(xy_points, conf_points, POSE_KEYPOINT_INDEX["right_shoulder"])
        left_elbow = safe_keypoint(xy_points, conf_points, POSE_KEYPOINT_INDEX["left_elbow"])
        right_elbow = safe_keypoint(xy_points, conf_points, POSE_KEYPOINT_INDEX["right_elbow"])
        left_wrist = safe_keypoint(xy_points, conf_points, POSE_KEYPOINT_INDEX["left_wrist"])
        right_wrist = safe_keypoint(xy_points, conf_points, POSE_KEYPOINT_INDEX["right_wrist"])
        left_hip = safe_keypoint(xy_points, conf_points, POSE_KEYPOINT_INDEX["left_hip"])
        right_hip = safe_keypoint(xy_points, conf_points, POSE_KEYPOINT_INDEX["right_hip"])
        pose_people.append(
            PosePersonState(
                track_id=-1,
                bbox=(x1, y1, x2, y2),
                timestamp=timestamp,
                left_shoulder=left_shoulder,
                right_shoulder=right_shoulder,
                left_elbow=left_elbow,
                right_elbow=right_elbow,
                left_wrist=left_wrist,
                right_wrist=right_wrist,
                max_wrist_speed=0.0,
                max_wrist_accel=0.0,
                max_arm_extension_ratio=max(
                    compute_arm_extension_ratio(left_shoulder, left_wrist, (x1, y1, x2, y2)),
                    compute_arm_extension_ratio(right_shoulder, right_wrist, (x1, y1, x2, y2)),
                ),
                weapon_labels=[],
                left_hip=left_hip,
                right_hip=right_hip,
            )
        )

    return pose_people


def pose_people_to_sv_detections(pose_people: list[PosePersonState]) -> Any:
    """Build a supervision Detections (person boxes + track ids) for the zone monitor.

    Retail zones/concealment both ride on the pose pass, which already carries a track id,
    so we don't need a second tracker or to thread ids through the Detection dataclass.
    """
    import numpy as np
    import supervision as sv
    if not pose_people:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([list(p.bbox) for p in pose_people], dtype=float),
        class_id=np.zeros(len(pose_people), dtype=int),
        tracker_id=np.array([p.track_id for p in pose_people], dtype=int),
    )


def pose_people_to_concealment_frames(pose_people: list[PosePersonState], timestamp: float) -> list[Any]:
    """Adapt pose people into concealment.PoseFrame objects (keeps hips + track id)."""
    from cvti.retail.concealment import PoseFrame
    return [
        PoseFrame(
            track_id=p.track_id,
            timestamp=timestamp,
            keypoints={
                "left_shoulder": p.left_shoulder, "right_shoulder": p.right_shoulder,
                "left_wrist": p.left_wrist, "right_wrist": p.right_wrist,
                "left_hip": p.left_hip, "right_hip": p.right_hip,
            },
            bbox=tuple(float(v) for v in p.bbox),
        )
        for p in pose_people
    ]


def assign_pose_tracks(
    current_people: list[PosePersonState],
    previous_people: list[PosePersonState],
    next_track_id: int,
) -> tuple[list[PosePersonState], int]:
    unmatched_previous = {person.track_id: person for person in previous_people}
    for person in current_people:
        best_track_id: int | None = None
        best_ratio = 0.75
        for candidate in unmatched_previous.values():
            ratio = center_distance_ratio(person.bbox, candidate.bbox)
            if ratio < best_ratio:
                best_ratio = ratio
                best_track_id = candidate.track_id
        if best_track_id is None:
            person.track_id = next_track_id
            next_track_id += 1
        else:
            person.track_id = best_track_id
            unmatched_previous.pop(best_track_id, None)
    return current_people, next_track_id


def enrich_pose_people_with_history(
    current_people: list[PosePersonState],
    track_history: dict[int, deque[PosePersonState]],
) -> list[PosePersonState]:
    for person in current_people:
        history = track_history.get(person.track_id)
        if history:
            previous = history[-1]
            dt = max(person.timestamp - previous.timestamp, 1e-6)
            wrist_speeds: list[float] = []
            for current_point, previous_point in (
                (person.left_wrist, previous.left_wrist),
                (person.right_wrist, previous.right_wrist),
            ):
                speed_distance = point_distance(current_point, previous_point)
                if speed_distance is not None:
                    wrist_speeds.append(speed_distance / dt)
            person.max_wrist_speed = max(wrist_speeds, default=0.0)
            person.max_wrist_accel = (person.max_wrist_speed - previous.max_wrist_speed) / dt
        history = track_history.setdefault(person.track_id, deque(maxlen=6))
        history.append(person)
    # Prune departed tracks: pose ids are minted fresh and never reused, so a
    # key that isn't in this frame is dead forever. Without this, the dict
    # gained one entry per person who EVER walked past — tens of MB/day on a
    # busy camera, unbounded over weeks. (RAM audit 24 Aug, #1.)
    alive = {p.track_id for p in current_people}
    for tid in [t for t in track_history if t not in alive]:
        track_history.pop(tid, None)
    return current_people


class ViolenceTemporalGate:
    """Rolling-window majority vote — fires only when min_votes of the last window frames were active."""

    def __init__(self, window: int = 8, min_votes: int = 3) -> None:
        self._history: deque[bool] = deque(maxlen=window)
        self.min_votes = min_votes

    def update(self, assessment: ThreatAssessment) -> ThreatAssessment:
        self._history.append(assessment.active)
        if sum(self._history) >= self.min_votes:
            return assessment
        if assessment.active:
            return ThreatAssessment(
                active=False,
                title="CLEAR",
                level="none",
                reasons=assessment.reasons,
                weapon_labels=assessment.weapon_labels,
                explicit_labels=assessment.explicit_labels,
            )
        return assessment


def attach_weapons_to_pose_people(
    pose_people: list[PosePersonState],
    weapon_detections: list[Detection],
    hand_distance_ratio: float,
) -> list[PosePersonState]:
    for person in pose_people:
        matched_labels: set[str] = set()
        wrist_points = [person.left_wrist, person.right_wrist]
        for weapon in weapon_detections:
            weapon_center = bbox_center(weapon.bbox)
            weapon_inside_person = point_in_expanded_bbox(weapon_center, person.bbox)
            hand_near_weapon = any(
                wrist is not None and point_in_expanded_bbox(wrist, weapon.bbox, margin_ratio=hand_distance_ratio)
                for wrist in wrist_points
            )
            if weapon_inside_person or hand_near_weapon:
                matched_labels.add(normalize_label(weapon.label))
        person.weapon_labels = sorted(matched_labels)
    return pose_people


def predict_with_model(
    model: LoadedModel,
    frame: Any,
    conf: float,
    imgsz: int,
    threat_classes: set[str],
    source_model: str,
    use_tracking: bool = False,
) -> list[Detection]:
    if model.kind == "ultralytics":
        if use_tracking:
            results = model.runner.track(
                frame, conf=conf, imgsz=imgsz, verbose=False,
                tracker="bytetrack.yaml", persist=True,
            )
        else:
            results = model.runner.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        detections = extract_detections(results[0], model.names, threat_classes)
    elif model.kind == "yolov5":
        model.runner.conf = conf
        results = model.runner(frame, size=imgsz)
        detections = extract_yolov5_detections(results, model.names, threat_classes)
    else:
        raise RuntimeError(f"Unsupported model kind: {model.kind}")

    for detection in detections:
        detection.source_model = source_model
    return detections


def merge_detections(primary: list[Detection], secondary: list[Detection]) -> list[Detection]:
    merged = list(primary)
    for candidate in secondary:
        duplicate_found = False
        for existing in merged:
            if (
                normalize_label(existing.label) == normalize_label(candidate.label)
                and center_distance_ratio(existing.bbox, candidate.bbox) <= 0.12
            ):
                duplicate_found = True
                if candidate.confidence > existing.confidence:
                    existing.confidence = candidate.confidence
                    existing.bbox = candidate.bbox
                    existing.is_threat = existing.is_threat or candidate.is_threat
                    existing.source_model = candidate.source_model
                break
        if not duplicate_found:
            merged.append(candidate)
    return merged


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_diagonal(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return (width**2 + height**2) ** 0.5


def point_in_expanded_bbox(
    point: tuple[float, float],
    bbox: tuple[int, int, int, int],
    margin_ratio: float = 0.15,
) -> bool:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    px, py = point
    return (x1 - margin_x) <= px <= (x2 + margin_x) and (y1 - margin_y) <= py <= (y2 + margin_y)


def center_distance_ratio(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    ax, ay = bbox_center(bbox_a)
    bx, by = bbox_center(bbox_b)
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    scale = max(bbox_diagonal(bbox_a), bbox_diagonal(bbox_b), 1.0)
    return distance / scale


def summarize_labels(detections: list[Detection]) -> list[str]:
    return sorted({normalize_label(detection.label) for detection in detections})


def filter_detections_by_labels(
    detections: list[Detection],
    configured_labels: set[str],
) -> list[Detection]:
    return [detection for detection in detections if label_matches_any(detection.label, configured_labels)]


def bbox_area_ratio(bbox: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> float:
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, frame_width * frame_height)
    return bbox_area / frame_area


def bbox_touches_frame_edge(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    margin_ratio: float,
) -> bool:
    frame_height, frame_width = frame_shape[:2]
    margin_x = frame_width * margin_ratio
    margin_y = frame_height * margin_ratio
    x1, y1, x2, y2 = bbox
    return x1 <= margin_x or y1 <= margin_y or x2 >= (frame_width - margin_x) or y2 >= (frame_height - margin_y)


def weapon_is_attached_to_person(
    weapon: Detection,
    person_detections: list[Detection],
    pose_people: list[PosePersonState],
    hand_distance_ratio: float,
) -> bool:
    weapon_center = bbox_center(weapon.bbox)
    for person in person_detections:
        if point_in_expanded_bbox(weapon_center, person.bbox, margin_ratio=0.10):
            return True

    for person in pose_people:
        if any(
            wrist is not None and point_in_expanded_bbox(wrist, weapon.bbox, margin_ratio=hand_distance_ratio)
            for wrist in (person.left_wrist, person.right_wrist)
        ):
            return True
        if point_in_expanded_bbox(weapon_center, person.bbox, margin_ratio=0.10):
            return True

    return False


def validate_weapon_detections(
    detections: list[Detection],
    weapon_classes: set[str],
    person_classes: set[str],
    pose_people: list[PosePersonState],
    frame_shape: tuple[int, int, int],
    weapon_min_area_ratio: float,
    weapon_max_area_ratio: float,
    weapon_border_margin_ratio: float,
    weapon_hand_distance_ratio: float,
    allow_unattached_weapons: bool,
) -> list[Detection]:
    raw_weapon_detections = filter_detections_by_labels(detections, weapon_classes)
    person_detections = filter_detections_by_labels(detections, person_classes)
    validated: list[Detection] = []

    for weapon in raw_weapon_detections:
        area_ratio = bbox_area_ratio(weapon.bbox, frame_shape)
        if area_ratio < weapon_min_area_ratio or area_ratio > weapon_max_area_ratio:
            continue
        if bbox_touches_frame_edge(weapon.bbox, frame_shape, weapon_border_margin_ratio):
            continue
        if not allow_unattached_weapons and not weapon_is_attached_to_person(
            weapon,
            person_detections=person_detections,
            pose_people=pose_people,
            hand_distance_ratio=weapon_hand_distance_ratio,
        ):
            continue
        validated.append(weapon)

    return validated


def build_display_detections(
    detections: list[Detection],
    validated_weapon_detections: list[Detection],
    person_classes: set[str],
    threat_classes: set[str],
    show_all_detections: bool,
) -> list[Detection]:
    if show_all_detections:
        return detections

    visible: list[Detection] = []
    validated_weapon_keys = {
        (normalize_label(detection.label), detection.bbox, round(detection.confidence, 3), detection.source_model)
        for detection in validated_weapon_detections
    }

    for detection in detections:
        key = (normalize_label(detection.label), detection.bbox, round(detection.confidence, 3), detection.source_model)
        if key in validated_weapon_keys:
            visible.append(detection)
            continue
        if label_matches_any(detection.label, person_classes):
            visible.append(detection)
            continue
        if label_matches_any(detection.label, threat_classes):
            visible.append(detection)
            continue

    return visible


def format_detection_debug(detection: Detection) -> str:
    return (
        f"{normalize_label(detection.label)}@{detection.confidence:.2f}"
        f" bbox={detection.bbox} src={detection.source_model}"
    )


def build_weapon_debug_signature(weapon_detections: list[Detection]) -> str:
    if not weapon_detections:
        return ""
    ordered = sorted(
        weapon_detections,
        key=lambda detection: (
            normalize_label(detection.label),
            round(detection.confidence, 2),
            detection.bbox,
            detection.source_model,
        ),
    )
    return " | ".join(format_detection_debug(detection) for detection in ordered)


def build_pose_debug_signature(pose_people: list[PosePersonState]) -> str:
    if not pose_people:
        return ""
    ordered = sorted(pose_people, key=lambda person: person.track_id)
    parts = []
    for person in ordered:
        weapons = ",".join(person.weapon_labels) if person.weapon_labels else "-"
        parts.append(
            f"id={person.track_id} speed={person.max_wrist_speed:.1f}"
            f" arm={person.max_arm_extension_ratio:.2f} weapons={weapons}"
        )
    return " | ".join(parts)


def gate_assessment(
    assessment: ThreatAssessment,
    consecutive_threat_frames: int,
    min_threat_frames: int,
) -> ThreatAssessment:
    if not assessment.active or min_threat_frames <= 1 or consecutive_threat_frames >= min_threat_frames:
        return assessment

    remaining = max(0, min_threat_frames - consecutive_threat_frames)
    reasons = [f"Pending persistence: {consecutive_threat_frames}/{min_threat_frames} frames"] + assessment.reasons
    return ThreatAssessment(
        active=False,
        title="VERIFYING THREAT",
        level="pending",
        reasons=reasons,
        weapon_labels=assessment.weapon_labels,
        explicit_labels=assessment.explicit_labels,
    )


THEFT_OBJECT_CLASSES: frozenset[str] = frozenset({
    "backpack", "handbag", "suitcase", "bottle", "laptop", "cell phone", "book", "umbrella",
})


@dataclass
class TrackedObject:
    obj_id: int
    label: str
    bbox: tuple[int, int, int, int]
    origin_bbox: tuple[int, int, int, int]
    last_seen: float


@dataclass
class TheftPersonState:
    state: str = "IDLE"
    target_obj_id: int | None = None
    state_frames: int = 0
    acquire_frames: int = 0


class TheftDetector:
    """Per-person IDLE→APPROACH→ACQUIRE→DEPART theft state machine."""

    IDLE = "IDLE"
    APPROACH = "APPROACH"
    ACQUIRE = "ACQUIRE"
    DEPART = "DEPART"

    def __init__(
        self,
        acquire_frames: int = 8,
        depart_frames: int = 6,
        approach_ratio: float = 2.0,
        object_classes: frozenset[str] = THEFT_OBJECT_CLASSES,
        object_max_age: float = 2.0,
        debug: bool = False,
    ) -> None:
        self._person_states: dict[int, TheftPersonState] = {}
        self._objects: dict[int, TrackedObject] = {}
        self._next_obj_id = 1
        self.acquire_frames = acquire_frames
        self.depart_frames = depart_frames
        self.approach_ratio = approach_ratio
        self.object_classes = object_classes
        self.object_max_age = object_max_age
        self.debug = debug

    def _match_objects(self, detections: list[Detection], timestamp: float) -> None:
        relevant = [
            d for d in detections
            if normalize_label(d.label) not in {"person"}
            and normalize_label(d.label) in self.object_classes
        ]
        for det in relevant:
            best_id: int | None = None
            best_dist = 1.5
            for obj_id, obj in self._objects.items():
                if normalize_label(obj.label) != normalize_label(det.label):
                    continue
                dist = center_distance_ratio(obj.bbox, det.bbox)
                if dist < best_dist:
                    best_dist = dist
                    best_id = obj_id
            if best_id is not None:
                self._objects[best_id].bbox = det.bbox
                self._objects[best_id].last_seen = timestamp
            else:
                self._objects[self._next_obj_id] = TrackedObject(
                    obj_id=self._next_obj_id,
                    label=normalize_label(det.label),
                    bbox=det.bbox,
                    origin_bbox=det.bbox,
                    last_seen=timestamp,
                )
                self._next_obj_id += 1
        self._objects = {
            oid: obj for oid, obj in self._objects.items()
            if timestamp - obj.last_seen <= self.object_max_age
        }

    def _wrist_near_object(self, person: PosePersonState, obj: TrackedObject) -> bool:
        return any(
            wrist is not None and point_in_expanded_bbox(wrist, obj.bbox, margin_ratio=0.25)
            for wrist in (person.left_wrist, person.right_wrist)
        )

    def _nearest_object(self, person: PosePersonState) -> tuple[int, TrackedObject] | None:
        best: tuple[int, TrackedObject] | None = None
        best_dist = self.approach_ratio
        for obj_id, obj in self._objects.items():
            dist = center_distance_ratio(person.bbox, obj.bbox)
            if dist < best_dist:
                best_dist = dist
                best = (obj_id, obj)
        return best

    def update(
        self,
        pose_people: list[PosePersonState],
        detections: list[Detection],
        timestamp: float,
    ) -> ThreatAssessment:
        self._match_objects(detections, timestamp)
        alerts: list[tuple[int, str]] = []

        for person in pose_people:
            pid = person.track_id
            prev_state = self._person_states.get(pid, TheftPersonState()).state
            state = self._person_states.setdefault(pid, TheftPersonState())

            if state.state == self.IDLE:
                nearest = self._nearest_object(person)
                if nearest is not None:
                    state.state = self.APPROACH
                    state.target_obj_id = nearest[0]
                    state.state_frames = 1

            elif state.state == self.APPROACH:
                obj = self._objects.get(state.target_obj_id) if state.target_obj_id is not None else None
                if obj is None:
                    state.state, state.target_obj_id, state.state_frames = self.IDLE, None, 0
                elif self._wrist_near_object(person, obj):
                    state.state, state.state_frames, state.acquire_frames = self.ACQUIRE, 1, 1
                elif center_distance_ratio(person.bbox, obj.bbox) > self.approach_ratio:
                    state.state, state.target_obj_id, state.state_frames = self.IDLE, None, 0
                else:
                    state.state_frames += 1

            elif state.state == self.ACQUIRE:
                obj = self._objects.get(state.target_obj_id) if state.target_obj_id is not None else None
                if obj is None:
                    # Object vanished while hand was on it — strongest theft signal
                    state.state = self.DEPART
                    state.state_frames = self.depart_frames
                else:
                    if self._wrist_near_object(person, obj):
                        state.acquire_frames += 1
                        state.state_frames += 1
                    else:
                        if state.acquire_frames >= self.acquire_frames:
                            # Key fix: check if the OBJECT moved from its origin (taken with person)
                            # rather than checking if the person moved
                            obj_moved = center_distance_ratio(obj.bbox, obj.origin_bbox) > 0.4
                            person_left_area = center_distance_ratio(person.bbox, obj.origin_bbox) > 0.8
                            if obj_moved or person_left_area:
                                state.state, state.state_frames = self.DEPART, 1
                            else:
                                state.state, state.state_frames, state.acquire_frames = self.IDLE, 0, 0
                        else:
                            state.state, state.state_frames = self.APPROACH, 1

            elif state.state == self.DEPART:
                state.state_frames += 1
                if state.state_frames >= self.depart_frames:
                    obj = self._objects.get(state.target_obj_id) if state.target_obj_id is not None else None
                    obj_label = obj.label if obj else "unknown"
                    alerts.append((pid, obj_label))
                    state.state, state.target_obj_id = self.IDLE, None
                    state.state_frames, state.acquire_frames = 0, 0

            if self.debug and state.state != prev_state:
                obj = self._objects.get(state.target_obj_id) if state.target_obj_id is not None else None
                obj_info = f" target={obj.label}#{state.target_obj_id}" if obj else ""
                log.debug(f"[THEFT] person={pid} {prev_state} → {state.state}{obj_info}")

        active_ids = {p.track_id for p in pose_people}
        for pid in list(self._person_states.keys()):
            if pid not in active_ids:
                del self._person_states[pid]

        if alerts:
            pid, obj_label = alerts[0]
            return ThreatAssessment(
                active=True,
                title="POSSIBLE THEFT",
                level="warning",
                reasons=[
                    f"Person {pid} completed APPROACH→ACQUIRE→DEPART sequence",
                    f"Object: {obj_label}",
                ],
                weapon_labels=[],
                explicit_labels=[],
            )

        return ThreatAssessment(
            active=False, title="CLEAR", level="none",
            reasons=[], weapon_labels=[], explicit_labels=[],
        )

    @property
    def person_states(self) -> dict[int, TheftPersonState]:
        return self._person_states

    @property
    def tracked_objects(self) -> dict[int, TrackedObject]:
        return self._objects


def draw_theft_states(
    frame: Any,
    pose_people: list[PosePersonState],
    theft_detector: TheftDetector,
) -> Any:
    STATE_COLORS = {
        "IDLE":     (180, 180, 180),
        "APPROACH": (0,   220, 255),
        "ACQUIRE":  (0,   140, 255),
        "DEPART":   (0,   0,   255),
    }
    for person in pose_people:
        state = theft_detector.person_states.get(person.track_id)
        if state is None or state.state == "IDLE":
            continue
        x1, y1, x2, y2 = person.bbox
        color = STATE_COLORS.get(state.state, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = state.state
        if state.state == "ACQUIRE":
            label = f"ACQUIRE ({state.acquire_frames}f)"
        elif state.state == "DEPART":
            label = f"DEPART ({state.state_frames}f)"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    for obj in theft_detector.tracked_objects.values():
        ox1, oy1, ox2, oy2 = obj.bbox
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (255, 200, 0), 2)
        cv2.putText(frame, f"{obj.label}#{obj.obj_id}",
                    (ox1, max(15, oy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
    return frame


def run_classifier(
    classifier: Any,
    frame: Any,
    conf_threshold: float,
) -> ThreatAssessment:
    """Run the fine-tuned YOLOv8s-cls classifier on a frame and return a ThreatAssessment."""
    result = classifier.runner.predict(frame, verbose=False, imgsz=224)[0]
    pred_cls = result.names[int(result.probs.top1)]
    confidence = float(result.probs.top1conf)

    if pred_cls == "normal" or confidence < conf_threshold:
        return ThreatAssessment(
            active=False, title="CLEAR", level="none",
            reasons=[], weapon_labels=[], explicit_labels=[],
        )

    level = "high" if pred_cls == "violence" else "warning"
    title = "VIOLENCE SUSPECTED" if pred_cls == "violence" else "POSSIBLE THEFT"
    return ThreatAssessment(
        active=True,
        title=title,
        level=level,
        reasons=[f"Classifier: {pred_cls} ({confidence:.0%} confidence)"],
        weapon_labels=[],
        explicit_labels=[pred_cls],
    )


def choose_assessment(
    object_assessment: ThreatAssessment,
    violence_assessment: ThreatAssessment,
    theft_assessment: ThreatAssessment | None = None,
) -> ThreatAssessment:
    if violence_assessment.active or violence_assessment.level == "pending":
        return violence_assessment
    if theft_assessment is not None and theft_assessment.active:
        return theft_assessment
    return object_assessment


def assess_threat(
    detections: list[Detection],
    threat_classes: set[str],
    person_classes: set[str],
    validated_weapon_detections: list[Detection],
    assault_distance_ratio: float,
) -> ThreatAssessment:
    explicit_matches = filter_detections_by_labels(detections, threat_classes)
    person_detections = filter_detections_by_labels(detections, person_classes)
    weapon_detections = validated_weapon_detections

    if not detections:
        return ThreatAssessment(
            active=False,
            title="CLEAR",
            level="none",
            reasons=[],
            weapon_labels=[],
            explicit_labels=[],
        )

    armed_people: list[Detection] = []
    for person in person_detections:
        for weapon in weapon_detections:
            if point_in_expanded_bbox(bbox_center(weapon.bbox), person.bbox):
                armed_people.append(person)
                break

    possible_assault = False
    for armed_person in armed_people:
        for candidate in person_detections:
            if candidate is armed_person:
                continue
            if center_distance_ratio(armed_person.bbox, candidate.bbox) <= assault_distance_ratio:
                possible_assault = True
                break
        if possible_assault:
            break

    reasons: list[str] = []
    if explicit_matches:
        reasons.append(
            "Explicit threat class match: " + ", ".join(summarize_labels(explicit_matches))
        )
    if weapon_detections:
        reasons.append(
            "Dangerous object visible: " + ", ".join(summarize_labels(weapon_detections))
        )
    if armed_people:
        reasons.append("Weapon appears attached to a person")
    if possible_assault:
        reasons.append("Armed person is close to another person")

    if possible_assault:
        return ThreatAssessment(
            active=True,
            title="POSSIBLE ASSAULT",
            level="critical",
            reasons=reasons,
            weapon_labels=summarize_labels(weapon_detections),
            explicit_labels=summarize_labels(explicit_matches),
        )
    if armed_people:
        return ThreatAssessment(
            active=True,
            title="ARMED PERSON",
            level="critical",
            reasons=reasons,
            weapon_labels=summarize_labels(weapon_detections),
            explicit_labels=summarize_labels(explicit_matches),
        )
    if weapon_detections or explicit_matches:
        return ThreatAssessment(
            active=True,
            title="DANGEROUS OBJECT",
            level="warning",
            reasons=reasons,
            weapon_labels=summarize_labels(weapon_detections),
            explicit_labels=summarize_labels(explicit_matches),
        )
    return ThreatAssessment(
        active=False,
        title="CLEAR",
        level="none",
        reasons=[],
        weapon_labels=[],
        explicit_labels=[],
    )


def assess_violence(
    pose_people: list[PosePersonState],
    validated_weapon_detections: list[Detection],
    violence_distance_ratio: float,
    violence_wrist_speed: float,
    violence_arm_extension_ratio: float,
    weapon_hand_distance_ratio: float,
    violence_wrist_accel: float = 800.0,
) -> ThreatAssessment:
    if len(pose_people) < 2:
        return ThreatAssessment(
            active=False,
            title="CLEAR",
            level="none",
            reasons=[],
            weapon_labels=[],
            explicit_labels=[],
        )

    weapon_detections = validated_weapon_detections
    pose_people = attach_weapons_to_pose_people(pose_people, weapon_detections, weapon_hand_distance_ratio)

    close_pairs: list[tuple[PosePersonState, PosePersonState, float]] = []
    for index, left_person in enumerate(pose_people):
        for right_person in pose_people[index + 1 :]:
            distance_ratio = center_distance_ratio(left_person.bbox, right_person.bbox)
            if distance_ratio <= violence_distance_ratio:
                close_pairs.append((left_person, right_person, distance_ratio))

    if not close_pairs:
        return ThreatAssessment(
            active=False,
            title="CLEAR",
            level="none",
            reasons=[],
            weapon_labels=[],
            explicit_labels=[],
        )

    for left_person, right_person, distance_ratio in close_pairs:
        for attacker, target in ((left_person, right_person), (right_person, left_person)):
            if not attacker.weapon_labels:
                continue
            attacker_aggressive = (
                attacker.max_wrist_speed >= violence_wrist_speed
                or attacker.max_arm_extension_ratio >= violence_arm_extension_ratio
                or attacker.max_wrist_accel >= violence_wrist_accel
            )
            if not attacker_aggressive:
                continue
            if "knife" in attacker.weapon_labels:
                return ThreatAssessment(
                    active=True,
                    title="POSSIBLE STABBING",
                    level="critical",
                    reasons=[
                        f"Knife attached to person {attacker.track_id}",
                        f"Close target proximity ratio={distance_ratio:.2f}",
                        f"Wrist speed={attacker.max_wrist_speed:.1f}px/s arm={attacker.max_arm_extension_ratio:.2f}",
                    ],
                    weapon_labels=["knife"],
                    explicit_labels=[],
                )
            if "gun" in attacker.weapon_labels:
                return ThreatAssessment(
                    active=True,
                    title="POSSIBLE ARMED ASSAULT",
                    level="critical",
                    reasons=[
                        f"Gun attached to person {attacker.track_id}",
                        f"Close target proximity ratio={distance_ratio:.2f}",
                        f"Wrist speed={attacker.max_wrist_speed:.1f}px/s arm={attacker.max_arm_extension_ratio:.2f}",
                    ],
                    weapon_labels=["gun"],
                    explicit_labels=[],
                )

    for left_person, right_person, distance_ratio in close_pairs:
        peak_speed = max(left_person.max_wrist_speed, right_person.max_wrist_speed)
        peak_accel = max(left_person.max_wrist_accel, right_person.max_wrist_accel)
        if peak_speed >= violence_wrist_speed or peak_accel >= violence_wrist_accel:
            return ThreatAssessment(
                active=True,
                title="VIOLENCE SUSPECTED",
                level="warning",
                reasons=[
                    f"Two people in close contact ratio={distance_ratio:.2f}",
                    f"Peak wrist speed={peak_speed:.1f}px/s accel={peak_accel:.1f}px/s²",
                ],
                weapon_labels=summarize_labels(weapon_detections),
                explicit_labels=[],
            )

    # Close pair present but motion below aggression thresholds — monitoring state
    if close_pairs:
        _, _, distance_ratio = close_pairs[0]
        return ThreatAssessment(
            active=False,
            title="PROXIMITY ALERT",
            level="monitor",
            reasons=[f"Two persons in close proximity ratio={distance_ratio:.2f} — monitoring"],
            weapon_labels=[],
            explicit_labels=[],
        )

    return ThreatAssessment(
        active=False,
        title="CLEAR",
        level="none",
        reasons=[],
        weapon_labels=[],
        explicit_labels=[],
    )


def draw_detections(
    frame: Any,
    detections: list[Detection],
    fps: float,
    active_event: bool,
    assessment: ThreatAssessment,
) -> Any:
    annotated = frame.copy()
    for detection in detections:
        if label_matches_any(detection.label, set(assessment.weapon_labels)):
            color = (0, 0, 255)
        elif detection.is_threat:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        if detection.source_model != "default":
            label = f"{label} [{detection.source_model}]"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if assessment.active or assessment.level == "pending":
        if assessment.level == "critical":
            banner_color = (0, 0, 255)
        elif assessment.level == "pending":
            banner_color = (255, 140, 0)
        else:
            banner_color = (0, 165, 255)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 80), banner_color, -1)
        cv2.putText(
            annotated,
            assessment.title,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        detail_text = "; ".join(assessment.reasons[:2]) if assessment.reasons else "Threat rule triggered"
        cv2.putText(
            annotated,
            detail_text[:90],
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    elif active_event:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 60), (0, 165, 255), -1)
        cv2.putText(
            annotated,
            "RECORDING EVENT",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, annotated.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def open_capture(source: int | str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")
    return capture


def main() -> None:
    # Entrypoint: configure logging before anything can fail.
    from cvti.logging_setup import setup_logging
    setup_logging(component="argus-detect")
    args = parse_args()
    source = normalize_source(args.source)
    threat_classes = normalize_threat_classes(args.threat_classes)
    person_classes = normalize_threat_classes(args.person_classes)
    weapon_classes = normalize_threat_classes(args.weapon_classes)
    output_root = Path(args.save_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading model...")
    default_model = load_detection_model(args.weights, args.yolov5_repo)
    person_model = load_detection_model(args.person_weights, args.yolov5_repo) if args.person_weights else None
    weapon_model = (
        load_detection_model(args.weapon_weights, args.yolov5_repo, preferred_kind=args.weapon_loader)
        if args.weapon_weights
        else None
    )
    pose_model = load_ultralytics_model(args.pose_weights) if args.pose_weights else None
    classifier_model = load_ultralytics_model(args.classifier_weights) if args.classifier_weights else None
    label_map = default_model.names

    log.info(f"Configured threat classes: {sorted(threat_classes)}")
    log.info(f"Configured person classes: {sorted(person_classes)}")
    log.info(f"Configured weapon classes: {sorted(weapon_classes)}")
    log.info(f"Available model classes: {list(label_map.values())[:15]}{' ...' if len(label_map) > 15 else ''}")
    log.info(f"Loaded default model from: {default_model.source_path} ({default_model.kind})")
    if person_model is not None:
        log.info(f"Loaded dedicated person model: {person_model.source_path} ({person_model.kind})")
    if weapon_model is not None:
        log.info(f"Loaded dedicated weapon model: {weapon_model.source_path} ({weapon_model.kind})")
    if pose_model is not None:
        log.info(f"Loaded pose model: {pose_model.source_path} ({pose_model.kind})")
    if classifier_model is not None:
        log.info(f"Loaded classifier: {classifier_model.source_path} (conf≥{args.classifier_conf})")
    tracking_enabled = not args.no_track and default_model.kind == "ultralytics"
    log.info(f"ByteTrack person tracking: {'ON' if tracking_enabled else 'OFF (use --no-track to disable)'}")

    capture = open_capture(source)
    source_name = str(source)
    fps_from_capture = capture.get(cv2.CAP_PROP_FPS)
    recorder = EventRecorder(
        output_root=output_root,
        clip_seconds=args.clip_seconds,
        fps_fallback=15.0,
    )

    last_event_time = 0.0
    frame_count = 0
    time_anchor = time.time()
    threat_visible_last_frame = False
    video_writer: cv2.VideoWriter | None = None
    object_threat_frames = 0
    consecutive_threat_frames = 0
    last_weapon_debug_signature = ""
    last_violence_debug_signature = ""
    previous_pose_people: list[PosePersonState] = []
    pose_track_history: dict[int, deque[PosePersonState]] = {}
    next_pose_track_id = 1
    violence_gate = ViolenceTemporalGate(
        window=args.violence_gate_window,
        min_votes=args.violence_gate_votes,
    )
    theft_detector = TheftDetector(
        acquire_frames=args.theft_acquire_frames,
        depart_frames=args.theft_depart_frames,
        approach_ratio=args.theft_approach_ratio,
        debug=args.debug_theft,
    )
    customization_engine = CustomizationEngine(
        args.config,
        baseline_path=None if args.no_baseline else args.baseline_config,
    )
    # Phase 1: queue ALL matching candidates (dedup + throttle) instead of only
    # verifying candidate_alerts[0]. Lets concurrent threats each get verified.
    alert_queue = AlertQueue()
    # Phase 4: rolling buffer of recent frames so the gate can be sent the
    # clearest / motion-peak evidence for a rule, not just the flagged frame.
    _fps_buf = fps_from_capture if fps_from_capture and fps_from_capture > 0 else 25.0
    gate_frame_buffer: deque = deque(maxlen=max(2, int(_fps_buf * 1.5)))
    video_action_runtime = None
    if args.video_action_backend != "none":
        fps_for_video_action = fps_from_capture if fps_from_capture and fps_from_capture > 0 else 30.0
        video_action_runtime = build_video_action_runtime(
            backend=args.video_action_backend,
            model_name=args.video_action_model,
            fps=fps_for_video_action,
            window_seconds=args.video_action_window_seconds,
            frame_count=args.video_action_frames,
            top_k=args.video_action_top_k,
            cooldown_seconds=args.video_action_cooldown,
            device=args.video_action_device,
            verbose=True,
        )
        log.info(f"[VideoAction] {args.video_action_backend} ON — "
            f"{args.video_action_frames} frames over {args.video_action_window_seconds:.1f}s event windows")

    # Retail action layer (optional): shelf zones + concealment, both riding the pose pass.
    zone_monitor = None
    concealment_detector = None
    if args.zones:
        from cvti.retail.zones import RetailZoneMonitor, load_zone_config
        zone_monitor = RetailZoneMonitor(load_zone_config(args.zones))
        log.info(f"[Zones] Loaded {len(zone_monitor.zones)} zone(s): {', '.join(z.name for z in zone_monitor.zones)}")
    if args.concealment:
        from cvti.retail.concealment import ConcealmentDetector
        concealment_detector = ConcealmentDetector()
        log.info("[Concealment] Pose-based concealment detector ON")
    if (zone_monitor is not None or concealment_detector is not None) and pose_model is None:
        log.warning("[Warning] --zones/--concealment need a pose model; pass --pose-weights yolov8n-pose.pt")

    scene_context: dict | None = None
    if args.context_file:
        context_path = Path(args.context_file)
        if context_path.exists():
            scene_context = json.loads(context_path.read_text())
            log.info(f"[SceneContext] Loaded: {scene_context.get('environment_type', 'unknown')} — {scene_context.get('scene_description', '')[:80]}")
        else:
            log.warning(f"[SceneContext] Warning: context file not found: {context_path}")

    from cvti.verification.gate import MOCK_GATE_BANNER, assert_engine_gate_allowed
    mock_gate = assert_engine_gate_allowed(args.gate_provider)

    verification_gate = VerificationGate(
        provider=args.gate_provider,
        model=args.gate_model,
        base_url=args.gate_base_url,
        api_key_env=args.gate_api_key_env,
        save_dir=output_root / "gate" if args.gate_provider != "mock" else None,
    )
    if args.gate_provider != "mock":
        log.info(f"[VerificationGate] Provider: {args.gate_provider} ({verification_gate.model}) — alerts confirmed by VLM before recording.")
    else:
        log.info(f"[VerificationGate] *** {MOCK_GATE_BANNER} *** provider=mock: all alerts auto-confirmed, "
              "nothing is verified. Use --gate-provider ollama/openrouter/anthropic for real verification.")
        assert mock_gate  # only reachable via the explicit opt-in above

    log.info("Starting inference loop. Press 'q' to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                log.error("Stream ended or frame could not be read.")
                break
            current_frame_index = frame_count
            gate_frame_buffer.append(frame)
            if video_action_runtime is not None:
                video_action_runtime.add_frame(frame, frame_index=current_frame_index)

            # Only one model should own the ByteTrack state at a time.
            # If a dedicated person model is set, it handles tracking.
            # The default model falls back to plain predict to avoid two trackers fighting.
            detections = predict_with_model(
                default_model,
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                threat_classes=threat_classes,
                source_model="default",
                use_tracking=tracking_enabled and person_model is None,
            )
            if person_model is not None:
                person_detections = predict_with_model(
                    person_model,
                    frame,
                    conf=args.person_conf,
                    imgsz=args.imgsz,
                    threat_classes=threat_classes,
                    source_model="person",
                    use_tracking=tracking_enabled,
                )
                detections = merge_detections(detections, person_detections)
            if weapon_model is not None:
                weapon_detections = predict_with_model(
                    weapon_model,
                    frame,
                    conf=args.weapon_conf,
                    imgsz=args.imgsz,
                    threat_classes=threat_classes,
                    source_model="weapon",
                )
                detections = merge_detections(detections, weapon_detections)

            pose_people: list[PosePersonState] = []
            if pose_model is not None:
                pose_people = extract_pose_people(
                    pose_model,
                    frame,
                    conf=args.pose_conf,
                    imgsz=args.imgsz,
                )
                pose_people, next_pose_track_id = assign_pose_tracks(
                    pose_people,
                    previous_people=previous_pose_people,
                    next_track_id=next_pose_track_id,
                )
                pose_people = enrich_pose_people_with_history(pose_people, pose_track_history)
                previous_pose_people = list(pose_people)

            raw_object_assessment = assess_threat(
                detections=detections,
                threat_classes=threat_classes,
                person_classes=person_classes,
                validated_weapon_detections=validate_weapon_detections(
                    detections=detections,
                    weapon_classes=weapon_classes,
                    person_classes=person_classes,
                    pose_people=pose_people,
                    frame_shape=frame.shape,
                    weapon_min_area_ratio=args.weapon_min_area_ratio,
                    weapon_max_area_ratio=args.weapon_max_area_ratio,
                    weapon_border_margin_ratio=args.weapon_border_margin_ratio,
                    weapon_hand_distance_ratio=args.weapon_hand_distance_ratio,
                    allow_unattached_weapons=args.allow_unattached_weapons,
                ),
                assault_distance_ratio=args.assault_distance_ratio,
            )
            validated_weapon_detections = validate_weapon_detections(
                detections=detections,
                weapon_classes=weapon_classes,
                person_classes=person_classes,
                pose_people=pose_people,
                frame_shape=frame.shape,
                weapon_min_area_ratio=args.weapon_min_area_ratio,
                weapon_max_area_ratio=args.weapon_max_area_ratio,
                weapon_border_margin_ratio=args.weapon_border_margin_ratio,
                weapon_hand_distance_ratio=args.weapon_hand_distance_ratio,
                allow_unattached_weapons=args.allow_unattached_weapons,
            )
            raw_violence_assessment = assess_violence(
                pose_people=pose_people,
                validated_weapon_detections=validated_weapon_detections,
                violence_distance_ratio=args.violence_distance_ratio,
                violence_wrist_speed=args.violence_wrist_speed,
                violence_arm_extension_ratio=args.violence_arm_extension_ratio,
                weapon_hand_distance_ratio=args.weapon_hand_distance_ratio,
                violence_wrist_accel=args.violence_wrist_accel,
            )

            if raw_object_assessment.active:
                object_threat_frames += 1
            else:
                object_threat_frames = 0
            _clear = ThreatAssessment(active=False, title="CLEAR", level="none",
                                      reasons=[], weapon_labels=[], explicit_labels=[])
            object_assessment = (
                gate_assessment(raw_object_assessment,
                                consecutive_threat_frames=object_threat_frames,
                                min_threat_frames=max(1, args.min_threat_frames))
                if args.mode in ("all", "weapons") else _clear
            )
            violence_assessment = (
                violence_gate.update(raw_violence_assessment)
                if args.mode in ("all", "violence") else _clear
            )
            theft_assessment = (
                theft_detector.update(
                    pose_people=pose_people,
                    detections=detections,
                    timestamp=time.time() - time_anchor,
                )
                if args.mode in ("all", "theft") else _clear
            )

            # Classifier — overrides pose heuristic when loaded
            if classifier_model is not None:
                cls_assessment = run_classifier(classifier_model, frame, args.classifier_conf)
                if cls_assessment.active:
                    if "theft" in cls_assessment.explicit_labels:
                        theft_assessment = cls_assessment
                    else:
                        violence_assessment = cls_assessment

            assessment = choose_assessment(object_assessment, violence_assessment, theft_assessment)
            threat_detected = assessment.active

            # Customization Engine + Verification Gate. Runs when there are any
            # rules — including the always-on critical baseline, so weapons/
            # violence are gated even with no customer --config.
            if customization_engine.has_rules():
                ts_now = time.time() - time_anchor
                raw_events = assessments_to_events(
                    object_assessment, violence_assessment, theft_assessment,
                    timestamp=ts_now,
                    theft_detector=theft_detector,
                )
                should_run_video_action = (
                    video_action_runtime is not None
                    and (
                        object_assessment.active
                        or violence_assessment.active
                        or theft_assessment.active
                    )
                )
                # Retail action layer events (zones + concealment) ride the same pose pass
                # and merge into the same event stream the user's rules evaluate.
                if zone_monitor is not None:
                    zone_states = zone_monitor.update(pose_people_to_sv_detections(pose_people), ts_now)
                    raw_events += zone_states_to_events(zone_states, ts_now)
                if concealment_detector is not None:
                    bag_bboxes = [d.bbox for d in detections
                                  if normalize_label(d.label) in CONCEALMENT_BAG_CLASSES]
                    conceal = concealment_detector.update(
                        pose_people_to_concealment_frames(pose_people, ts_now),
                        ts_now,
                        bag_bboxes=bag_bboxes or None,
                    )
                    concealment_events = concealment_to_events(conceal, ts_now)
                    raw_events += concealment_events
                    should_run_video_action = should_run_video_action or any(event.active for event in concealment_events)
                if should_run_video_action and video_action_runtime is not None:
                    try:
                        raw_events += video_action_runtime.analyze_event(
                            center_frame_index=current_frame_index,
                            timestamp=ts_now,
                        )
                    except Exception as exc:  # noqa: BLE001 - optional weak signal must not kill detector
                        log.error(f"[VideoAction error] {str(exc)[:140]}", exc_info=True)
                candidate_alerts = customization_engine.evaluate(raw_events, scene_context=scene_context)
                # Phase 1: enqueue EVERY matching candidate (deduped by rule/track/
                # object within a cooldown), not just candidate_alerts[0]. Concurrent
                # threats (e.g. a weapon AND a theft) are then each verified over
                # successive frames instead of the top one starving the rest.
                for candidate in candidate_alerts:
                    alert_queue.add(QueuedAlert(
                        camera_id=source_name,
                        rule_name=candidate.rule_name,
                        priority=candidate.priority,
                        title=candidate.title,
                        timestamp=ts_now,
                        track_id=candidate.person_id,
                        object_label=candidate.object_label,
                        payload=candidate,
                    ))
                # Verify a bounded, priority-ordered burst per frame so a slow gate
                # can't stall the loop; the rest stay queued for the next frames.
                confirmed_any = False
                verified_this_frame = False
                for queued in alert_queue.drain(max_per_drain=args.max_alerts_per_frame):
                    candidate = queued.payload
                    verified_this_frame = True
                    # Phase 4: send the gate per-rule evidence (1 clearest frame for
                    # weapons; several motion-peak frames for violence/concealment)
                    # instead of the single flagged frame.
                    evidence_frames, sel_meta = select_evidence_frames(
                        list(gate_frame_buffer), candidate.rule_name)
                    try:
                        gate_result = verification_gate.verify(
                            evidence_frames or frame, candidate, scene_context)
                    except Exception as exc:  # noqa: BLE001 - a transient gate/API error must not kill the run
                        log.error(f"[gate error] {candidate.rule_name} — {str(exc)[:140]} (alert held, not raised)", exc_info=True)
                        continue
                    if gate_result is not None and gate_result.confirmed:
                        log.info(f"[CONFIRMED] {candidate.rule_name} ({candidate.priority.upper()}) "
                            f"— {candidate.title}"
                            + (f" [obj: {candidate.object_label}]" if candidate.object_label else "")
                            + f" | confidence={gate_result.confidence:.2f} | {gate_result.reason}"
                            + f" | frames={sel_meta['count']}({sel_meta['strategy']})")
                        confirmed_any = True
                    elif gate_result is not None:
                        log.info(f"[REJECTED]  {candidate.rule_name} — {gate_result.reason}")
                # Only let the config gate govern threat_detected when we actually
                # verified something this frame; otherwise keep the baseline signal.
                if verified_this_frame:
                    threat_detected = confirmed_any

            weapon_detections_for_debug = validated_weapon_detections
            if args.debug_weapon:
                current_weapon_debug_signature = build_weapon_debug_signature(weapon_detections_for_debug)
                if current_weapon_debug_signature != last_weapon_debug_signature:
                    if current_weapon_debug_signature:
                        log.info(f"Weapon detections: {current_weapon_debug_signature}")
                    elif last_weapon_debug_signature:
                        log.info("Weapon detections cleared")
                    last_weapon_debug_signature = current_weapon_debug_signature

            if args.debug_violence and pose_model is not None:
                current_violence_debug_signature = (
                    f"{assessment.title} :: {build_pose_debug_signature(pose_people)}"
                    if pose_people
                    else ""
                )
                if current_violence_debug_signature != last_violence_debug_signature:
                    if current_violence_debug_signature:
                        log.info(f"Violence debug: {current_violence_debug_signature}")
                    elif last_violence_debug_signature:
                        log.info("Violence debug cleared")
                    last_violence_debug_signature = current_violence_debug_signature

            frame_count += 1
            elapsed = max(time.time() - time_anchor, 1e-6)
            fps = frame_count / elapsed
            active_event = recorder.writer is not None
            display_detections = build_display_detections(
                detections=detections,
                validated_weapon_detections=validated_weapon_detections,
                person_classes=person_classes,
                threat_classes=threat_classes,
                show_all_detections=args.show_all_detections,
            )
            annotated = draw_detections(
                frame,
                display_detections,
                fps=fps,
                active_event=active_event,
                assessment=assessment,
            )
            if args.debug_theft:
                annotated = draw_theft_states(annotated, pose_people, theft_detector)

            new_threat_event = threat_detected and not threat_visible_last_frame
            if (
                new_threat_event
                and recorder.writer is None
                and (time.time() - last_event_time) >= args.cooldown
            ):
                event_dir = recorder.start(
                    frame=annotated,
                    detections=detections,
                    assessment=assessment,
                    source=source_name,
                    fps=fps_from_capture,
                )
                last_event_time = time.time()
                log.info(f"Threat event saved to: {event_dir}")
            threat_visible_last_frame = threat_detected

            recorder.write(annotated)
            if recorder.should_stop():
                recorder.stop()

            if args.save_video:
                if video_writer is None:
                    h, w = annotated.shape[:2]
                    video_writer = cv2.VideoWriter(
                        args.save_video,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(fps_from_capture, 1.0),
                        (w, h),
                    )
                video_writer.write(annotated)

            if args.show:
                cv2.imshow("CV Threat Intelligence POC", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        recorder.stop()
        if video_writer is not None:
            video_writer.release()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
