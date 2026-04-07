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

import cv2
import torch
from ultralytics import YOLO

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
    max_arm_extension_ratio: float
    weapon_labels: list[str]


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
        default="yolov8n.pt",
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
        default="yolov8n-pose.pt",
        help="Pose model weights used for violence heuristics. Set to empty to disable pose-based violence logic.",
    )
    parser.add_argument(
        "--threat-classes",
        default="person",
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
}


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
                max_arm_extension_ratio=max(
                    compute_arm_extension_ratio(left_shoulder, left_wrist, (x1, y1, x2, y2)),
                    compute_arm_extension_ratio(right_shoulder, right_wrist, (x1, y1, x2, y2)),
                ),
                weapon_labels=[],
            )
        )

    return pose_people


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
        history = track_history.setdefault(person.track_id, deque(maxlen=6))
        history.append(person)
    return current_people


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
) -> list[Detection]:
    if model.kind == "ultralytics":
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


def choose_assessment(
    object_assessment: ThreatAssessment,
    violence_assessment: ThreatAssessment,
) -> ThreatAssessment:
    if violence_assessment.active or violence_assessment.level == "pending":
        return violence_assessment
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
        if (
            left_person.max_wrist_speed >= violence_wrist_speed
            or right_person.max_wrist_speed >= violence_wrist_speed
        ):
            return ThreatAssessment(
                active=True,
                title="VIOLENCE SUSPECTED",
                level="warning",
                reasons=[
                    f"Two people in close contact ratio={distance_ratio:.2f}",
                    f"Peak wrist speed={max(left_person.max_wrist_speed, right_person.max_wrist_speed):.1f}px/s",
                ],
                weapon_labels=summarize_labels(weapon_detections),
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
    args = parse_args()
    source = normalize_source(args.source)
    threat_classes = normalize_threat_classes(args.threat_classes)
    person_classes = normalize_threat_classes(args.person_classes)
    weapon_classes = normalize_threat_classes(args.weapon_classes)
    output_root = Path(args.save_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    default_model = load_detection_model(args.weights, args.yolov5_repo)
    person_model = load_detection_model(args.person_weights, args.yolov5_repo) if args.person_weights else None
    weapon_model = (
        load_detection_model(args.weapon_weights, args.yolov5_repo, preferred_kind=args.weapon_loader)
        if args.weapon_weights
        else None
    )
    pose_model = load_ultralytics_model(args.pose_weights) if args.pose_weights else None
    label_map = default_model.names

    print(f"Configured threat classes: {sorted(threat_classes)}")
    print(f"Configured person classes: {sorted(person_classes)}")
    print(f"Configured weapon classes: {sorted(weapon_classes)}")
    print(f"Available model classes: {list(label_map.values())[:15]}{' ...' if len(label_map) > 15 else ''}")
    print(f"Loaded default model from: {default_model.source_path} ({default_model.kind})")
    if person_model is not None:
        print(f"Loaded dedicated person model: {person_model.source_path} ({person_model.kind})")
    if weapon_model is not None:
        print(f"Loaded dedicated weapon model: {weapon_model.source_path} ({weapon_model.kind})")
    if pose_model is not None:
        print(f"Loaded pose model: {pose_model.source_path} ({pose_model.kind})")

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
    object_threat_frames = 0
    violence_threat_frames = 0
    consecutive_threat_frames = 0
    last_weapon_debug_signature = ""
    last_violence_debug_signature = ""
    previous_pose_people: list[PosePersonState] = []
    pose_track_history: dict[int, deque[PosePersonState]] = {}
    next_pose_track_id = 1

    print("Starting inference loop. Press 'q' to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Stream ended or frame could not be read.")
                break

            detections = predict_with_model(
                default_model,
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                threat_classes=threat_classes,
                source_model="default",
            )
            if person_model is not None:
                person_detections = predict_with_model(
                    person_model,
                    frame,
                    conf=args.person_conf,
                    imgsz=args.imgsz,
                    threat_classes=threat_classes,
                    source_model="person",
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
            )

            if raw_object_assessment.active:
                object_threat_frames += 1
            else:
                object_threat_frames = 0
            if raw_violence_assessment.active:
                violence_threat_frames += 1
            else:
                violence_threat_frames = 0

            object_assessment = gate_assessment(
                raw_object_assessment,
                consecutive_threat_frames=object_threat_frames,
                min_threat_frames=max(1, args.min_threat_frames),
            )
            violence_assessment = gate_assessment(
                raw_violence_assessment,
                consecutive_threat_frames=violence_threat_frames,
                min_threat_frames=max(1, args.violence_min_frames),
            )
            assessment = choose_assessment(object_assessment, violence_assessment)
            threat_detected = assessment.active

            weapon_detections_for_debug = validated_weapon_detections
            if args.debug_weapon:
                current_weapon_debug_signature = build_weapon_debug_signature(weapon_detections_for_debug)
                if current_weapon_debug_signature != last_weapon_debug_signature:
                    if current_weapon_debug_signature:
                        print(f"Weapon detections: {current_weapon_debug_signature}")
                    elif last_weapon_debug_signature:
                        print("Weapon detections cleared")
                    last_weapon_debug_signature = current_weapon_debug_signature

            if args.debug_violence and pose_model is not None:
                current_violence_debug_signature = (
                    f"{assessment.title} :: {build_pose_debug_signature(pose_people)}"
                    if pose_people
                    else ""
                )
                if current_violence_debug_signature != last_violence_debug_signature:
                    if current_violence_debug_signature:
                        print(f"Violence debug: {current_violence_debug_signature}")
                    elif last_violence_debug_signature:
                        print("Violence debug cleared")
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
                print(f"Threat event saved to: {event_dir}")
            threat_visible_last_frame = threat_detected

            recorder.write(annotated)
            if recorder.should_stop():
                recorder.stop()

            if args.show:
                cv2.imshow("CV Threat Intelligence POC", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        recorder.stop()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
