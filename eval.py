"""eval.py — Baseline evaluation for the CV threat intelligence pipeline.

Runs headless inference on a folder of labeled video clips and computes
per-class metrics (precision, recall, false-positive rate, F1).

Ground-truth CSV format (--ground-truth):
    clip,threat_class,has_threat
    fight_001.mp4,violence,1
    street_walk.mp4,violence,0
    knife_demo.mp4,weapons,1
    empty_room.mp4,weapons,0

Threat classes (suggested): violence, weapons, theft, loitering
has_threat: 1 = threat present in clip, 0 = negative sample

Usage example:
    python eval.py \\
        --clips-dir data/test_clips \\
        --ground-truth data/ground_truth.csv \\
        --weapon-weights models/weapon_best.pt \\
        --weapon-loader yolov5 \\
        --report-out reports/baseline.json

To generate a ground-truth CSV template from a clips folder:
    python eval.py --clips-dir data/test_clips --generate-template
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

# Pull shared logic from detector.py
sys.path.insert(0, str(Path(__file__).parent))
from detector import (
    PosePersonState,
    ThreatAssessment,
    assess_threat,
    assess_violence,
    assign_pose_tracks,
    build_display_detections,
    choose_assessment,
    draw_detections,
    enrich_pose_people_with_history,
    extract_pose_people,
    gate_assessment,
    load_detection_model,
    load_ultralytics_model,
    merge_detections,
    normalize_threat_classes,
    predict_with_model,
    validate_weapon_detections,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthRow:
    clip: str          # filename, relative to --clips-dir
    threat_class: str  # violence | weapons | theft | loitering | ...
    has_threat: bool   # ground truth label


@dataclass
class ClipResult:
    clip: str
    threat_class: str
    has_threat: bool       # ground truth
    detected: bool         # did the pipeline fire?
    threat_frames: int     # frames where assessment.active == True
    total_frames: int
    threats_seen: list[str]  # distinct threat titles triggered
    elapsed_seconds: float

    @property
    def tp(self) -> bool:
        return self.has_threat and self.detected

    @property
    def fp(self) -> bool:
        return not self.has_threat and self.detected

    @property
    def fn(self) -> bool:
        return self.has_threat and not self.detected

    @property
    def tn(self) -> bool:
        return not self.has_threat and not self.detected


@dataclass
class ClassMetrics:
    threat_class: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def fpr(self) -> float | None:
        denom = self.fp + self.tn
        return self.fp / denom if denom else None

    @property
    def f1(self) -> float | None:
        p = self.precision
        r = self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict[str, Any]:
        def fmt(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "n/a"
        return {
            "threat_class": self.threat_class,
            "tp": self.tp, "fp": self.fp,
            "fn": self.fn, "tn": self.tn,
            "precision": fmt(self.precision),
            "recall": fmt(self.recall),
            "fpr": fmt(self.fpr),
            "f1": fmt(self.f1),
        }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline evaluation for the CV threat intelligence pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clips-dir",
        required=True,
        help="Directory containing test video clips.",
    )
    parser.add_argument(
        "--ground-truth",
        default="",
        help="Path to ground-truth CSV (clip, threat_class, has_threat).",
    )
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="Print a ground-truth CSV template from --clips-dir and exit.",
    )
    # Model args mirror detector.py
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--person-weights", default="")
    parser.add_argument("--weapon-weights", default="")
    parser.add_argument("--pose-weights", default="yolov8n-pose.pt")
    parser.add_argument(
        "--weapon-loader",
        default="auto",
        choices=("auto", "ultralytics", "yolov5"),
    )
    parser.add_argument("--yolov5-repo", default="external/yolov5")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--weapon-conf", type=float, default=0.35)
    parser.add_argument("--pose-conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threat-classes", default="person")
    parser.add_argument("--person-classes", default="person")
    parser.add_argument("--weapon-classes", default="knife,gun")
    parser.add_argument("--assault-distance-ratio", type=float, default=1.2)
    parser.add_argument("--min-threat-frames", type=int, default=3)
    parser.add_argument("--violence-min-frames", type=int, default=4)
    parser.add_argument("--violence-distance-ratio", type=float, default=1.1)
    parser.add_argument("--violence-wrist-speed", type=float, default=120.0)
    parser.add_argument("--violence-arm-extension-ratio", type=float, default=0.35)
    parser.add_argument("--weapon-hand-distance-ratio", type=float, default=0.20)
    parser.add_argument("--weapon-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--weapon-max-area-ratio", type=float, default=0.18)
    parser.add_argument("--weapon-border-margin-ratio", type=float, default=0.03)
    parser.add_argument(
        "--allow-unattached-weapons",
        action="store_true",
    )
    parser.add_argument(
        "--min-detect-frames",
        type=int,
        default=1,
        help=(
            "Minimum number of frames with an active threat before the clip is "
            "counted as 'detected'. Higher values reduce FPs at the cost of recall."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap frames per clip for speed. 0 = no cap.",
    )
    parser.add_argument(
        "--report-out",
        default="",
        help="Optional path to save a JSON report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-clip results as they complete.",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Write annotated output videos to reports/annotated/<clip_name>.",
    )
    parser.add_argument(
        "--annotated-dir",
        default="reports/annotated",
        help="Directory to save annotated videos when --save-annotated is set.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path: str) -> list[GroundTruthRow]:
    rows: list[GroundTruthRow] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line in reader:
            rows.append(
                GroundTruthRow(
                    clip=line["clip"].strip(),
                    threat_class=line["threat_class"].strip().lower(),
                    has_threat=line["has_threat"].strip() in ("1", "true", "yes"),
                )
            )
    return rows


def generate_template(clips_dir: str) -> None:
    clips_path = Path(clips_dir)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    clips = sorted(
        p.name for p in clips_path.iterdir()
        if p.suffix.lower() in video_extensions
    )
    print("clip,threat_class,has_threat")
    for clip in clips:
        print(f"{clip},violence,")  # user fills in threat_class and has_threat
    if not clips:
        print("# No video files found in", clips_dir, file=sys.stderr)


# ---------------------------------------------------------------------------
# Single-clip inference
# ---------------------------------------------------------------------------

def run_clip(
    clip_path: Path,
    row: GroundTruthRow,
    args: argparse.Namespace,
    default_model: Any,
    person_model: Any,
    weapon_model: Any,
    pose_model: Any,
) -> ClipResult:
    threat_classes = normalize_threat_classes(args.threat_classes)
    person_classes = normalize_threat_classes(args.person_classes)
    weapon_classes = normalize_threat_classes(args.weapon_classes)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    # Set up annotated video writer if requested
    annotated_writer: cv2.VideoWriter | None = None
    if args.save_annotated:
        annotated_out_dir = Path(args.annotated_dir)
        annotated_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = annotated_out_dir / clip_path.name
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_writer = cv2.VideoWriter(str(out_path), fourcc, source_fps, (width, height))

    frame_count = 0
    threat_frame_count = 0
    threats_seen: set[str] = set()
    object_threat_frames = 0
    violence_threat_frames = 0
    previous_pose_people: list[PosePersonState] = []
    pose_track_history: dict[int, deque[PosePersonState]] = {}
    next_pose_track_id = 1
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            if args.max_frames > 0 and frame_count > args.max_frames:
                break

            detections = predict_with_model(
                default_model, frame,
                conf=args.conf, imgsz=args.imgsz,
                threat_classes=threat_classes,
                source_model="default",
                use_tracking=False,
            )
            if person_model is not None:
                person_detections = predict_with_model(
                    person_model, frame,
                    conf=args.person_conf, imgsz=args.imgsz,
                    threat_classes=threat_classes,
                    source_model="person",
                )
                detections = merge_detections(detections, person_detections)
            if weapon_model is not None:
                weapon_detections = predict_with_model(
                    weapon_model, frame,
                    conf=args.weapon_conf, imgsz=args.imgsz,
                    threat_classes=threat_classes,
                    source_model="weapon",
                )
                detections = merge_detections(detections, weapon_detections)

            pose_people: list[PosePersonState] = []
            if pose_model is not None:
                pose_people = extract_pose_people(
                    pose_model, frame,
                    conf=args.pose_conf, imgsz=args.imgsz,
                )
                pose_people, next_pose_track_id = assign_pose_tracks(
                    pose_people,
                    previous_people=previous_pose_people,
                    next_track_id=next_pose_track_id,
                )
                pose_people = enrich_pose_people_with_history(pose_people, pose_track_history)
                previous_pose_people = list(pose_people)

            validated_weapons = validate_weapon_detections(
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
            raw_object = assess_threat(
                detections=detections,
                threat_classes=threat_classes,
                person_classes=person_classes,
                validated_weapon_detections=validated_weapons,
                assault_distance_ratio=args.assault_distance_ratio,
            )
            raw_violence = assess_violence(
                pose_people=pose_people,
                validated_weapon_detections=validated_weapons,
                violence_distance_ratio=args.violence_distance_ratio,
                violence_wrist_speed=args.violence_wrist_speed,
                violence_arm_extension_ratio=args.violence_arm_extension_ratio,
                weapon_hand_distance_ratio=args.weapon_hand_distance_ratio,
            )

            if raw_object.active:
                object_threat_frames += 1
            else:
                object_threat_frames = 0
            if raw_violence.active:
                violence_threat_frames += 1
            else:
                violence_threat_frames = 0

            obj_assessment = gate_assessment(
                raw_object,
                consecutive_threat_frames=object_threat_frames,
                min_threat_frames=max(1, args.min_threat_frames),
            )
            vio_assessment = gate_assessment(
                raw_violence,
                consecutive_threat_frames=violence_threat_frames,
                min_threat_frames=max(1, args.violence_min_frames),
            )
            assessment = choose_assessment(obj_assessment, vio_assessment)

            if assessment.active:
                threat_frame_count += 1
                threats_seen.add(assessment.title)

            if annotated_writer is not None:
                elapsed = max(time.time() - t0, 1e-6)
                display = build_display_detections(
                    detections=detections,
                    validated_weapon_detections=validated_weapons,
                    person_classes=person_classes,
                    threat_classes=threat_classes,
                    show_all_detections=False,
                )
                annotated_frame = draw_detections(
                    frame,
                    display,
                    fps=frame_count / elapsed,
                    active_event=False,
                    assessment=assessment,
                )
                annotated_writer.write(annotated_frame)
    finally:
        cap.release()
        if annotated_writer is not None:
            annotated_writer.release()
            print(f"           annotated → {Path(args.annotated_dir) / clip_path.name}")

    detected = threat_frame_count >= args.min_detect_frames

    return ClipResult(
        clip=row.clip,
        threat_class=row.threat_class,
        has_threat=row.has_threat,
        detected=detected,
        threat_frames=threat_frame_count,
        total_frames=frame_count,
        threats_seen=sorted(threats_seen),
        elapsed_seconds=round(time.time() - t0, 2),
    )


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(results: list[ClipResult]) -> dict[str, ClassMetrics]:
    metrics: dict[str, ClassMetrics] = {}
    for result in results:
        cls = result.threat_class
        if cls not in metrics:
            metrics[cls] = ClassMetrics(threat_class=cls)
        m = metrics[cls]
        if result.tp:
            m.tp += 1
        elif result.fp:
            m.fp += 1
        elif result.fn:
            m.fn += 1
        else:
            m.tn += 1
    return metrics


def compute_overall(metrics: dict[str, ClassMetrics]) -> ClassMetrics:
    overall = ClassMetrics(threat_class="OVERALL")
    for m in metrics.values():
        overall.tp += m.tp
        overall.fp += m.fp
        overall.fn += m.fn
        overall.tn += m.tn
    return overall


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict[str, ClassMetrics], overall: ClassMetrics) -> None:
    header = f"{'Class':<16} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  {'Prec':>6} {'Recall':>7} {'FPR':>6} {'F1':>6}"
    divider = "-" * len(header)
    print("\n" + divider)
    print(header)
    print(divider)

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "  n/a"

    for m in sorted(metrics.values(), key=lambda x: x.threat_class):
        print(
            f"{m.threat_class:<16} {m.tp:>4} {m.fp:>4} {m.fn:>4} {m.tn:>4}"
            f"  {fmt(m.precision):>6} {fmt(m.recall):>7} {fmt(m.fpr):>6} {fmt(m.f1):>6}"
        )
    print(divider)
    print(
        f"{'OVERALL':<16} {overall.tp:>4} {overall.fp:>4} {overall.fn:>4} {overall.tn:>4}"
        f"  {fmt(overall.precision):>6} {fmt(overall.recall):>7} {fmt(overall.fpr):>6} {fmt(overall.f1):>6}"
    )
    print(divider + "\n")


def print_gap_analysis(metrics: dict[str, ClassMetrics]) -> None:
    print("Gap Analysis")
    print("------------")
    for m in sorted(metrics.values(), key=lambda x: x.threat_class):
        issues: list[str] = []
        if m.recall is not None and m.recall < 0.6:
            issues.append(f"low recall ({m.recall:.2f}) — misses too many real threats")
        if m.fpr is not None and m.fpr > 0.3:
            issues.append(f"high FPR ({m.fpr:.2f}) — too many false alarms")
        if not issues:
            status = "OK"
        else:
            status = "; ".join(issues)
        print(f"  {m.threat_class:<16} {status}")
    print()


def save_report(
    path: str,
    results: list[ClipResult],
    metrics: dict[str, ClassMetrics],
    overall: ClassMetrics,
    args: argparse.Namespace,
) -> None:
    report = {
        "config": {
            "clips_dir": args.clips_dir,
            "ground_truth": args.ground_truth,
            "weights": args.weights,
            "weapon_weights": args.weapon_weights,
            "pose_weights": args.pose_weights,
            "conf": args.conf,
            "weapon_conf": args.weapon_conf,
            "min_detect_frames": args.min_detect_frames,
            "min_threat_frames": args.min_threat_frames,
        },
        "per_class": [m.as_dict() for m in sorted(metrics.values(), key=lambda x: x.threat_class)],
        "overall": overall.as_dict(),
        "clips": [
            {
                "clip": r.clip,
                "threat_class": r.threat_class,
                "has_threat": r.has_threat,
                "detected": r.detected,
                "result": (
                    "TP" if r.tp else "FP" if r.fp else "FN" if r.fn else "TN"
                ),
                "threat_frames": r.threat_frames,
                "total_frames": r.total_frames,
                "threats_seen": r.threats_seen,
                "elapsed_seconds": r.elapsed_seconds,
            }
            for r in results
        ],
    }
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.generate_template:
        generate_template(args.clips_dir)
        return

    if not args.ground_truth:
        print("Error: --ground-truth is required unless --generate-template is used.", file=sys.stderr)
        sys.exit(1)

    rows = load_ground_truth(args.ground_truth)
    clips_dir = Path(args.clips_dir)

    print(f"Loading models...")
    default_model = load_detection_model(args.weights, args.yolov5_repo)
    person_model = load_detection_model(args.person_weights, args.yolov5_repo) if args.person_weights else None
    weapon_model = (
        load_detection_model(args.weapon_weights, args.yolov5_repo, preferred_kind=args.weapon_loader)
        if args.weapon_weights
        else None
    )
    pose_model = load_ultralytics_model(args.pose_weights) if args.pose_weights else None

    print(f"Evaluating {len(rows)} clips from: {clips_dir}")
    print(f"Min detect frames: {args.min_detect_frames} | Max frames/clip: {args.max_frames or 'unlimited'}\n")

    results: list[ClipResult] = []
    skipped = 0

    for i, row in enumerate(rows, 1):
        clip_path = clips_dir / row.clip
        if not clip_path.exists():
            print(f"  [{i}/{len(rows)}] SKIP (not found): {row.clip}")
            skipped += 1
            continue

        label_str = "THREAT" if row.has_threat else "CLEAR "
        print(f"  [{i}/{len(rows)}] {label_str} [{row.threat_class}] {row.clip} ... ", end="", flush=True)

        result = run_clip(
            clip_path=clip_path,
            row=row,
            args=args,
            default_model=default_model,
            person_model=person_model,
            weapon_model=weapon_model,
            pose_model=pose_model,
        )
        results.append(result)

        outcome = "TP" if result.tp else "FP" if result.fp else "FN" if result.fn else "TN"
        detail = f"{result.threat_frames}/{result.total_frames} threat frames"
        print(f"{outcome}  ({detail}, {result.elapsed_seconds}s)")

        if args.verbose and result.threats_seen:
            print(f"           threats seen: {', '.join(result.threats_seen)}")

    if not results:
        print("No clips were evaluated. Check --clips-dir and --ground-truth paths.")
        return

    if skipped:
        print(f"\nWarning: {skipped} clip(s) not found and skipped.")

    metrics = aggregate_metrics(results)
    overall = compute_overall(metrics)

    print_metrics_table(metrics, overall)
    print_gap_analysis(metrics)

    if args.report_out:
        save_report(args.report_out, results, metrics, overall, args)


if __name__ == "__main__":
    main()
