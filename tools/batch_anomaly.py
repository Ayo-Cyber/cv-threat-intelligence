"""batch_anomaly.py — run the unified detector across a folder of clips, ONE model load.

Tallies, per clip, which threat signals fire: weapons, violence, theft, concealment.
Reuses detector.py's exact per-frame wiring (same as eval.py) and adds the concealment
(action) layer. No ground truth needed — it reports detection coverage across the set.

Usage:
    python tools/batch_anomaly.py --clips-dir data/anomaly \
        --weapon-weights models/weapon_best.pt --stride 2
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for detector/concealment

from concealment import ConcealmentDetector
from detector import (
    CONCEALMENT_BAG_CLASSES,
    PosePersonState,
    TheftDetector,
    ViolenceTemporalGate,
    assess_threat,
    assess_violence,
    assign_pose_tracks,
    choose_assessment,
    enrich_pose_people_with_history,
    extract_pose_people,
    gate_assessment,
    load_detection_model,
    load_ultralytics_model,
    merge_detections,
    normalize_label,
    normalize_threat_classes,
    pose_people_to_concealment_frames,
    predict_with_model,
    validate_weapon_detections,
)

# Defaults mirror eval.py / detector.py so behaviour matches the real pipeline.
THREAT_CLASSES = normalize_threat_classes("person")
PERSON_CLASSES = normalize_threat_classes("person")
WEAPON_CLASSES = normalize_threat_classes("knife,gun")


def run_clip(clip: Path, models: dict, args) -> dict:
    default_model, weapon_model, pose_model = models["default"], models["weapon"], models["pose"]
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    violence_gate = ViolenceTemporalGate(window=8, min_votes=3)
    theft_detector = TheftDetector(acquire_frames=8, depart_frames=6, approach_ratio=2.0)
    concealment = ConcealmentDetector()
    prev_pose: list[PosePersonState] = []
    pose_hist: dict[int, deque] = {}
    next_tid = 1

    counts = {"weapon": 0, "violence": 0, "theft": 0, "concealment": 0}
    conceal_peak = 0.0
    obj_streak = 0
    idx = 0
    processed = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if args.max_frames and idx > args.max_frames:
            break
        if (idx - 1) % args.stride != 0:
            continue
        processed += 1
        ts = idx / fps

        detections = predict_with_model(default_model, frame, conf=0.35, imgsz=args.imgsz,
                                        threat_classes=THREAT_CLASSES, source_model="default",
                                        use_tracking=False)
        if weapon_model is not None:
            detections = merge_detections(detections, predict_with_model(
                weapon_model, frame, conf=0.35, imgsz=args.imgsz,
                threat_classes=THREAT_CLASSES, source_model="weapon"))

        pose_people: list[PosePersonState] = []
        if pose_model is not None:
            pose_people = extract_pose_people(pose_model, frame, conf=0.35, imgsz=args.imgsz)
            pose_people, next_tid = assign_pose_tracks(pose_people, previous_people=prev_pose, next_track_id=next_tid)
            pose_people = enrich_pose_people_with_history(pose_people, pose_hist)
            prev_pose = list(pose_people)

        validated = validate_weapon_detections(
            detections=detections, weapon_classes=WEAPON_CLASSES, person_classes=PERSON_CLASSES,
            pose_people=pose_people, frame_shape=frame.shape,
            weapon_min_area_ratio=0.002, weapon_max_area_ratio=0.18,
            weapon_border_margin_ratio=0.03, weapon_hand_distance_ratio=0.20,
            allow_unattached_weapons=False,
        )
        raw_object = assess_threat(detections=detections, threat_classes=THREAT_CLASSES,
                                   person_classes=PERSON_CLASSES, validated_weapon_detections=validated,
                                   assault_distance_ratio=1.2)
        raw_violence = assess_violence(
            pose_people=pose_people, validated_weapon_detections=validated,
            violence_distance_ratio=1.1, violence_wrist_speed=120.0,
            violence_arm_extension_ratio=0.35, weapon_hand_distance_ratio=0.20,
            violence_wrist_accel=800.0,
        )
        obj_streak = obj_streak + 1 if raw_object.active else 0
        obj_assessment = gate_assessment(raw_object, consecutive_threat_frames=obj_streak, min_threat_frames=3)
        vio_assessment = violence_gate.update(raw_violence)
        theft_assessment = theft_detector.update(pose_people=pose_people, detections=detections, timestamp=ts)

        # Concealment (action) layer — bags come from the existing object detections.
        bag_bboxes = [d.bbox for d in detections if normalize_label(d.label) in CONCEALMENT_BAG_CLASSES]
        conceal = concealment.update(pose_people_to_concealment_frames(pose_people, ts), ts, bag_bboxes=bag_bboxes or None)

        if validated:
            counts["weapon"] += 1
        if vio_assessment.active:
            counts["violence"] += 1
        if theft_assessment.active:
            counts["theft"] += 1
        if any(a.candidate for a in conceal):
            counts["concealment"] += 1
        conceal_peak = max(conceal_peak, *( [a.score for a in conceal] or [0.0]))

    cap.release()
    fired = {
        "weapon": counts["weapon"] >= 2,
        "violence": counts["violence"] >= 3,
        "theft": counts["theft"] >= 1,
        "concealment": counts["concealment"] >= 1,
    }
    return {"clip": clip.name, "processed": processed, "counts": counts,
            "fired": fired, "conceal_peak": round(conceal_peak, 2),
            "secs": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--weapon-weights", default="models/weapon_best.pt")
    ap.add_argument("--weapon-loader", default="yolov5")
    ap.add_argument("--yolov5-repo", default="external/yolov5")
    ap.add_argument("--pose-weights", default="yolov8n-pose.pt")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=2, help="Process every Nth frame for speed.")
    ap.add_argument("--max-frames", type=int, default=450)
    args = ap.parse_args()

    print("Loading models (once)...")
    models = {
        "default": load_detection_model(args.weights, args.yolov5_repo),
        "weapon": load_detection_model(args.weapon_weights, args.yolov5_repo, preferred_kind=args.weapon_loader)
                  if args.weapon_weights else None,
        "pose": load_ultralytics_model(args.pose_weights) if args.pose_weights else None,
    }
    clips = sorted(Path(args.clips_dir).glob("*.mp4"))
    print(f"Running {len(clips)} clips (stride={args.stride}, max_frames={args.max_frames})\n")

    results = []
    for i, clip in enumerate(clips, 1):
        r = run_clip(clip, models, args)
        results.append(r)
        tags = [k.upper() for k, v in r["fired"].items() if v] or ["clear"]
        print(f"[{i:2}/{len(clips)}] {r['clip']:<10} -> {', '.join(tags)}  "
              f"(conceal_peak={r['conceal_peak']}, {r['secs']}s)")

    n = len(results)
    print(f"\n=== SUMMARY over {n} clips ===")
    for key in ("weapon", "violence", "theft", "concealment"):
        c = sum(1 for r in results if r["fired"][key])
        print(f"  {key:<12}: fired in {c}/{n} clips ({100*c//max(n,1)}%)")
    any_fired = sum(1 for r in results if any(r["fired"].values()))
    print(f"  ANY THREAT  : {any_fired}/{n} clips ({100*any_fired//max(n,1)}%)")


if __name__ == "__main__":
    main()
