"""train_classifier.py — Extract frames from CamNuvem dataset, fine-tune YOLOv8s-cls, report metrics.

Usage:
    python3 train_classifier.py                     # full run: extract + train + eval
    python3 train_classifier.py --skip-extract      # re-train without re-extracting frames
    python3 train_classifier.py --eval-only         # evaluate an existing trained model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("CamNuvem Robbery Dataset/videos/samples")
FRAMES_ROOT  = Path("data/classify")
RUNS_ROOT    = Path("runs/classify")
MODEL_NAME   = "robbery_v19"

SPLITS = ("training", "test")
CLASSES = ("theft", "violence", "normal")

YOLO_SPLIT_MAP = {"training": "train", "test": "val"}

FRAMES_PER_CLIP   = 10    # max frames extracted per clip
FRAME_SAMPLE_RATE = 5     # take every Nth frame (skip duplicates)
TRAIN_EPOCHS      = 5
IMAGE_SIZE        = 224
BATCH_SIZE        = 16


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(force: bool = False) -> None:
    print("\n=== Step 1: Extracting frames ===")
    total_written = 0

    # Source 1 — CamNuvem dataset (structured splits)
    for split in SPLITS:
        yolo_split = YOLO_SPLIT_MAP[split]
        for cls in CLASSES:
            src_dir = DATASET_ROOT / split / cls
            if not src_dir.exists():
                continue
            dst_dir = FRAMES_ROOT / yolo_split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            clips = list(src_dir.glob("*.mp4"))
            print(f"  CamNuvem {split}/{cls}: {len(clips)} clips → {dst_dir}")
            total_written += _extract_clips(clips, dst_dir, force)

    # Source 2 — YouTube downloads sitting directly in train/*/  as MP4s
    for cls in CLASSES:
        src_dir = FRAMES_ROOT / "train" / cls
        if not src_dir.exists():
            continue
        mp4s = list(src_dir.glob("*.mp4"))
        if not mp4s:
            continue
        print(f"  YouTube {cls}: {len(mp4s)} clips → {src_dir}")
        total_written += _extract_clips(mp4s, src_dir, force)

    print(f"\n  Total frames written: {total_written}")
    for split in ("train", "val"):
        for cls in CLASSES:
            n = len(list((FRAMES_ROOT / split / cls).glob("*.jpg")))
            print(f"  {split}/{cls}: {n} frames")


def _extract_clips(clips: list, dst_dir: Path, force: bool) -> int:
    written = 0
    for clip in clips:
        stem = clip.stem
        existing = list(dst_dir.glob(f"{stem}_*.jpg"))
        if existing and not force:
            written += len(existing)
            continue
        cap = cv2.VideoCapture(str(clip))
        frame_idx = 0
        saved = 0
        while saved < FRAMES_PER_CLIP:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % FRAME_SAMPLE_RATE == 0:
                out_path = dst_dir / f"{stem}_{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1
                written += 1
            frame_idx += 1
        cap.release()
    return written


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> Path:
    print("\n=== Step 2: Training YOLOv8s-cls ===")
    print(f"  Dataset : {FRAMES_ROOT}")
    print(f"  Epochs  : {TRAIN_EPOCHS}")
    print(f"  Img size: {IMAGE_SIZE}")
    print(f"  Batch   : {BATCH_SIZE}")

    model = YOLO("yolov8s-cls.pt")
    results = model.train(
        data=str(FRAMES_ROOT),
        epochs=TRAIN_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(RUNS_ROOT),
        name=MODEL_NAME,
        patience=10,
        workers=4,
        verbose=False,
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n  Best weights saved to: {best_weights}")
    return best_weights


# ---------------------------------------------------------------------------
# Evaluation + metrics
# ---------------------------------------------------------------------------

def evaluate(weights: Path) -> dict:
    print("\n=== Step 3: Evaluation metrics ===")
    model = YOLO(str(weights))

    # Collect predictions on val set
    tp = tn = fp = fn = 0
    results_log = []

    for cls_name in CLASSES:
        val_dir = FRAMES_ROOT / "val" / cls_name
        frames = list(val_dir.glob("*.jpg"))
        print(f"  Evaluating {cls_name}: {len(frames)} frames")

        for frame_path in frames:
            result = model.predict(str(frame_path), verbose=False)[0]
            pred_idx = int(result.probs.top1)
            pred_cls = result.names[pred_idx]
            confidence = float(result.probs.top1conf)

            correct = (pred_cls == cls_name)
            is_threat = (cls_name != "normal")

            if is_threat and correct:
                tp += 1
            elif is_threat and not correct:
                fn += 1
            elif not is_threat and correct:
                tn += 1
            else:
                fp += 1

            results_log.append({
                "file": frame_path.name,
                "true": cls_name,
                "pred": pred_cls,
                "confidence": round(confidence, 3),
                "correct": correct,
            })

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    metrics = {
        "model": str(weights),
        "classes": list(CLASSES),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "precision":  round(precision, 3),
        "recall":     round(recall, 3),
        "fpr":        round(fpr, 3),
        "f1":         round(f1, 3),
        "accuracy":   round(accuracy, 3),
    }

    # Print summary
    print(f"""
  ┌─────────────────────────────────┐
  │   Confusion Matrix              │
  │   TP={tp:4d}   FN={fn:4d}             │
  │   FP={fp:4d}   TN={tn:4d}             │
  └─────────────────────────────────┘

  Precision : {precision:.3f}
  Recall    : {recall:.3f}   (catches {recall*100:.1f}% of threats)
  FPR       : {fpr:.3f}   (false alarm rate)
  F1        : {f1:.3f}
  Accuracy  : {accuracy:.3f}
""")

    # Save report
    report_path = RUNS_ROOT / MODEL_NAME / "metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Metrics saved to: {report_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8s classifier on CamNuvem robbery dataset.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip frame extraction if already done.")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on an existing model.")
    parser.add_argument("--weights", type=str, default="", help="Path to weights for --eval-only mode.")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract frames even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.eval_only:
        weights = Path(args.weights) if args.weights else Path(RUNS_ROOT / MODEL_NAME / "weights" / "best.pt")
        if not weights.exists():
            print(f"Weights not found: {weights}")
            print("Run without --eval-only first to train the model.")
            return
        evaluate(weights)
        return

    if not args.skip_extract:
        extract_frames(force=args.force_extract)

    weights = train()
    evaluate(weights)

    print("\n=== Done ===")
    print(f"Best model : {weights}")
    print(f"Use in detector with: --classifier-weights {weights}")


if __name__ == "__main__":
    main()
