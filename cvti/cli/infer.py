"""infer_video.py — Run the trained classifier on a video and show live predictions.

Usage:
    python3 infer_video.py --source data/test_clips/stabbing_event.mp4
    python3 infer_video.py --source data/test_clips/normal_street_02.mp4
    python3 infer_video.py --source 0   # webcam
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_WEIGHTS = "runs/classify/runs/classify/robbery_v14/weights/best.pt"

COLORS = {
    "theft":    (0, 100, 255),   # orange-red
    "violence": (0, 0, 255),     # red
    "normal":   (0, 200, 0),     # green
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video path or webcam index (0)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to trained best.pt")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold to show ANOMALY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    print("Press Q to quit.")
    frame_count = 0
    fps_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run classifier
        result = model.predict(frame, verbose=False, imgsz=224)[0]
        pred_cls   = result.names[int(result.probs.top1)]
        confidence = float(result.probs.top1conf)

        # Only flag anomaly if above threshold
        if pred_cls in ("theft", "violence") and confidence < args.conf:
            pred_cls = "normal"

        color = COLORS.get(pred_cls, (200, 200, 200))

        # Draw banner
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
        label = f"{pred_cls.upper()}  {confidence:.0%}"
        cv2.putText(frame, label, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("CV Threat — Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
