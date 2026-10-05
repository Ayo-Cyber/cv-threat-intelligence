"""Real-model prerequisite check; not an end-to-end KPI 9 acceptance test."""
import json
import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from cvti.detector.core import load_ultralytics_model, extract_detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/chi_validation/caviar/LeftBag_PickedUp.mpg"))
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("runs/chi_validation/scenarios/09_object_state/detection_probe"))
    args = parser.parse_args()
    if args.image_size <= 0:
        parser.error("image-size must be positive")
    source, output = args.source, args.output
    output.mkdir(parents=True, exist_ok=True)
    model = load_ultralytics_model(args.model)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError(f"Cannot decode {source}")
    rows = []
    index = 0
    inference_seconds = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % round(fps) == 0:
                started = time.perf_counter()
                result = model.runner.predict(frame, conf=.4, imgsz=args.image_size,
                                              device="cpu", verbose=False)[0]
                inference_seconds += time.perf_counter() - started
                objects = extract_detections(result, model.names, set())
                bags = [d for d in objects if d.label in {"suitcase", "backpack", "handbag"}]
                rows.append({"seconds": index / fps, "bags": [
                    {"label": d.label, "confidence": float(d.confidence),
                     "bbox": list(map(float, d.bbox))} for d in bags]})
                if bags or index % (round(fps) * 5) == 0:
                    for d in bags:
                        x1, y1, x2, y2 = map(int, d.bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
                        cv2.putText(frame, f"{d.label} {d.confidence:.2f}",
                                    (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX,
                                    .35, (0, 200, 0), 1)
                    cv2.imwrite(str(output / f"sample-{index:05d}.jpg"), frame)
            index += 1
    finally:
        cap.release()
    report = {"source": str(source), "model": args.model,
              "confidence": .4, "image_size": args.image_size, "sample_fps": 1,
              "mean_inference_ms": 1000 * inference_seconds / max(1, len(rows)),
              "samples": len(rows), "samples_with_bags": sum(bool(r["bags"]) for r in rows),
              "status": "Prerequisite detection probe only; no VLM or incident validation",
              "observations": rows}
    (output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "observations"}))


if __name__ == "__main__":
    main()
