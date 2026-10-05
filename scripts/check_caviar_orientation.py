"""Compare real person detections under camera roll; no model downloads."""
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

out = Path("runs/chi_validation/caviar_orientation")
out.mkdir(parents=True, exist_ok=True)
model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture("runs/chi_validation/caviar/Meet_WalkTogether1.mpg")
records = []
for seconds in (2, 10, 20, 35):
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    if not ok:
        continue
    for angle in (0, -30, -60, -90, 30, 60, 90):
        h, w = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
        nw, nh = round(h * s + w * c), round(h * c + w * s)
        matrix[0, 2] += (nw - w) / 2
        matrix[1, 2] += (nh - h) / 2
        rotated = cv2.warpAffine(frame, matrix, (nw, nh))
        result = model.predict(rotated, imgsz=960, conf=0.25, classes=[0], device="cpu", verbose=False)[0]
        records.append({"seconds": seconds, "angle": angle,
                        "boxes": result.boxes.data.tolist()})
        cv2.imwrite(str(out / f"{seconds}s_{angle}.jpg"), result.plot())
        print(seconds, angle, result.boxes.conf.tolist(), flush=True)
cap.release()
(out / "results.json").write_text(json.dumps(records, indent=2))
