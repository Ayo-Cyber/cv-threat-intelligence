"""Save same-frame nano/small-model predictions for manual coverage review."""
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--offset", type=int, choices=(0, 25), default=0)
args = parser.parse_args()
out = Path("runs/chi_validation/caviar_models") / f"offset_{args.offset}"
out.mkdir(parents=True, exist_ok=True)
models = {"nano": YOLO("models/yolov8n.pt"),
          "small": YOLO("runs/chi_validation/yolov8s.pt")}
cap = cv2.VideoCapture("runs/chi_validation/caviar/Meet_WalkTogether1.mpg")
index, records = 0, []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if index % 50 == args.offset:
        for name, model in models.items():
            for angle in (0, -30, -60, -90):
                h, w = frame.shape[:2]
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
                nw, nh = round(h * s + w * c), round(h * c + w * s)
                matrix[0, 2] += (nw - w) / 2
                matrix[1, 2] += (nh - h) / 2
                rotated = cv2.warpAffine(frame, matrix, (nw, nh))
                result = model.predict(rotated, imgsz=960, conf=0.25, classes=[0], device="cpu", verbose=False)[0]
                boxes = []
                inverse = cv2.invertAffineTransform(matrix)
                for x1, y1, x2, y2, confidence, cls in result.boxes.data.tolist():
                    corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
                    original = corners @ inverse[:, :2].T + inverse[:, 2]
                    boxes.append([*original.min(axis=0), *original.max(axis=0), confidence, cls])
                records.append({"frame": index, "model": name, "angle": angle, "boxes": boxes})
                cv2.imwrite(str(out / f"{index}_{name}_{angle}.jpg"), result.plot())
        print(index, flush=True)
    index += 1
cap.release()
(out / "results.json").write_text(json.dumps(records, indent=2))

truth = {}
for item in ET.parse("runs/chi_validation/caviar/mwt1gt.xml").getroot().findall("frame"):
    boxes = []
    for obj in item.findall("objectlist/object"):
        b = obj.find("box")
        x, y, w, h = [float(b.get(k)) for k in ("xc", "yc", "w", "h")]
        boxes.append([x-w/2, y-h/2, x+w/2, y+h/2])
    truth[int(item.get("number"))] = boxes

def iou(a,b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection / max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

summary = []
for name in models:
    for angle in (0,-30,-60,-90):
        for threshold in (0.25,0.4,0.5,0.6):
            tp = fp = fn = 0
            for row in records:
                if row["model"] != name or row["angle"] != angle:
                    continue
                gt = list(truth[row["frame"]])
                for pred in sorted(row["boxes"], key=lambda b:-b[4]):
                    if pred[4] < threshold:
                        continue
                    scores = [iou(pred,b) for b in gt]
                    if scores and max(scores)>=0.3:
                        tp+=1
                        gt.pop(int(np.argmax(scores)))
                    else:
                        fp+=1
                fn+=len(gt)
            summary.append(dict(model=name,angle=angle,threshold=threshold,tp=tp,fp=fp,fn=fn))
print(json.dumps(summary,indent=2))
(out / "metrics.json").write_text(json.dumps(summary,indent=2))
