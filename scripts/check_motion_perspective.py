"""Compare old/new motion filtering on identical real detections and tracks."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import supervision as sv
from ultralytics import YOLO
from cvti.detector.person_motion import PersonMotionTracker

model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture("data/test_clips/normal_street_01.mp4")
tracker = sv.ByteTrack()
motion = [PersonMotionTracker(), PersonMotionTracker(perspective_compensation=True)]
counts = {"tracked_top": 0, "tracked_bottom": 0,
          "old_top": 0, "old_bottom": 0, "new_top": 0, "new_bottom": 0}
fps = cap.get(cv2.CAP_PROP_FPS)
for i in range(500):
    ok, frame = cap.read()
    if not ok:
        break
    if i % 6:
        continue
    result = model.predict(frame, conf=.4, imgsz=512, device="cpu", verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    tracks = tracker.update_with_detections(detections[detections.class_id == 0])
    people = [(int(tid), *box) for tid, box in zip(tracks.tracker_id, tracks.xyxy)]
    for _, x1, y1, x2, y2 in people:
        counts["tracked_top" if (y1+y2)/2 < frame.shape[0]/2 else "tracked_bottom"] += 1
    for prefix, state in zip(("old", "new"), motion):
        for person in state.update(people, i/fps, frame.shape[:2]):
            if person.observed and person.moving:
                half = "top" if (person.bbox[1]+person.bbox[3])/2 < frame.shape[0]/2 else "bottom"
                counts[f"{prefix}_{half}"] += 1
cap.release()
out = Path("runs/chi_validation/scenarios/04_normal_movement/evidence/perspective-comparison.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(counts, indent=2))
print(json.dumps(counts))
