"""Create a disclosed frozen-frame negative control, not natural video evidence."""
from pathlib import Path
import cv2

out = Path("runs/chi_validation/scenarios/05_multiple_people_moving")
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture("data/test_clips/normal_street_01.mp4")
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError("Cannot read crowd source")
h, w = frame.shape[:2]
writer = cv2.VideoWriter(str(out / "stationary_frozen_control.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
if not writer.isOpened():
    raise RuntimeError("Cannot encode frozen control")
for _ in range(25 * 30):
    writer.write(frame)
writer.release()
print("Created 30-second frozen-frame control from first crowd frame; NOT natural stationary footage.")
