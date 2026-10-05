"""Decode a bounded local evaluation excerpt and document its provenance."""
import json
from pathlib import Path
import cv2
import numpy as np

root = Path("runs/chi_validation/caviar")
source = root / "WalkByShop1front.mpg"
target = root / "WalkByShop1front_excerpt.mp4"
cap = cv2.VideoCapture(str(source))
fps = cap.get(cv2.CAP_PROP_FPS)
ok, frame = cap.read()
if not ok or fps <= 0:
    raise RuntimeError("Downloaded source has no decodable frames")
h, w = frame.shape[:2]
writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
if not writer.isOpened():
    raise RuntimeError("Unable to create evaluation excerpt")
count = 0
samples = []
while ok and count < int(fps * 20):
    writer.write(frame)
    if count % max(1, int(fps * 2)) == 0:
        samples.append(frame.copy())
    count += 1
    ok, frame = cap.read()
writer.release()
cap.release()
if count < fps * 3:
    raise RuntimeError("Less than three seconds decoded; control is unusable")
sheet = np.full((h * 2, w * 5, 3), 230, dtype=np.uint8)
for i, sample in enumerate(samples[:10]):
    y, x = (i // 5) * h, (i % 5) * w
    sheet[y:y+h, x:x+w] = sample
cv2.imwrite(str(root / "WalkByShop1front_excerpt_review.jpg"), sheet)
metadata = {
    "source_url": "https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA2/WalkByShop1front/WalkByShop1front.mpg",
    "local_source": str(source), "excerpt": str(target), "frames": count,
    "fps": fps, "duration_seconds": count / fps, "width": w, "height": h,
    "note": "Bounded decoded prefix of a partial download; not the complete dataset clip. Local evaluation only; redistribution rights not established.",
}
(root / "WalkByShop1front_excerpt.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
