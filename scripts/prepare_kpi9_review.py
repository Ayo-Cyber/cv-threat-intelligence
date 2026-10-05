"""Prepare a source-timed review sheet before choosing an object-state zone."""
from pathlib import Path
import json
import cv2
import numpy as np

source = Path("runs/chi_validation/caviar/LeftBag_PickedUp.mpg")
root = Path("runs/chi_validation/scenarios/09_object_state")
root.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(source))
fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not cap.isOpened() or fps <= 0:
    raise RuntimeError(str(source))
sheet = np.full((3 * 270, 4 * 384, 3), 245, dtype=np.uint8)
for i, index in enumerate(np.linspace(0, count - 1, 12, dtype=int)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.resize(frame, (320, 240))
    y, x = (i // 4) * 270, (i % 4) * 384
    sheet[y:y+240, x:x+320] = frame
    cv2.putText(sheet, f"{index/fps:.1f}s", (x+5, y+262), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 1)
cv2.imwrite(str(root / "left_bag_source_review.jpg"), sheet)
cap.release()
metadata = {"source": str(source), "fps": fps, "frames": count,
            "duration_seconds": count / fps, "status": "source review only; model not yet tested"}
(root / "source-review.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata))
