"""Preserve gate inputs and make dense source-frame sheets for manual review."""
from pathlib import Path
import shutil
import cv2
import numpy as np

root = Path("runs/chi_validation/scenarios/10_concealment/trace")
root.mkdir(parents=True, exist_ok=True)
gate = root / "original-gate"
if not gate.exists():
    shutil.copytree("runs/chi_validation/desktop/gate", gate)
for number, start, stop in [(1, 8, 14), (1, 14, 20), (2, 4, 10)]:
    cap = cv2.VideoCapture(f"data/test_clips/theft_shop_0{number}.mp4")
    sheet = np.full((4*350, 3*240, 3), 245, dtype=np.uint8)
    for i in range(12):
        sec = start + i*.5
        cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Missing frame at {sec}")
        scale = min(240/frame.shape[1],320/frame.shape[0])
        frame = cv2.resize(frame,None,fx=scale,fy=scale)
        y,x=(i//3)*350,(i%3)*240
        sheet[y:y+frame.shape[0],x:x+frame.shape[1]]=frame
        cv2.putText(sheet,f"{sec:.1f}s",(x+5,y+343),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
    cv2.imwrite(str(root/f"retail_{number}_{start}_{stop}.jpg"),sheet)
    cap.release()
