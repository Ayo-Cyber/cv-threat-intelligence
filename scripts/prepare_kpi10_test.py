"""Prepare isolated concealment test inputs and source-frame review sheets."""
import json
from pathlib import Path
import cv2
import numpy as np

root = Path("runs/chi_validation/scenarios/10_concealment")
root.mkdir(parents=True, exist_ok=True)
rules = {"use_case_id": "kpi10_validation", "rules": [{
    "name": "product_concealment", "trigger": {"detector": "concealment"},
    "priority": "high",
    "gate_question": "Is a person visibly pocketing or putting a product into a personal bag? Require visual evidence of the action; normal handling, returning an item, and merely touching clothing are not sufficient. Do not infer theft or intent from captions."}]}
(root / "rules.json").write_text(json.dumps(rules, indent=2))
cameras = []
for n in (1, 2):
    source = f"data/test_clips/theft_shop_0{n}.mp4"
    cameras.append({"id": f"retail_{n}", "source": source,
                    "config": str(root / "rules.json"), "concealment": True,
                    "normal_movement": True, "box_display": "synchronized"})
    cap = cv2.VideoCapture(source)
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError(source)
    sheet = np.full((3*300, 4*320, 3), 245, dtype=np.uint8)
    for i, index in enumerate(np.linspace(0, count-1, 12, dtype=int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        scale = min(320/frame.shape[1], 270/frame.shape[0])
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        y, x = (i//4)*300, (i%4)*320
        sheet[y:y+frame.shape[0], x:x+frame.shape[1]] = frame
        cv2.putText(sheet, f"{index/fps:.2f}s", (x+5,y+290), cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
    cv2.imwrite(str(root / f"retail_{n}_review.jpg"), sheet)
    cap.release()
site = {"name": "KPI 10 concealment validation", "configured": True,
        "notify": "console", "inference": {"imgsz": 512, "confidence": .4, "target_fps": 4},
        "cameras": cameras}
(root / "site.json").write_text(json.dumps(site, indent=2))
site["name"] = "KPI 10 dense pose experiment"
site["inference"]["target_fps"] = 8
for camera in cameras:
    camera["heavy_stride"] = 1
(root / "site-dense.json").write_text(json.dumps(site, indent=2))
site["name"] = "KPI 10 time-based gesture validation"
site["inference"]["target_fps"] = 4
(root / "site-timed.json").write_text(json.dumps(site, indent=2))
site["name"] = "KPI 10 wig sequence validation"
site["cameras"] = cameras[:1]
(root / "site-wig.json").write_text(json.dumps(site, indent=2))
control = "runs/chi_validation/caviar/WalkByShop1front_excerpt.mp4"
if Path(control).exists():
    site["name"] = "KPI 10 three-camera checkpoint"
    site["cameras"] = cameras + [{
        "id": "normal_shopping", "source": control,
        "config": str(root / "rules.json"), "concealment": True,
        "normal_movement": True, "box_display": "synchronized", "heavy_stride": 1,
    }]
    (root / "site-three.json").write_text(json.dumps(site, indent=2))
