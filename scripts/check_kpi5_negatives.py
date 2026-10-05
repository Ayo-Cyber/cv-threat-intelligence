"""Real-model negative probes; complements, does not replace, desktop evidence."""
import json
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import supervision as sv
from ultralytics import YOLO
from cvti.serving.camera import build_camera_states
from cvti.serving.camera_inference import CameraInference

root = Path("runs/chi_validation/scenarios/05_multiple_people_moving")
parser = argparse.ArgumentParser()
parser.add_argument("--site", default="negative-site.json")
parser.add_argument("--report", default="negative-probe.json")
args = parser.parse_args()
site = json.loads((root / args.site).read_text())
base = YOLO("models/yolov8n.pt")
reports = []
for camera in site["cameras"][:2]:
    camera_id = camera["id"]
    states = build_camera_states({"cameras": [camera]})
    state = states[camera_id]["state"]
    profiles = {camera_id: camera["detection_inference"]} if "detection_inference" in camera else {}
    predictor = CameraInference(profiles)
    predictor.load(base)
    cap = cv2.VideoCapture(camera["source"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError("Cannot decode negative control")
    stride = max(1, round(fps / 2))
    report = {"camera": camera_id, "source": camera["source"], "samples": 0,
              "frames_with_tracks": 0, "frames_with_movers": 0,
              "max_tracks": 0, "max_movers": 0, "candidates": [], "timeline": []}
    index = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frame = SimpleNamespace(camera_id=camera_id, image=image)
            result = predictor.predict(base, [frame], imgsz=960, conf=.25,
                                       device="cpu", half=False, verbose=False)[0]
            alerts = state.process(sv.Detections.from_ultralytics(result), image, index / fps)
            tracks = len(state._box_by_track)
            movers = sum(o["label"].endswith(" MOVING") for o in state._motion_overlays)
            report["samples"] += 1
            report["frames_with_tracks"] += int(tracks > 0)
            report["frames_with_movers"] += int(movers > 0)
            report["max_tracks"] = max(report["max_tracks"], tracks)
            report["max_movers"] = max(report["max_movers"], movers)
            report["timeline"].append({"seconds": index/fps, "tracks": tracks, "movers": movers})
            report["candidates"].extend({"rule": a.rule_name, "seconds": index/fps} for a in alerts)
        index += 1
    cap.release()
    report["duration_seconds"] = index / fps
    reports.append(report)
    (root / args.report).write_text(json.dumps(reports, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "timeline"}), flush=True)
