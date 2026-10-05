"""Offline real-model probe; counts are diagnostic, not accuracy measurements."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import supervision as sv
from cvti.detector.core import load_ultralytics_model, extract_detections
from cvti.serving.camera import build_camera_states


def main():
    output = Path("runs/chi_validation/probe")
    output.mkdir(parents=True, exist_ok=True)
    model = load_ultralytics_model("models/yolov8n.pt")
    pose = load_ultralytics_model("models/yolov8n-pose.pt")
    reports = []
    for source in sys.argv[1:]:
        camera = {"id": "probe", "source": source, "config": "configs/chi_pilot_v1.json",
                  "normal_movement": True, "multiple_people_moving": True, "concealment": True}
        state = build_camera_states({"cameras": [camera]}, pose_model=pose)["probe"]["state"]
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            raise RuntimeError(f"Cannot decode {source}")
        stride = max(1, round(fps / 4))
        report = {"source": source, "fps": fps, "samples": 0, "frames_with_people": 0,
                  "frames_with_movement": 0, "max_people": 0, "candidates": [],
                  "raw_person_frames": 0, "bag_frames": 0}
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                result = model.runner.predict(frame, conf=.4, imgsz=512, device="cpu", verbose=False)[0]
                objects = extract_detections(result, model.names, set())
                report["raw_person_frames"] += int(any(d.label == "person" for d in objects))
                report["bag_frames"] += int(any(d.label in {"suitcase", "backpack", "handbag"} for d in objects))
                alerts = state.process(sv.Detections.from_ultralytics(result), frame, index / fps, objects)
                people = len(state._box_by_track)
                report["samples"] += 1
                report["frames_with_people"] += int(people > 0)
                report["max_people"] = max(report["max_people"], people)
                report["frames_with_movement"] += int(bool(state._motion_overlays))
                report["candidates"].extend({"rule": a.rule_name, "seconds": a.timestamp} for a in alerts)
            index += 1
        cap.release()
        report["duration_seconds"] = index / fps
        reports.append(report)
        (output / "results.json").write_text(json.dumps(reports, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
