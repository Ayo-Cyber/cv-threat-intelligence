"""Measure pose coverage and concealment scores with unchanged serving defaults."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import supervision as sv
from cvti.detector.core import load_ultralytics_model
from cvti.serving.camera import build_camera_states

root = Path("runs/chi_validation/scenarios/10_concealment")
parser = argparse.ArgumentParser()
parser.add_argument("--fps", type=float, default=4)
parser.add_argument("--heavy-stride", type=int, default=2)
parser.add_argument("--report", default="pose-probe.json")
parser.add_argument("--verify", action="store_true", help="Verify the first candidate per video using real Ollama")
args = parser.parse_args()
site = json.loads((root / "site.json").read_text())
model = load_ultralytics_model("models/yolov8n.pt")
pose = load_ultralytics_model("models/yolov8n-pose.pt")
reports = []
gate = None
if args.verify:
    from cvti.verification.gate import VerificationGate
    gate = VerificationGate(provider="ollama", model="gemma3:4b",
                            save_dir=root / (Path(args.report).stem + "-gate"))
for camera in site["cameras"]:
    state = build_camera_states({"cameras": [camera]}, pose_model=pose)[camera["id"]]["state"]
    state.imgsz = site["inference"]["imgsz"]
    state.heavy_stride = args.heavy_stride
    report = {"camera": camera["id"], "source": camera["source"],
              "samples": 0, "pose_updates": 0, "poses": 0, "complete_poses": 0,
              "max_score": 0, "candidate_assessments": 0, "alerts": [], "timeline": [], "verification": []}
    original = state._conceal.update_with_bag_detections
    def observe(frames, timestamp, bags):
        assessments = original(frames, timestamp, bags)
        report["pose_updates"] += 1
        report["poses"] += len(frames)
        report["complete_poses"] += sum(all(v is not None for v in f.keypoints.values()) for f in frames)
        report["max_score"] = max([report["max_score"]] + [a.score for a in assessments])
        report["candidate_assessments"] += sum(a.candidate for a in assessments)
        report["timeline"].append({"seconds": timestamp, "poses": len(frames),
            "bags": len(bags), "scores": [a.score for a in assessments],
            "assessments": [{"track": a.track_id, "score": a.score,
                             "candidate": a.candidate, "components": a.components,
                             "destination": a.destination} for a in assessments]})
        return assessments
    state._conceal.update_with_bag_detections = observe
    cap = cv2.VideoCapture(camera["source"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError(camera["source"])
    stride = max(1, round(fps/args.fps))
    index = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if index % stride == 0:
            result = model.runner.predict(image, conf=.4, imgsz=512, device="cpu", verbose=False)[0]
            alerts = state.process(sv.Detections.from_ultralytics(result), image, index/fps)
            report["samples"] += 1
            report["alerts"].extend({"rule": a.rule_name, "seconds": index/fps} for a in alerts)
            if gate is not None and alerts and not report["verification"]:
                payload = alerts[0].payload
                verdict = gate.verify(payload["frames"], payload["candidate"], payload["scene"])
                report["verification"].append({"seconds": index/fps, "confirmed": verdict.confirmed,
                    "confidence": verdict.confidence, "reason": verdict.reason, "error": verdict.error})
        index += 1
    cap.release()
    reports.append(report)
    report["requested_fps"] = args.fps
    report["heavy_stride"] = args.heavy_stride
    (root / args.report).write_text(json.dumps(reports, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "timeline"}), flush=True)
