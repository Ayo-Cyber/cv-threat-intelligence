"""gate_bakeoff.py — compare local (Ollama) VLMs on OUR gate task, one command.

Rule-aware: each labeled clip has a `rule` (concealment | violence | weapon). The harness
selects frames for that rule (using the real detectors), asks the gate the matching
question, and scores confirm/reject vs the label. Prints per-model recall / specificity /
accuracy / valid-JSON / latency, plus a per-rule breakdown.

Three A/B toggles for improving the gate (test which actually moves the numbers):
  --gate-frames N        multi-frame: send N frames SPANNING the motion (not 1 still)
  --cot                  chain-of-thought: the model reasons step-by-step before verdict
  --use-agent-mapper     ground the gate with the Agent Mapper's scene description
                         (e.g. "a wig/hair-products shop") instead of a generic one

SETUP: install Ollama, `brew services start ollama`, pull the models, then e.g.:
  python tools/gate_bakeoff.py --models gemma3:4b --gate-frames 4 --cot --use-agent-mapper --verbose
  `--models mock` runs the wiring without Ollama.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from concealment import ConcealmentDetector  # noqa: E402
from customization import CandidateAlert  # noqa: E402
from detector import (  # noqa: E402
    CONCEALMENT_BAG_CLASSES, PosePersonState, ViolenceTemporalGate, assess_violence,
    assign_pose_tracks, enrich_pose_people_with_history, extract_pose_people,
    load_detection_model, load_ultralytics_model, merge_detections, normalize_label,
    normalize_threat_classes, pose_people_to_concealment_frames, predict_with_model,
    validate_weapon_detections,
)
from verification_gate import VerificationGate  # noqa: E402

os.environ.setdefault("OLLAMA_API_KEY", "ollama")

THREAT = normalize_threat_classes("person")
PERSON = normalize_threat_classes("person")
WEAPON = normalize_threat_classes("knife,gun")
RULE_TO_GATE = {"concealment": "shoplifting", "violence": "violence_in_store", "weapon": "weapon_sighting"}


def scan_clip(clip: str, models: dict, args) -> tuple[dict, float, object]:
    """One detector pass; return per-rule [(frame_pos, score), ...], fps, and a mid frame."""
    pose_model, default_model, weapon_model = models["pose"], models["default"], models["weapon"]
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps if fps > 1e-3 else 1.0 / 30.0
    conceal_det = ConcealmentDetector()
    vio_gate = ViolenceTemporalGate(window=8, min_votes=3)
    prev_pose: list[PosePersonState] = []
    pose_hist: dict = {}
    next_tid = 1
    records = {"concealment": [], "violence": [], "weapon": []}
    mid = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if args.max_frames and idx > args.max_frames:
            break
        if (idx - 1) % args.stride != 0:
            continue
        if mid is None or idx % 60 == 0:
            mid = frame.copy()
        pos, ts = idx - 1, idx * dt

        detections = predict_with_model(default_model, frame, conf=0.35, imgsz=args.imgsz,
                                        threat_classes=THREAT, source_model="default", use_tracking=False)
        validated = []
        if weapon_model is not None:
            detections = merge_detections(detections, predict_with_model(
                weapon_model, frame, conf=0.35, imgsz=args.imgsz, threat_classes=THREAT, source_model="weapon"))
        pose_people = extract_pose_people(pose_model, frame, conf=0.4, imgsz=args.imgsz)
        pose_people, next_tid = assign_pose_tracks(pose_people, previous_people=prev_pose, next_track_id=next_tid)
        pose_people = enrich_pose_people_with_history(pose_people, pose_hist)
        prev_pose = list(pose_people)
        if weapon_model is not None:
            validated = validate_weapon_detections(
                detections=detections, weapon_classes=WEAPON, person_classes=PERSON, pose_people=pose_people,
                frame_shape=frame.shape, weapon_min_area_ratio=0.002, weapon_max_area_ratio=0.18,
                weapon_border_margin_ratio=0.03, weapon_hand_distance_ratio=0.20, allow_unattached_weapons=False)
        bag = [d.bbox for d in detections if normalize_label(d.label) in CONCEALMENT_BAG_CLASSES]
        conceal = conceal_det.update(pose_people_to_concealment_frames(pose_people, ts), ts, bag_bboxes=bag or None)
        records["concealment"].append((pos, max([a.score for a in conceal], default=0.0)))
        records["violence"].append((pos, 1.0 if vio_gate.update(assess_violence(
            pose_people=pose_people, validated_weapon_detections=validated, violence_distance_ratio=1.1,
            violence_wrist_speed=120.0, violence_arm_extension_ratio=0.35, weapon_hand_distance_ratio=0.20,
            violence_wrist_accel=800.0)).active else 0.0))
        records["weapon"].append((pos, float(len(validated))))
    cap.release()
    return records, fps, mid


def pick_frames(clip: str, rule_records: list, fps: float, n: int, span_seconds: float) -> list | None:
    """Re-read n frames SPANNING [peak-span, peak] (lead-up motion + peak). n=1 -> just the peak."""
    if not rule_records:
        return None
    peak_pos, peak_score = max(rule_records, key=lambda r: r[1])
    if peak_score <= 0 and n <= 1:
        peak_pos = rule_records[len(rule_records) // 2][0]  # nothing fired: use a mid frame
    if n <= 1:
        idxs = [peak_pos]
    else:
        window = int(span_seconds * fps)
        start = max(0, peak_pos - window)
        idxs = sorted({int(round(start + i * (peak_pos - start) / (n - 1))) for i in range(n)})
    cap = cv2.VideoCapture(clip)
    frames = []
    for pos in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames or None


def map_scene(frame, mapper_model: str) -> dict | None:
    """Run the Agent Mapper (local Ollama) on a frame -> {environment_type, scene_description}."""
    try:
        import agent_mapper as am
        prompt = am.build_prompt(am.load_text_file(am.DEFAULT_PROMPT_PATH),
                                 camera_id="bakeoff", source_type="video_file",
                                 source_frame_path="frame.jpg", max_zone_suggestions=4)
        raw = am.call_provider("openai_compatible", prompt, am.encode_frame_as_jpeg_bytes(frame),
                               frame.shape, "bakeoff", "video_file", model=mapper_model,
                               api_key_env="OLLAMA_API_KEY", api_base_url="http://localhost:11434/v1")
        sc = am.parse_and_validate_scene_context(raw, am.load_schema(am.DEFAULT_SCHEMA_PATH),
                                                 "bakeoff", "video_file", "frame.jpg")
        return {"environment_type": sc["environment_type"], "scene_description": sc["scene_description"]}
    except Exception as exc:  # noqa: BLE001
        print(f"    [agent-mapper failed, using generic context: {str(exc)[:60]}]")
        return None


def make_gate(model: str, save_dir: Path, cot: bool) -> VerificationGate:
    if model == "mock":
        return VerificationGate(provider="mock", save_dir=save_dir / "mock", cot=cot)
    return VerificationGate(provider="ollama", model=model, save_dir=save_dir / model.replace(":", "_"), cot=cot)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare local VLMs on the gate task, with A/B toggles.")
    ap.add_argument("--models", required=True, help="Comma-separated Ollama tags (or 'mock').")
    ap.add_argument("--labels", default="tools/gate_bakeoff_labels.json")
    ap.add_argument("--gate-frames", type=int, default=1, help="Frames sent per clip (multi-frame spanning).")
    ap.add_argument("--span-seconds", type=float, default=1.0, help="Time window the multi-frames span.")
    ap.add_argument("--cot", action="store_true", help="Chain-of-thought gate prompt.")
    ap.add_argument("--use-agent-mapper", action="store_true", help="Ground the gate with Agent Mapper scene context.")
    ap.add_argument("--mapper-model", default="gemma3:4b", help="Ollama model for the Agent Mapper.")
    ap.add_argument("--pose-weights", default="yolov8n-pose.pt")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--weapon-weights", default="models/weapon_best.pt")
    ap.add_argument("--weapon-loader", default="yolov5")
    ap.add_argument("--yolov5-repo", default="external/yolov5")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole clip (recommended).")
    ap.add_argument("--frames-dir", default="runs/bakeoff_frames")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    spec = json.loads(Path(args.labels).read_text())
    clips = [c for c in spec["clips"] if Path(c["clip"]).exists()]
    generic_ctx = {"environment_type": "retail_shop", "scene_description": spec.get("scene_description", "")}
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    need_weapon = any(c["rule"] == "weapon" for c in clips)

    print(f"Config: gate_frames={args.gate_frames} cot={args.cot} agent_mapper={args.use_agent_mapper} "
          f"| {len(clips)} clips, {len(models)} model(s)")
    print("Loading detectors + selecting frames per clip (once)...")
    loaded = {
        "pose": load_ultralytics_model(args.pose_weights),
        "default": load_detection_model(args.weights, args.yolov5_repo),
        "weapon": load_detection_model(args.weapon_weights, args.yolov5_repo, preferred_kind=args.weapon_loader)
                  if need_weapon else None,
    }
    chosen = []
    for c in clips:
        records, fps, mid = scan_clip(c["clip"], loaded, args)
        frames = pick_frames(c["clip"], records[c["rule"]], fps, args.gate_frames, args.span_seconds) or [mid]
        ctx = (map_scene(frames[-1], args.mapper_model) if args.use_agent_mapper else None) or generic_ctx
        name = Path(c["clip"]).stem
        cv2.imwrite(str(out_dir / f"{name}.jpg"), frames[-1])
        chosen.append({"name": name, "rule": c["rule"], "label": c["label"], "frames": frames, "ctx": ctx})
        extra = f" env='{ctx['environment_type']}'" if args.use_agent_mapper else ""
        print(f"  {name:<22} rule={c['rule']:<12} label={c['label']:<9} frames={len(frames)}{extra}")

    rows = []
    for model in models:
        print(f"\n=== {model} ===")
        gate = make_gate(model, out_dir, args.cot)
        t = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "json_ok": 0}
        per_rule = {r: {"correct": 0, "total": 0} for r in RULE_TO_GATE}
        latencies = []
        for item in chosen:
            alert = CandidateAlert(rule_name=RULE_TO_GATE[item["rule"]], priority="high", detector=item["rule"],
                                   title=f"POSSIBLE {item['rule'].upper()}", person_id=1, object_label=None, timestamp=0.0)
            t0 = time.time()
            try:
                v = gate.verify(item["frames"], alert, item["ctx"])
                latencies.append(time.time() - t0)
                confirmed = bool(v.confirmed)
                t["json_ok"] += 1 if (v.reason and not v.reason.startswith("Gate parse error")) else 0
                reason = v.reason
            except Exception as exc:  # noqa: BLE001
                confirmed, reason = False, f"ERROR: {str(exc)[:70]}"
            is_pos = item["label"] == "positive"
            correct = (is_pos and confirmed) or (not is_pos and not confirmed)
            t["tp" if (is_pos and confirmed) else "fn" if is_pos else "fp" if confirmed else "tn"] += 1
            per_rule[item["rule"]]["total"] += 1
            per_rule[item["rule"]]["correct"] += 1 if correct else 0
            if args.verbose:
                print(f"  {'OK ' if correct else 'XX '}{item['name']:<22} {item['rule']:<12} {item['label']:<9} "
                      f"-> confirmed={confirmed} | {reason[:60]}")
        rows.append({"model": model, **t, "total": t["tp"] + t["fn"] + t["tn"] + t["fp"],
                     "lat": (sum(latencies) / len(latencies)) if latencies else 0.0, "per_rule": per_rule})

    def pct(num, den):
        return "  n/a" if den == 0 else f"{100 * num // den:4d}%"

    print(f"\n========= GATE BAKE-OFF (frames={args.gate_frames}, cot={args.cot}, mapper={args.use_agent_mapper}) =========")
    print(f"{'model':<22} {'recall':>7} {'specif':>7} {'acc':>6} {'json':>6} {'lat(s)':>7}")
    for r in rows:
        print(f"{r['model']:<22} {pct(r['tp'], r['tp']+r['fn']):>7} {pct(r['tn'], r['tn']+r['fp']):>7} "
              f"{pct(r['tp']+r['tn'], r['total']):>6} {r['json_ok']:>3}/{r['total']:<2} {r['lat']:>6.1f}")
    print("\nPer-rule accuracy:")
    print(f"{'model':<22} " + " ".join(f"{r:>13}" for r in RULE_TO_GATE))
    for r in rows:
        print(f"{r['model']:<22} " + " ".join(
            f"{pct(r['per_rule'][rule]['correct'], r['per_rule'][rule]['total']):>13}" for rule in RULE_TO_GATE))
    print("\nrecall=caught positives · specificity=cleared normals · acc=overall · Frames saved to", out_dir)


if __name__ == "__main__":
    main()
