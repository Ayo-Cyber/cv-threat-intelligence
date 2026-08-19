"""retail_pipeline.py — the full retail shoplifting orchestra (V1 integration).

Composes the building blocks into ONE customizable pipeline:

    YOLO-pose detect + track
      ├─ RetailZoneMonitor   (shelf zones + dwell)          → 'presence' events
      └─ ConcealmentDetector (pose action + bag/pocket dest) → 'concealment' events
                     │ (merged raw events)
                     ▼
          CustomizationEngine  (user_config.json rules)  → candidate alerts
                     ▼
          VerificationGate     (VLM confirms / rejects)  → ALERT + saved evidence clip

One pose model yields person boxes + track ids + keypoints (feeds both zones and
concealment). An optional object model adds personal-bag detection for the concealment
destination. The gate is throttled (only on a NEW rule match) so the VLM runs a few
times per event, not per frame, and a rolling buffer saves the evidence clip.

This is the retail use case's orchestrator. It REUSES the shared layers
(CustomizationEngine, VerificationGate) rather than duplicating them, and deliberately
does not touch detector.py (Ayo's general threat pipeline) to avoid merge conflicts.

Example (no API key needed — mock gate auto-confirms, proving the wiring):
    python retail_pipeline.py --source data/test_clips/theft_shop_01.mp4 \
        --config configs/retail_pipeline_v1.json --zones configs/retail_zones.theft_shop_01.json \
        --gate-provider mock --show

Real verification (kills false positives) needs an Anthropic key:
    export ANTHROPIC_API_KEY=...; python retail_pipeline.py ... --gate-provider anthropic
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import supervision as sv

from cvti.event_adapters import concealment_to_events, zone_states_to_events
from cvti.retail.concealment import COCO_BAG_IDS, ConcealmentDetector, PoseFrame, keypoints_from_ultralytics
from cvti.retail.zones import RetailZoneMonitor, filter_person_detections, load_zone_config
from cvti.rules.customization import CustomizationEngine
from cvti.verification.gate import VerificationGate
from cvti.logging_setup import get_logger

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full retail shoplifting pipeline.")
    p.add_argument("--source", required=True, help="Video path / RTSP URL / webcam index.")
    p.add_argument("--config", required=True, help="user_config rules JSON, e.g. configs/retail_pipeline_v1.json")
    p.add_argument("--zones", default="", help="Zone geometry JSON (enables shelf zones + loitering).")
    p.add_argument("--pose-weights", default="models/yolov8n-pose.pt")
    p.add_argument("--object-weights", default="models/yolov8n.pt", help="Detector for personal-bag destinations.")
    p.add_argument("--no-bags", action="store_true", help="Disable bag-destination detection.")
    p.add_argument("--tracker", default="configs/bytetrack_retail.yaml")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--gate-provider", default="mock",
                   choices=("mock", "anthropic", "local", "openai_compatible", "openrouter", "ollama"))
    p.add_argument("--gate-model", default="", help="VLM model; blank uses a per-provider default.")
    p.add_argument("--gate-base-url", default="", help="Override VLM endpoint (e.g. local Ollama).")
    p.add_argument("--gate-frames", type=int, default=3,
                   help="How many of the CLEAREST recent frames to send the gate (multi-frame "
                        "verification). 1 = single best frame.")
    p.add_argument("--scene-context", default="", help="Optional scene_context.json from agent_mapper.")
    p.add_argument("--scene-description", default="",
                   help="Free-text scene description fed to the gate when no --scene-context file is given. "
                        "Priming the gate with context markedly improves its judgment.")
    p.add_argument("--environment-type", default="retail_shop", help="Environment type for the gate context.")
    p.add_argument("--simulate-time", default="", help="HH:MM override for time-filtered rules (demo).")
    p.add_argument("--clip-seconds", type=float, default=4.0, help="Rolling evidence-clip length.")
    p.add_argument("--cooldown", type=float, default=5.0, help="Min seconds between saved events.")
    p.add_argument("--output-dir", default="runs/retail")
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", default="", help="Write an annotated .mp4 to this path.")
    return p.parse_args()


def main() -> None:
    # Entrypoint: configure logging before anything can fail.
    from cvti.logging_setup import setup_logging
    setup_logging(component="argus-retail")
    args = _parse_args()
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"Needs ultralytics + opencv ({exc}). Run: pip install -r requirements.txt")

    pose_model = YOLO(args.pose_weights)
    object_model = None if args.no_bags else YOLO(args.object_weights)
    zone_monitor = RetailZoneMonitor(load_zone_config(args.zones)) if args.zones else None
    concealment = ConcealmentDetector()
    engine = CustomizationEngine(args.config)
    out_root = Path(args.output_dir)
    gate = VerificationGate(provider=args.gate_provider, model=args.gate_model,
                            base_url=args.gate_base_url, save_dir=out_root / "gate")

    if args.scene_context:
        scene_context = json.loads(Path(args.scene_context).read_text())
    elif args.scene_description:
        scene_context = {"environment_type": args.environment_type, "scene_description": args.scene_description}
    else:
        scene_context = None
    sim_now = None
    if args.simulate_time:
        hh, mm = (int(x) for x in args.simulate_time.split(":"))
        sim_now = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    dt = 1.0 / fps if fps > 1e-3 else 1.0 / 25.0
    buffer: deque = deque(maxlen=max(1, int(fps * args.clip_seconds)))
    # Per-frame (frame, {track_id: concealment_score}) — lets the gate be sent the CLEAREST
    # frames of the candidate, not the first (often blurry) one.
    evidence: deque = deque(maxlen=max(1, int(fps * args.clip_seconds)))

    log.info(f"[retail_pipeline] zones={'on' if zone_monitor else 'off'} "
          f"bags={'off' if args.no_bags else 'on'} gate={args.gate_provider} rules={args.config}"
          + (f" simulated_time={args.simulate_time}" if args.simulate_time else ""))

    last_rule_sig: str | None = None
    last_event_t = -1e9
    event_count = 0
    alert_until = -1.0
    alert_text = ""
    writer = None
    n = 0

    for result in pose_model.track(source=source, stream=True, persist=True, tracker=args.tracker,
                                   conf=args.conf, classes=[0], verbose=False):
        frame = result.orig_img
        ts = n * dt
        frame_copy = frame.copy()
        buffer.append(frame_copy)

        detections = sv.Detections.from_ultralytics(result)
        detections = filter_person_detections(detections, frame.shape[:2])

        zone_states = zone_monitor.update(detections, ts) if zone_monitor else []

        pose_frames: list[PoseFrame] = []
        ids = result.boxes.id if result.boxes is not None else None
        if ids is not None and result.keypoints is not None:
            for i in range(len(ids)):
                pose_frames.append(PoseFrame(
                    track_id=int(ids[i]), timestamp=ts,
                    keypoints=keypoints_from_ultralytics(result, i),
                    bbox=tuple(float(v) for v in result.boxes.xyxy[i]),
                ))

        bag_bboxes = None
        if object_model is not None:
            obj = object_model(frame, classes=list(COCO_BAG_IDS), conf=0.35, verbose=False)[0]
            if obj.boxes is not None and len(obj.boxes) > 0:
                bag_bboxes = [tuple(float(v) for v in b) for b in obj.boxes.xyxy]
        conceal = concealment.update(pose_frames, ts, bag_bboxes=bag_bboxes)
        evidence.append((frame_copy, {a.track_id: a.score for a in conceal}))

        # Merge every signal into one event stream, then let the USER's rules decide.
        raw_events = zone_states_to_events(zone_states, ts) + concealment_to_events(conceal, ts)
        alerts = engine.evaluate(raw_events, scene_context=scene_context, now=sim_now)
        top = alerts[0] if alerts else None

        # Verification gate — throttled to a NEW rule match so the VLM runs per-event, not per-frame.
        if top is not None:
            sig = f"{top.rule_name}:{top.person_id}"
            if sig != last_rule_sig:
                last_rule_sig = sig
                evidence_frames = _best_frames(evidence, top.person_id, args.gate_frames) or [frame]
                try:
                    verdict = gate.verify(evidence_frames, top, scene_context)
                except Exception as exc:  # noqa: BLE001 - a transient gate/API error must not kill the run
                    log.error(f"[gate error] {top.rule_name} person #{top.person_id} — "
                          f"{str(exc)[:140]} (alert held, not raised)")
                    verdict = None
                if verdict is not None and verdict.confirmed and (ts - last_event_t) >= args.cooldown:
                    event_count += 1
                    last_event_t = ts
                    alert_until, alert_text = ts + 2.0, f"{top.rule_name.upper()} ({top.priority}) {verdict.confidence:.2f}"
                    log.info(f"[ALERT #{event_count}] {top.rule_name} ({top.priority.upper()}) "
                          f"person #{top.person_id} — {top.title} | confirmed {verdict.confidence:.2f} | {verdict.reason}")
                    _save_event(cv2, out_root, event_count, list(buffer), fps, top, verdict)
                elif verdict is not None and not verdict.confirmed:
                    log.info(f"[rejected] {top.rule_name} person #{top.person_id} — {verdict.reason}")
        else:
            last_rule_sig = None

        if args.show or args.save:
            annotated = zone_monitor.annotate(frame.copy(), detections, zone_states) if zone_monitor else frame.copy()
            annotated = _draw(cv2, annotated, pose_frames, {a.track_id: a for a in conceal},
                              alert_text if ts < alert_until else "")
            if args.save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(annotated)
            if args.show:
                cv2.imshow("retail_pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        n += 1

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    log.info(f"\n=== done: {n} frames, {event_count} confirmed alert(s) -> {out_root} ===")


def _best_frames(evidence: deque, person_id: Any, k: int) -> list | None:
    """Pick the k highest-concealment-score frames for this person, in chronological order.

    Fixes the single-frame gate problem: the first frame a candidate fires on is often
    blurry/ambiguous; the clearest evidence is the peak-score frames around it.
    """
    items = list(evidence)  # chronological
    scored = [(i, scores.get(person_id, 0.0)) for i, (_, scores) in enumerate(items)]
    scored = [(i, s) for i, s in scored if s > 0.0]
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    keep = sorted(i for i, _ in scored[: max(1, k)])
    return [items[i][0] for i in keep]


def _save_event(cv2: Any, out_root: Path, idx: int, frames: list, fps: float, alert: Any, verdict: Any) -> None:
    d = out_root / f"event_{idx:04d}"
    d.mkdir(parents=True, exist_ok=True)
    if frames:
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(d / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
    (d / "alert.json").write_text(
        json.dumps({**alert.to_dict(), "verification": verdict.to_dict()}, indent=2), encoding="utf-8"
    )


def _draw(cv2: Any, frame: Any, pose_frames: list[PoseFrame], conceal_by_id: dict, alert_text: str) -> Any:
    for pf in pose_frames:
        a = conceal_by_id.get(pf.track_id)
        if a is None or pf.bbox is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in pf.bbox)
        color = (0, 0, 255) if a.candidate else (0, 170, 255) if a.score >= 0.4 else (0, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"CONCEAL>{(a.destination or '?').upper()}" if a.candidate else f"{a.score:.2f}"
        cv2.putText(frame, f"#{pf.track_id} {tag}", (x1 + 3, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if alert_text:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 200), -1)
        cv2.putText(frame, f"ALERT: {alert_text}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


if __name__ == "__main__":
    main()
