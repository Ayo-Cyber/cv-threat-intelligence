"""Multi-stream pipeline orchestrator (plan.md Phase 8.1).

Ties the pieces together in ONE process with a SHARED detector model:

    decoders (latest-frame-drop) -> collect_batch -> ONE batched YOLO pass
      -> scatter per camera -> per-camera handler (tracker/rules seam)

The batched detector is stateless, so frames from all cameras go through one
forward pass. Stateful work (ByteTrack association, zones, rules, the alert
queue) is per camera and lives in the handler — that is the seam where the
existing cvti detector/rules logic gets wired in next (8.1 continued / 8.3).

Run it directly to prove cross-camera batching:

    python -m cvti.serving.pipeline --sources clipA.mp4 clipB.mp4 --seconds 15
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Callable

from cvti.serving.batcher import collect_batch
from cvti.serving.streams import Frame, StreamDecoder

# Handler signature: (frame, ultralytics_result) -> None
ResultHandler = Callable[[Frame, Any], None]


def _auto_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MultiStreamPipeline:
    def __init__(self, sources: dict[str, int | str], *, weights: str = "models/yolov8n.pt",
                 target_fps: float = 5.0, tick_seconds: float = 0.15, imgsz: int = 640,
                 conf: float = 0.4, device: str = "", half: bool = False,
                 max_batch: int = 32, on_result: ResultHandler | None = None,
                 camera_states: dict[str, Any] | None = None, alert_queue: Any = None) -> None:
        self.sources = sources
        self.weights = weights
        self.target_fps = target_fps
        self.tick_seconds = tick_seconds
        self.imgsz = imgsz
        self.conf = conf
        self.device = device or _auto_device()
        self.half = half and self.device == "cuda"
        self.max_batch = max_batch
        self._camera_states = camera_states
        self._alert_queue = alert_queue
        # When per-camera states + a queue are supplied, route detections through
        # them (track -> zones -> rules -> alert queue). Otherwise just count.
        if on_result is not None:
            self.on_result = on_result
        elif camera_states is not None and alert_queue is not None:
            self.on_result = self._route_to_queue
        else:
            self.on_result = self._default_handler
        self._decoders: dict[str, StreamDecoder] = {}
        self._model = None
        # stats
        self.batches = 0
        self.frames_processed = 0
        self.batch_hist: Counter = Counter()
        self.per_cam: Counter = Counter()
        self.alerts_queued = 0
        self._detect_ms_total = 0.0

    def start(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self.weights)
        # For the per-camera detector path we also need core.py's Detection list
        # (weapons/violence/theft), built from the same shared-model result.
        self._names = self._model.names
        from cvti.detector.core import normalize_threat_classes
        self._threat_classes = normalize_threat_classes("gun,knife")
        for cam_id, src in self.sources.items():
            self._decoders[cam_id] = StreamDecoder(cam_id, src, target_fps=self.target_fps).start()
        print(f"[serving] {len(self._decoders)} camera(s) | device={self.device} "
              f"half={self.half} target_fps={self.target_fps} | model={self.weights}")

    def _default_handler(self, frame: Frame, result: Any) -> None:
        n = 0 if result.boxes is None else len(result.boxes)
        self.per_cam[frame.camera_id] += n

    def _route_to_queue(self, frame: Frame, result: Any) -> None:
        import supervision as sv
        from cvti.detector.core import extract_detections
        state = self._camera_states.get(frame.camera_id)
        if state is None:
            return
        detections = sv.Detections.from_ultralytics(result)          # tracking / zones
        object_detections = extract_detections(result, self._names, self._threat_classes)  # weapons/violence/theft
        for alert in state.process(detections, frame.image, frame.timestamp,
                                   object_detections=object_detections):
            if self._alert_queue.add(alert):
                self.alerts_queued += 1

    def _all_ended(self) -> bool:
        return all(d.ended and not d.read_latest() for d in self._decoders.values())

    def run(self, max_seconds: float = 20.0) -> None:
        assert self._model is not None, "call start() first"
        t_end = time.perf_counter() + max_seconds
        last_report = time.perf_counter()
        while time.perf_counter() < t_end:
            tick = time.perf_counter()
            batch = collect_batch(self._decoders, max_batch=self.max_batch)
            if batch:
                images = [f.image for f in batch]
                t0 = time.perf_counter()
                results = self._model.predict(images, imgsz=self.imgsz, conf=self.conf,
                                              device=self.device, half=self.half, verbose=False)
                self._detect_ms_total += (time.perf_counter() - t0) * 1000.0
                for frame, result in zip(batch, results):
                    self.on_result(frame, result)
                self.batches += 1
                self.frames_processed += len(batch)
                self.batch_hist[len(batch)] += 1
            if time.perf_counter() - last_report >= 3.0:
                self._report()
                last_report = time.perf_counter()
            if self._all_ended():
                print("[serving] all streams ended")
                break
            slept = self.tick_seconds - (time.perf_counter() - tick)
            if slept > 0:
                time.sleep(slept)
        self._report(final=True)

    def _report(self, final: bool = False) -> None:
        avg_batch = (self.frames_processed / self.batches) if self.batches else 0.0
        avg_ms = (self._detect_ms_total / self.batches) if self.batches else 0.0
        fps = (self.frames_processed / (self._detect_ms_total / 1000.0)) if self._detect_ms_total else 0.0
        tag = "FINAL" if final else "stats"
        print(f"[{tag}] batches={self.batches} frames={self.frames_processed} "
              f"avg_batch={avg_batch:.1f} detect={avg_ms:.1f}ms/batch "
              f"throughput={fps:.0f} frames/s | batch_sizes={dict(sorted(self.batch_hist.items()))}")
        if final:
            if self._camera_states is not None:
                print(f"[FINAL] alerts_queued={self.alerts_queued}")
            else:
                print(f"[FINAL] detections/camera={dict(self.per_cam)}")

    def stop(self) -> None:
        for d in self._decoders.values():
            d.stop()


def run_site(site_config_path: str, *, weights: str = "models/yolov8n.pt",
             target_fps: float = 5.0, imgsz: int = 640, conf: float = 0.4,
             device: str = "", half: bool = False, seconds: float = 90.0,
             gate_provider: str = "mock", gate_model: str = "", gate_base_url: str = "",
             pose_weights: str = "models/yolov8n-pose.pt",
             weapon_weights: str = "models/weapon_best.pt", yolov5_repo: str = "external/yolov5",
             video_action_model_path: str = "runs/video_finetune/videomae",
             baseline_config: str | None = "configs/baseline_critical_v1.json",
             notify: str = "console", output_dir: str = "runs/serving",
             gate_workers: int = 1, gate_drain: float = 180.0) -> None:
    """End-to-end multi-camera run: shared batched detector + shared pose/weapon
    models -> per-camera track/zones/concealment/violence/weapons/theft/rules ->
    shared alert queue -> async VLM gate."""
    from pathlib import Path

    from cvti.serving.alert_queue import AlertQueue
    from cvti.serving.camera import build_camera_states, load_site_config
    from cvti.serving.gate_pool import GatePool
    from cvti.verification.gate import VerificationGate

    site = load_site_config(site_config_path)
    cams_cfg = site["cameras"]
    # Load ONE shared pose model iff any camera enables a pose-based signal.
    pose_model = None
    if any(c.get(k) for c in cams_cfg for k in ("concealment", "violence", "theft")):
        from cvti.detector.core import load_ultralytics_model
        pose_model = load_ultralytics_model(pose_weights)
        print(f"[site] shared pose model loaded ({pose_weights})")
    # Load ONE shared weapon model iff any camera enables weapons (best-effort).
    weapon_model = None
    if any(c.get("weapons") for c in cams_cfg):
        try:
            from cvti.detector.core import load_detection_model
            weapon_model = load_detection_model(weapon_weights, yolov5_repo, preferred_kind="yolov5")
            print(f"[site] shared weapon model loaded ({weapon_weights})")
        except Exception as exc:  # noqa: BLE001
            print(f"[site] weapon model unavailable ({str(exc)[:80]}); weapons disabled")
    # Load ONE shared video-action model iff any camera enables it (best-effort).
    video_action_model = None
    if any(c.get("video_action") for c in cams_cfg):
        try:
            from cvti.video_action_model import VideoMAEActionModel
            video_action_model = VideoMAEActionModel(video_action_model_path)
            print(f"[site] shared video-action model loaded ({video_action_model_path})")
        except Exception as exc:  # noqa: BLE001
            print(f"[site] video-action model unavailable ({str(exc)[:80]}); disabled")
    cams = build_camera_states(site, pose_model=pose_model, weapon_model=weapon_model,
                               video_action_model=video_action_model, baseline_config=baseline_config)
    sources = {cid: c["source"] for cid, c in cams.items()}
    states = {cid: c["state"] for cid, c in cams.items()}
    queue = AlertQueue()

    # Confirmed alerts are persisted (SQLite + evidence bundle) and notified,
    # instead of only printed. sink.handle is the gate's verdict callback.
    from cvti.serving.alert_sink import AlertSink, build_notifier
    sink = AlertSink(output_dir, notifier=build_notifier(notify))

    save_dir = Path(output_dir) / "gate"
    gate_pool = GatePool(
        queue,
        gate_factory=lambda: VerificationGate(provider=gate_provider, model=gate_model,
                                              base_url=gate_base_url, save_dir=save_dir),
        workers=gate_workers,
        on_verdict=sink.handle,
    ).start()

    pipe = MultiStreamPipeline(sources, weights=weights, target_fps=target_fps, imgsz=imgsz,
                               conf=conf, device=device, half=half, camera_states=states,
                               alert_queue=queue)
    pipe.start()
    print(f"[site] {len(states)} camera(s) | gate={gate_provider} | notify={notify} | rules per camera")
    try:
        pipe.run(max_seconds=seconds)
    finally:
        pipe.stop()
        # Let in-flight VLM verdicts finish (a real gate is ~12s/verify, so the
        # queue keeps draining after the streams end) before tearing down.
        pending = queue.pending_count
        if pending or gate_pool._active:
            print(f"[site] draining gate: {pending} queued verdict(s) (up to {gate_drain:.0f}s)…")
            drained = gate_pool.drain(timeout=gate_drain)
            print(f"[site] gate drained cleanly={drained}")
        gate_pool.stop()
        sink.close()
    print(f"[site] alerts_queued={pipe.alerts_queued} gate={gate_pool.stats()}")
    print(f"[site] persisted {sink.persisted} confirmed event(s) -> {output_dir}/events.db")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 8.1 multi-stream pipeline.")
    p.add_argument("--sources", nargs="+", help="Video files / RTSP URLs / webcam indices (demo mode).")
    p.add_argument("--site-config", default="", help="Site JSON: per-camera source + rules + zones.")
    p.add_argument("--weights", default="models/yolov8n.pt")
    p.add_argument("--target-fps", type=float, default=5.0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--device", default="")
    p.add_argument("--half", action="store_true")
    p.add_argument("--seconds", type=float, default=90.0,
                   help="Run cap; file sources stop automatically at end-of-stream.")
    p.add_argument("--gate-provider", default="mock")
    p.add_argument("--gate-model", default="")
    p.add_argument("--gate-base-url", default="")
    p.add_argument("--gate-workers", type=int, default=1,
                   help="Concurrent VLM gate workers (raise for real/slow gates).")
    p.add_argument("--gate-drain", type=float, default=180.0,
                   help="Seconds to let in-flight verdicts finish after streams end.")
    p.add_argument("--notify", default="console",
                   help="Alert notifier: console | webhook:<url> | telegram:<token>:<chat_id> "
                        "| whatsapp (Twilio creds from env)")
    p.add_argument("--output-dir", default="runs/serving",
                   help="Where confirmed events + evidence + events.db are written.")
    args = p.parse_args()

    if args.site_config:
        run_site(args.site_config, weights=args.weights, target_fps=args.target_fps,
                 imgsz=args.imgsz, conf=args.conf, device=args.device, half=args.half,
                 seconds=args.seconds, gate_provider=args.gate_provider,
                 gate_model=args.gate_model, gate_base_url=args.gate_base_url,
                 notify=args.notify, output_dir=args.output_dir,
                 gate_workers=args.gate_workers, gate_drain=args.gate_drain)
        return

    if not args.sources:
        raise SystemExit("pass --sources <files...> (demo) or --site-config <site.json>")
    sources = {f"cam{i}": s for i, s in enumerate(args.sources)}
    pipe = MultiStreamPipeline(sources, weights=args.weights, target_fps=args.target_fps,
                               imgsz=args.imgsz, conf=args.conf, device=args.device, half=args.half)
    pipe.start()
    try:
        pipe.run(max_seconds=args.seconds)
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
