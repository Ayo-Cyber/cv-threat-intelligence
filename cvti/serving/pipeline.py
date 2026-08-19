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
import json
import time
from collections import Counter
from typing import Any, Callable

from cvti.serving.batcher import collect_batch
from cvti.serving.streams import Frame, StreamDecoder
from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Handler signature: (frame, ultralytics_result) -> None
ResultHandler = Callable[[Frame, Any], None]


def _gate_workers_for(requested: int, n_cameras: int) -> int:
    """How many VLM workers to run. 0 = derive from camera count.

    A verdict takes seconds, so on a multi-camera site alerts queue behind a
    single worker: measured on a 5-camera run, median detection->verified latency
    was 46.5s with one worker and 28.0s with two, for no extra memory. Capped at 3
    because they contend for the same local model.
    """
    if requested > 0:
        return requested
    return max(1, min(3, n_cameras // 2))


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
                 camera_states: dict[str, Any] | None = None, alert_queue: Any = None,
                 publisher: Any = None) -> None:
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
        # Publishes frames (with boxes) so the UI never decodes the stream twice.
        self.publisher = publisher
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
        log.info(f"[serving] {len(self._decoders)} camera(s) | device={self.device} "
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
        # `process` guards its detector section, but everything around it —
        # tracking, zones, rule evaluation, evidence selection — was unguarded,
        # so a failure there propagated out through run() and stopped EVERY
        # camera. The comment inside promised one bad detector could not kill the
        # camera loop; this is what makes that true.
        try:
            alerts = state.process(detections, frame.image, frame.timestamp,
                                   object_detections=object_detections)
        except Exception as exc:  # noqa: BLE001 - one camera must not stop the rest
            state._health.failed(exc, log, "processing a frame")
            return
        for alert in alerts:
            if self._alert_queue.add(alert):
                self.alerts_queued += 1
        # Publish this frame for the UI instead of letting it decode the stream a
        # second time. We already have the frame AND the tracks, so the boxes are free.
        if self.publisher is not None:
            boxes = [(tid, *box) for tid, box in
                     (getattr(state, "_box_by_track", None) or {}).items()]
            if alerts:
                self.publisher.mark_alerting(
                    frame.camera_id, {a.track_id for a in alerts if a.track_id is not None})
            self.publisher.publish(frame.camera_id, frame.image, boxes)

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
                log.info("[serving] all streams ended")
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
        log.info(f"[{tag}] batches={self.batches} frames={self.frames_processed} "
              f"avg_batch={avg_batch:.1f} detect={avg_ms:.1f}ms/batch "
              f"throughput={fps:.0f} frames/s | batch_sizes={dict(sorted(self.batch_hist.items()))}")
        if final:
            if self._camera_states is not None:
                log.info(f"[FINAL] alerts_queued={self.alerts_queued}")
            else:
                log.info(f"[FINAL] detections/camera={dict(self.per_cam)}")

    def stop(self) -> None:
        for d in self._decoders.values():
            d.stop()


def run_site(site_config_path: str, *, weights: str = "models/yolov8n.pt",
             target_fps: float = 5.0, imgsz: int = 640, conf: float = 0.4,
             device: str = "", half: bool = False, seconds: float = 90.0,
             gate_provider: str = "mock", gate_model: str = "", gate_base_url: str = "",
             gate_sensitivity: str = "balanced", publish_frames: bool = True,
             memory_guard: bool = True, memory_warn_gb: float = 2.0,
             memory_critical_gb: float = 1.0,
             pose_weights: str = "models/yolov8n-pose.pt",
             weapon_weights: str = "models/weapon_best.pt", yolov5_repo: str = "external/yolov5",
             video_action_model_path: str = "runs/video_finetune/videomae",
             baseline_config: str | None = "configs/baseline_critical_v1.json",
             notify: str = "console", output_dir: str = "runs/serving",
             gate_workers: int = 0, gate_drain: float = 180.0) -> None:
    """End-to-end multi-camera run: shared batched detector + shared pose/weapon
    models -> per-camera track/zones/concealment/violence/weapons/theft/rules ->
    shared alert queue -> async VLM gate."""
    from pathlib import Path

    from cvti.serving.alert_queue import AlertQueue
    from cvti.serving.camera import build_camera_states, load_site_config
    from cvti.serving.gate_pool import GatePool
    from cvti.verification.gate import MOCK_GATE_BANNER, VerificationGate, assert_engine_gate_allowed

    # Before anything expensive loads: a mock gate confirms every alert without
    # looking at it. Refuse unless the operator asked for that by name.
    mock_gate = assert_engine_gate_allowed(gate_provider)
    if mock_gate:
        log.info(f"[site] *** {MOCK_GATE_BANNER} *** every alert will be confirmed WITHOUT verification.")

    site = load_site_config(site_config_path)
    cams_cfg = site["cameras"]
    # Load ONE shared pose model iff any camera enables a pose-based signal.
    pose_model = None
    if any(c.get(k) for c in cams_cfg for k in ("concealment", "violence", "theft")):
        from cvti.detector.core import load_ultralytics_model
        pose_model = load_ultralytics_model(pose_weights)
        log.info(f"[site] shared pose model loaded ({pose_weights})")
    # Load ONE shared weapon model iff any camera enables weapons (best-effort).
    weapon_model = None
    if any(c.get("weapons") for c in cams_cfg):
        try:
            from cvti.detector.core import load_detection_model
            weapon_model = load_detection_model(weapon_weights, yolov5_repo, preferred_kind="yolov5")
            log.info(f"[site] shared weapon model loaded ({weapon_weights})")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[site] weapon model unavailable ({str(exc)[:80]}); weapons disabled", exc_info=True)
    # Load ONE shared video-action model iff any camera enables it (best-effort).
    video_action_model = None
    if any(c.get("video_action") for c in cams_cfg):
        try:
            from cvti.video_action_model import VideoMAEActionModel
            video_action_model = VideoMAEActionModel(video_action_model_path)
            log.info(f"[site] shared video-action model loaded ({video_action_model_path})")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[site] video-action model unavailable ({str(exc)[:80]}); disabled", exc_info=True)
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
    # Feedback loop: give the gate this site's recent operator-labeled examples for
    # each (camera, rule) as few-shot memory (cached ~60s). Empty on a fresh DB.
    from cvti.feedback.store import FeedbackStore
    _fb_store = FeedbackStore(str(Path(output_dir) / "events.db"))
    _fb_cache: dict = {}

    def _examples_provider(camera: str, rule: str) -> list:
        import time as _t
        key = (camera, rule)
        ent = _fb_cache.get(key)
        if ent and (_t.time() - ent[0]) < 60:
            return ent[1]
        ex = _fb_store.examples(camera, rule, k=4)
        _fb_cache[key] = (_t.time(), ex)
        return ex

    gate_pool = GatePool(
        queue,
        gate_factory=lambda: VerificationGate(provider=gate_provider, model=gate_model,
                                              base_url=gate_base_url, save_dir=save_dir,
                                              sensitivity=gate_sensitivity),
        workers=_gate_workers_for(gate_workers, len(cams_cfg)),
        on_verdict=sink.handle,
        examples_provider=_examples_provider,
    ).start()

    # The UI reads frames from here instead of opening every stream a second time
    # (decode is the dominant per-camera cost) — and gets live boxes for free.
    from cvti.serving.frame_publisher import FramePublisher
    publisher = FramePublisher().start(output_dir) if publish_frames else None

    pipe = MultiStreamPipeline(sources, weights=weights, target_fps=target_fps, imgsz=imgsz,
                               conf=conf, device=device, half=half, camera_states=states,
                               alert_queue=queue, publisher=publisher)
    pipe.start()
    # Live agent mapping: infer each camera's scene (reusing the local VLM) so the
    # gate reasons with real context. Background, non-blocking; only for a local
    # VLM gate (ollama/local). Cameras run generic until their scene lands.
    custom_scanner = None
    watch_runner = None      # defined here: teardown references it on every path
    if gate_provider in ("ollama", "local"):
        from cvti.serving.scene_map import map_cameras_async
        map_cameras_async(cams_cfg, states, model=gate_model or "gemma3:4b",
                          base_url=gate_base_url or "http://localhost:11434/v1")
        # Watches: plain-English subjects to FOLLOW. Binds a description to a
        # tracked person (via numbered boxes) and keeps a case open for them.
        if any(c.get("watches") for c in cams_cfg):
            from cvti.serving.watch_runner import WatchRunner

            def _latest_frame(cam_id: str):
                d = pipe._decoders.get(cam_id)
                f = d.read_latest() if d else None
                return f.image if f is not None else None

            watch_runner = WatchRunner(
                cams_cfg, states, sink, model=gate_model or "gemma3:4b",
                base_url=gate_base_url or "http://localhost:11434/v1",
                frame_source=_latest_frame).start()

        # Customer-written rules run as a slow VLM scan (the VLM is the detector).
        if any(c.get("custom_threats") for c in cams_cfg):
            from cvti.serving.custom_rules import CustomRuleScanner
            custom_scanner = CustomRuleScanner(
                cams_cfg, sink, model=gate_model or "gemma3:4b",
                base_url=gate_base_url or "http://localhost:11434/v1").start()
    # Watch our own footprint: on an edge box, running out means swapping, and a
    # swapping box misses every camera. Shed load on purpose instead.
    mem_guard = None
    if memory_guard:
        from cvti.serving.memory_guard import MemoryGuard, build_default_actions
        warn_actions, crit_actions = build_default_actions(pipe, states)
        mem_guard = MemoryGuard(warn_available_gb=memory_warn_gb,
                                critical_available_gb=memory_critical_gb,
                                warn_actions=warn_actions,
                                critical_actions=crit_actions).start()

    log.info(f"[site] {len(states)} camera(s) | gate={gate_provider} | notify={notify} | rules per camera")

    # Escalation ticker: re-notify alerts nobody acknowledged in time. Cheap poll,
    # daemon so it never holds shutdown open.
    import threading as _threading
    _esc_stop = _threading.Event()

    def _escalation_loop() -> None:
        while not _esc_stop.wait(30.0):
            try:
                sink.run_escalations()
            except Exception as exc:  # noqa: BLE001
                log.error(f"[routing] escalation tick failed: {str(exc)[:90]}", exc_info=True)

    _esc_thread = _threading.Thread(target=_escalation_loop, name="escalations", daemon=True)
    _esc_thread.start()

    # Gate health, published for the UI's System panel. The engine is a separate
    # process from the app, so a file next to events.db is the channel — same
    # trick the frame publisher uses.
    _health_path = Path(output_dir) / "gate_health.json"
    # Last totals written to the ledger, so each tick contributes only its delta.
    _ledger_seen = {"confirmed": 0, "rejected": 0, "deduped": 0, "errors": 0}

    def _write_health() -> None:
        from cvti.health import snapshot as health_snapshot
        st = gate_pool.stats()
        st.update({"provider": gate_provider, "model": gate_model,
                   "mock": mock_gate, "banner": MOCK_GATE_BANNER if mock_gate else "",
                   "updated_at": time.time()})
        # Per-component counters. Without these, "this detector found nothing"
        # and "this detector has thrown on every frame for a week" are the same
        # silence.
        st["health"] = health_snapshot()
        try:
            _health_path.write_text(json.dumps(st))
        except OSError:
            pass   # health reporting must never take the engine down
        # The value surface needs these to outlive the process — see
        # AlertSink.record_suppression.
        try:
            delta = {k: (st.get(k) or 0) - v for k, v in _ledger_seen.items()}
            sink.record_suppression(shown=delta["confirmed"], rejected=delta["rejected"],
                                    deduped=delta["deduped"], errors=delta["errors"])
            for k in _ledger_seen:
                _ledger_seen[k] = st.get(k) or 0
        except Exception as exc:  # noqa: BLE001 - bookkeeping never stops the engine
            log.error(f"[value] suppression ledger write failed: {str(exc)[:90]}", exc_info=True)

    def _health_loop() -> None:
        while not _esc_stop.wait(3.0):
            _write_health()

    _write_health()
    _threading.Thread(target=_health_loop, name="gate-health", daemon=True).start()

    try:
        pipe.run(max_seconds=seconds)
    finally:
        _esc_stop.set()
        pipe.stop()
        if custom_scanner is not None:
            custom_scanner.stop()
        if watch_runner is not None:
            watch_runner.stop()
            act = watch_runner.active_cases()
            if act or watch_runner.opened:
                log.info(f"[watch] cases opened={watch_runner.opened} still_active={len(act)}")
        # Let in-flight VLM verdicts finish (a real gate is ~12s/verify, so the
        # queue keeps draining after the streams end) before tearing down.
        pending = queue.pending_count
        if pending or gate_pool._active:
            log.info(f"[site] draining gate: {pending} queued verdict(s) (up to {gate_drain:.0f}s)…")
            drained = gate_pool.drain(timeout=gate_drain)
            log.info(f"[site] gate drained cleanly={drained}")
        gate_pool.stop()
        if mem_guard is not None:
            mem_guard.stop()
            if mem_guard.mitigations:
                log.warning(f"[memory] shed load {len(mem_guard.mitigations)} time(s): "
                      f"{'; '.join(mem_guard.mitigations)}")
        if publisher is not None:
            log.info(f"[frames] published {publisher.published} frame(s)")
            publisher.stop()
        _write_health()          # final totals, while the sink's DB is still open
        sink.close()
    log.info(f"[site] alerts_queued={pipe.alerts_queued} gate={gate_pool.stats()}")
    log.info(f"[site] persisted {sink.persisted} confirmed event(s) -> {output_dir}/events.db")
    if getattr(sink, "routing", None) and sink.routing.rules:
        log.info(f"[site] routed={sink.routed} escalated={sink.escalated} "
              f"pending_escalation={sink.escalations.pending_count}")


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
    p.add_argument("--no-memory-guard", action="store_true",
                   help="Do not shed load under memory pressure (may swap).")
    p.add_argument("--memory-warn-gb", type=float, default=2.0)
    p.add_argument("--memory-critical-gb", type=float, default=1.0)
    p.add_argument("--no-publish-frames", action="store_true",
                   help="Do not serve frames to the UI (it will decode streams itself).")
    p.add_argument("--gate-sensitivity", choices=("sensitive", "balanced", "strict"),
                   default="balanced",
                   help="Verification strictness. Measured on held-out clips: balanced = "
                        "recall 89%%/FPR 26%%, strict = recall 78%%/FPR 15%%.")
    p.add_argument("--gate-workers", type=int, default=0,
                   help="Concurrent VLM gate workers; 0 = auto from camera count.")
    p.add_argument("--gate-drain", type=float, default=180.0,
                   help="Seconds to let in-flight verdicts finish after streams end.")
    p.add_argument("--notify", default="console",
                   help="Alert notifier: console | webhook:<url> | telegram:<token>:<chat_id> "
                        "| whatsapp (Twilio creds from env)")
    p.add_argument("--output-dir", default="runs/serving",
                   help="Where confirmed events + evidence + events.db are written.")
    args = p.parse_args()

    # Before anything else: a failure during model loading is exactly the kind
    # that used to vanish. Component-scoped because the app runs this as a
    # subprocess pointed at the same output dir.
    from cvti.logging_setup import setup_logging
    setup_logging(args.output_dir, component="argus-engine")

    # Fail fast and readably — a traceback here reads as a crash, and this is a
    # deliberate refusal the operator needs to act on.
    from cvti.verification.gate import MockGateRefused, assert_engine_gate_allowed
    try:
        assert_engine_gate_allowed(args.gate_provider)
    except MockGateRefused as exc:
        log.info(f"[site] {exc}")
        raise SystemExit(2)

    if args.site_config:
        run_site(args.site_config, weights=args.weights, target_fps=args.target_fps,
                 imgsz=args.imgsz, conf=args.conf, device=args.device, half=args.half,
                 seconds=args.seconds, gate_provider=args.gate_provider,
                 gate_sensitivity=args.gate_sensitivity,
                 publish_frames=not args.no_publish_frames,
                 memory_guard=not args.no_memory_guard,
                 memory_warn_gb=args.memory_warn_gb,
                 memory_critical_gb=args.memory_critical_gb,
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
