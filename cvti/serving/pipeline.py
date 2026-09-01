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
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cvti.serving.alert_queue import QueuedAlert
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


def resolve_mapper_settings(
    *,
    gate_provider: str,
    gate_model: str,
    gate_base_url: str,
    mapper_provider: str,
    mapper_model: str,
    mapper_base_url: str,
) -> tuple[str, str, str]:
    supported = {"mock", "ollama", "local", "anthropic", "openai_compatible"}
    provider = mapper_provider.strip() if mapper_provider else ""
    if not provider:
        provider = gate_provider if gate_provider in supported else "ollama"
    model = mapper_model or (gate_model if provider == gate_provider else "")
    base_url = mapper_base_url or (
        gate_base_url if provider == gate_provider else ""
    )
    if provider in {"ollama", "local"} and not base_url:
        base_url = "http://localhost:11434/v1"
    return provider, model, base_url


def build_mapper_service(
    *,
    output_dir: str,
    gate_provider: str,
    gate_model: str,
    gate_base_url: str,
    mapper_provider: str = "",
    mapper_model: str = "",
    mapper_base_url: str = "",
):
    from cvti.scene.agent_mapper import AgentMapper
    from cvti.serving.scene_map import FullAgentMapperService

    provider, model, base_url = resolve_mapper_settings(
        gate_provider=gate_provider,
        gate_model=gate_model,
        gate_base_url=gate_base_url,
        mapper_provider=mapper_provider,
        mapper_model=mapper_model,
        mapper_base_url=mapper_base_url,
    )
    mapper = AgentMapper(provider=provider, model=model, base_url=base_url)
    return FullAgentMapperService(output_dir, mapper)


def prepare_scene_mapping(
    site: dict,
    *,
    output_dir: str,
    gate_provider: str,
    gate_model: str,
    gate_base_url: str,
    mapper_provider: str = "",
    mapper_model: str = "",
    mapper_base_url: str = "",
):
    service = build_mapper_service(
        output_dir=output_dir,
        gate_provider=gate_provider,
        gate_model=gate_model,
        gate_base_url=gate_base_url,
        mapper_provider=mapper_provider,
        mapper_model=mapper_model,
        mapper_base_url=mapper_base_url,
    )
    return service.prepare(
        list(site.get("cameras") or []),
        site.get("scene_context_policy", "auto"),
    )


def retry_failed_scene_mappings(service, cams_cfg: list, policy: str,
                                mapping_health: list, states: dict) -> set:
    """One retry pass over cameras whose scene mapping FAILED at startup.

    On a fresh install the preflight races the TrueSight model download and
    loses: every mapping call fails, cameras run generic, and — before this —
    nothing ever re-mapped once the model landed. The context stayed 'failed'
    until a human noticed (field screenshot, 1 Sep). Retried contexts are
    injected into the running camera states, so the site gets its scene
    intelligence without a restart. Returns the camera ids still failed."""
    failed = {str(row.get("camera_id")) for row in mapping_health
              if row.get("status") == "failed"}
    retry = [c for c in cams_cfg
             if str(c.get("id")) in failed and str(c.get("id")) in states]
    if not retry:
        return set()
    result = service.prepare(retry, policy)
    for camera_id, context in result.contexts.items():
        state = states.get(camera_id)
        if state is not None:
            state.scene_context = context
    for camera_id, status_doc in result.statuses.items():
        for i, row in enumerate(mapping_health):
            if str(row.get("camera_id")) == str(camera_id):
                mapping_health[i] = {"camera_id": camera_id, **status_doc}
    return {str(row.get("camera_id")) for row in mapping_health
            if row.get("status") == "failed"}


def active_cameras_after_preflight(cameras: list[dict], preflight) -> list[dict]:
    _ = preflight
    return [dict(camera) for camera in cameras]


def monitoring_scopes_from_preflight(preflight) -> dict[str, str]:
    return {
        str(camera_id): (
            "full" if status.get("status") == "ready_reviewed" else "critical_only"
        )
        for camera_id, status in preflight.statuses.items()
    }


def activate_reviewed_scene_contexts(states: dict, output_dir: str | Path) -> list[str]:
    """Hot-apply approvals written by the console's separate process."""
    from cvti.scene.context_store import SceneContextStore

    changed: list[str] = []
    context_root = Path(output_dir) / "context"
    for camera_id, state in states.items():
        if getattr(state, "monitoring_scope", "full") == "full":
            continue
        store = SceneContextStore(context_root, camera_id)
        if store.load_status().status != "ready_reviewed":
            continue
        context = store._load_context()
        if context is not None:
            state.activate_scene_context(context)
            changed.append(camera_id)
    return changed


def _mapping_health_rows(preflight) -> list[dict]:
    return [
        {"camera_id": camera_id, **status}
        for camera_id, status in sorted(preflight.statuses.items())
    ]


def write_starting_health(output_dir: str, site: dict, *,
                          gate_provider: str, gate_model: str,
                          phase: str = "starting — mapping camera scenes",
                          scene_mapping: list | None = None) -> None:
    """The engine's first words, written BEFORE anything slow.

    The scene preflight and cold model loads run before the first real
    heartbeat — minutes on a fresh Windows install (Defender scanning the
    bundle, torch loading, the mapper timing out against a model still
    downloading). For all of it the console said 'engine not running' under
    a green Monitoring dot (field screenshot, 1 Sep). This heartbeat makes
    the warm-up visible: every camera reports 'starting', and engine.phase
    says what the engine is actually doing."""
    from pathlib import Path

    from cvti.serving.health_doc import build_health_doc

    cameras = [{"camera_id": str(c.get("id")), "state": "starting",
                "time_in_state": 0, "reconnects": 0}
               for c in site.get("cameras") or []]
    doc = build_health_doc(
        started_at=time.time(),
        cameras=cameras,
        gate={"provider": gate_provider, "model": gate_model, "reachable": None},
        disk={},
        memory={},
        components={"degraded": []},
        engine={"phase": phase, "frames_processed": 0, "alerts_queued": 0,
                "cameras": len(cameras)},
        scene_mapping=scene_mapping or [],
    )
    target = Path(output_dir) / "gate_health.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(doc))
    temporary.replace(target)


def _write_mapping_only_health(
    output_dir: str,
    preflight,
    *,
    gate_provider: str,
    gate_model: str,
) -> None:
    from pathlib import Path

    from cvti.serving.health_doc import build_health_doc

    rows = _mapping_health_rows(preflight)
    doc = build_health_doc(
        started_at=time.time(),
        cameras=[],
        gate={"provider": gate_provider, "model": gate_model, "reachable": None},
        disk={},
        memory={},
        components={"degraded": []},
        engine={"frames_processed": 0, "alerts_queued": 0, "cameras": 0},
        scene_mapping=rows,
    )
    target = Path(output_dir) / "gate_health.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(doc))
    temporary.replace(target)


class MultiStreamPipeline:
    def __init__(self, sources: dict[str, int | str], *, weights: str = "models/yolov8n.pt",
                 target_fps: float = 5.0, tick_seconds: float = 0.15, imgsz: int = 640,
                 conf: float = 0.4, device: str = "", half: bool = False,
                 max_batch: int = 32, on_result: ResultHandler | None = None,
                 camera_states: dict[str, Any] | None = None, alert_queue: Any = None,
                 publisher: Any = None, on_link_change=None,
                 publish_fps: float = 12.0) -> None:
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
        self.publish_fps = publish_fps
        self.smooth_publish = bool(publisher is not None and publish_fps > 0)
        self.latest_boxes: dict = {}       # camera_id -> last detection's boxes
        self._smooth_thread = None
        self._last_detect: dict = {}       # camera_id -> last model-sample time
        self.on_link_change = on_link_change
        # Called the moment an alert is accepted onto the queue — the two-tier
        # fast path hangs off this, ahead of any verification.
        self.on_queued = None
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

        from cvti.detector.core import resolve_weights
        self._model = YOLO(resolve_weights(self.weights))
        # For the per-camera detector path we also need core.py's Detection list
        # (weapons/violence/theft), built from the same shared-model result.
        self._names = self._model.names
        from cvti.detector.core import normalize_threat_classes
        self._threat_classes = normalize_threat_classes("gun,knife")
        decode_fps = max(self.target_fps, self.publish_fps) if self.smooth_publish \
            else self.target_fps
        for cam_id, src in self.sources.items():
            self._decoders[cam_id] = StreamDecoder(
                cam_id, src, target_fps=decode_fps,
                on_state_change=self.on_link_change).start()
        if self.smooth_publish:
            import threading as _th
            self._smooth_thread = _th.Thread(target=self._smooth_publish_loop,
                                             name="smooth-publish", daemon=True)
            self._smooth_thread.start()
            log.info("[serving] live wall decoupled: publishing at %.0f fps, "
                     "detection sampling at %.0f fps", self.publish_fps, self.target_fps)
        log.info(f"[serving] {len(self._decoders)} camera(s) | device={self.device} "
              f"half={self.half} target_fps={self.target_fps} | model={self.weights}")

    def _smooth_publish_loop(self) -> None:
        """Ship the newest decoded frame per camera at publish_fps, overlaying
        the LAST detection's boxes — the live wall stops being chained to the
        model's cadence (user ask, 24 Aug: 'live stream smooth and fast, the
        detection on the other end'). Peek never consumes, so detection loses
        nothing; boxes lag the video by at most one detection interval."""
        last_seq: dict = {}
        period = 1.0 / self.publish_fps
        while True:
            t0 = time.perf_counter()
            for cam_id, d in list(self._decoders.items()):
                try:
                    if d.playout is not None:
                        # Live URL source: paced playout — content plays at its
                        # own rate a bounded lag behind live, instead of
                        # freeze-then-fast-forward at every burst.
                        jpeg = d.playout.pop_due(time.perf_counter())
                        if jpeg is not None:
                            self.publisher.publish_jpeg(cam_id, jpeg)
                        continue
                    frame, seq = d.peek_latest()
                except Exception:  # noqa: BLE001 - one camera must not stop the wall
                    log.debug("peek failed for %s", cam_id, exc_info=True)
                    continue
                if frame is None or last_seq.get(cam_id) == seq:
                    continue
                last_seq[cam_id] = seq
                try:
                    self.publisher.publish(cam_id, frame.image)   # raw glass — no boxes
                except Exception:  # noqa: BLE001
                    log.debug("smooth publish failed for %s", cam_id, exc_info=True)
            sleep = period - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _due_for_detection(self, camera_id: str, now: float) -> bool:
        """Detection samples at target_fps even when decoders run faster for
        the live wall — without this gate, faster decode = 2-3x model load."""
        last = self._last_detect.get(camera_id, 0.0)
        if now - last >= 0.95 / self.target_fps:
            self._last_detect[camera_id] = now
            return True
        return False

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
                if self.on_queued is not None:
                    try:
                        self.on_queued(alert)
                    except Exception:  # noqa: BLE001 - fast path must not stop routing
                        log.error("on_queued callback failed", exc_info=True)
        # The live wall is DECOUPLED from detection (user ask, 24 Aug): the
        # smooth-publish thread ships frames at stream cadence; detection only
        # refreshes the box overlay it draws. Boxes therefore lag the video by
        # at most one detection interval (~200ms) — standard for CCTV overlays.
        if self.publisher is not None:
            boxes = [(tid, *box) for tid, box in
                     (getattr(state, "_box_by_track", None) or {}).items()]
            self.latest_boxes[frame.camera_id] = boxes
            if alerts:
                self.publisher.mark_alerting(
                    frame.camera_id, {a.track_id for a in alerts if a.track_id is not None})
            if not self.smooth_publish:
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
            if batch and self.smooth_publish:
                now = time.perf_counter()
                batch = [f for f in batch if self._due_for_detection(f.camera_id, now)]
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
              f"model_capacity={fps:.0f} frames/s | batch_sizes={dict(sorted(self.batch_hist.items()))}")
        # capacity = frames / MODEL time (1/latency), not a wall-clock rate —
        # the old label "throughput" read as if detection ran that fast.
        if final:
            if self._camera_states is not None:
                log.info(f"[FINAL] alerts_queued={self.alerts_queued}")
            else:
                log.info(f"[FINAL] detections/camera={dict(self.per_cam)}")

    def stop(self) -> None:
        for d in self._decoders.values():
            d.stop()


@dataclass
class _LinkEvent:
    """A camera-link change, shaped like a candidate alert so it routes normally."""
    detector: str
    rule_name: str
    priority: str
    title: str
    person_id: Any = None
    object_label: Any = None
    track_id: Any = None
    zone: Any = None


def gate_reachable(stats: dict, now: float | None = None):
    """True/False/None — with hysteresis, because one bad verdict is not an outage.

    The old rule was 'whichever came last wins': a single timed-out verify
    flipped the status to UNAVAILABLE until the next success — minutes later
    on slow hardware. The pilot's laptop hit exactly this while screen-sharing
    a Meet call over a 90s-median verify (29 Aug): the AI was fine, merely
    busy, and the banner declared it dead and the English rules paused.

    False now requires sustained evidence: failures more recent than the last
    success AND no success within a patience window scaled to the machine's
    own measured latency. A genuinely absent server still reads False fast,
    because it has no last_success_at at all.
    """
    now = now or time.time()
    if not stats.get("verified") and not stats.get("unverified") \
            and not stats.get("errors"):
        return None
    last_ok = stats.get("last_success_at") or 0
    last_bad = stats.get("last_unverified_at") or 0
    if last_bad <= last_ok:
        return True
    patience = max(240.0, 4.0 * float(stats.get("median_latency_s") or 30.0))
    return (now - last_ok) <= patience


def run_site(site_config_path: str, *, weights: str = "models/yolov8n.pt",
             target_fps: float = 5.0, imgsz: int = 640, conf: float = 0.4,
             device: str = "", half: bool = False, seconds: float = 90.0,
             gate_provider: str = "mock", gate_model: str = "", gate_base_url: str = "",
             mapper_provider: str = "", mapper_model: str = "", mapper_base_url: str = "",
             gate_sensitivity: str = "balanced", publish_frames: bool = True,
             publish_fps: float = 12.0, security_dir: str | None = None,
             memory_guard: bool = True, memory_warn_gb: float = 2.0,
             memory_critical_gb: float = 1.0,
             pose_weights: str = "models/yolov8n-pose.pt",
             weapon_weights: str = "models/weapon_best.pt", yolov5_repo: str = "external/yolov5",
             video_action_model_path: str = "runs/video_finetune/videomae",
             baseline_config: str | None = "configs/baseline_critical_v1.json",
             notify: str = "console", output_dir: str = "runs/serving",
             gate_workers: int = 0, gate_drain: float = 180.0,
             mobile_port: int = 8710) -> None:
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
    # Heartbeat THROUGH the slow parts, not just before them: the preflight
    # alone can run minutes on a fresh install (mapper timing out against a
    # model still downloading), and a single stale write would read as a
    # stalled engine. Ticks until the real health loop takes over.
    _starting = {"phase": "starting — mapping camera scenes"}
    _starting_stop = threading.Event()

    def _starting_beat() -> None:
        while not _starting_stop.wait(10.0):
            try:
                write_starting_health(output_dir, site,
                                      gate_provider=gate_provider,
                                      gate_model=gate_model,
                                      phase=_starting["phase"],
                                      scene_mapping=_starting.get("scene"))
            except Exception:  # noqa: BLE001 - the warm-up beat must never kill startup
                log.debug("starting heartbeat write failed", exc_info=True)

    write_starting_health(output_dir, site,
                          gate_provider=gate_provider, gate_model=gate_model)
    threading.Thread(target=_starting_beat, name="starting-beat",
                     daemon=True).start()
    mapping_service = build_mapper_service(
        output_dir=output_dir,
        gate_provider=gate_provider,
        gate_model=gate_model,
        gate_base_url=gate_base_url,
        mapper_provider=mapper_provider,
        mapper_model=mapper_model,
        mapper_base_url=mapper_base_url,
    )

    def _preflight():
        # inspect() reads caches and hierarchy only — no VLM inference — so
        # startup is fast and the wait-loop below can re-check cheaply. The
        # actual mapping happens in the background coordinator once running.
        return mapping_service.inspect(
            list(site.get("cameras") or []),
            site.get("scene_context_policy", "auto"),
        )

    mapping_preflight = _preflight()
    cams_cfg = active_cameras_after_preflight(
        list(site.get("cameras") or []), mapping_preflight
    )
    mapping_health = _mapping_health_rows(mapping_preflight)
    # A strict policy with nothing approved used to EXIT here — and the
    # console's watchdog dutifully respawned the corpse forever. A pilot's
    # fresh install spent two days in that loop reading as 'the engine never
    # starts' (1 Sep). A policy block is a WAITING state, not a death: stay
    # alive, keep the heartbeat honest about why, and re-run the preflight —
    # the moment a human approves a context in the app (or a remap succeeds),
    # cameras start without anyone restarting anything.
    while not cams_cfg:
        _starting["phase"] = (
            "waiting — add a camera to begin monitoring"
            if not site.get("cameras") else
            "waiting — this site's policy requires reviewed "
            "scene context before cameras may start")
        _starting["scene"] = mapping_health
        log.warning("[site] no camera can start until scene context is ready — "
                    "retrying the preflight in 60s (approve or remap in the app)")
        time.sleep(60.0)
        # Reload the config: a camera added (or a context approved) from the
        # app while we wait must start cameras WITHOUT an engine restart.
        site = load_site_config(site_config_path)
        mapping_preflight = _preflight()
        cams_cfg = active_cameras_after_preflight(
            list(site.get("cameras") or []), mapping_preflight
        )
        mapping_health = _mapping_health_rows(mapping_preflight)
    _starting["phase"] = "starting — loading detection models"
    _starting["scene"] = mapping_health
    site = {**site, "cameras": cams_cfg}
    # Load ONE shared pose model iff any camera enables a pose-based signal.
    pose_model = None
    if any(c.get(k) for c in cams_cfg for k in ("concealment", "violence", "theft")):
        from cvti.detector.core import load_ultralytics_model
        pose_model = load_ultralytics_model(pose_weights)
        log.info(f"[site] shared pose model loaded ({pose_weights})")
    # Detectors the operator configured that could NOT actually run. Feeds the
    # health doc: "weapons: true" with a model that failed to load used to be
    # one log line and then permanent silent non-coverage. (Audit 23 Aug, #10.)
    model_failures: list = []
    # Load ONE shared weapon model iff any camera enables weapons (best-effort).
    weapon_model = None
    if any(c.get("weapons") for c in cams_cfg):
        try:
            from cvti.detector.core import load_detection_model
            repo = yolov5_repo
            if not Path(repo).exists():                    # frozen app: resolve inside the bundle
                from cvti.utils import resource_path
                repo = str(resource_path(yolov5_repo))
            weapon_model = load_detection_model(weapon_weights, repo, preferred_kind="yolov5")
            log.info(f"[site] shared weapon model loaded ({weapon_weights})")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[site] weapon model unavailable ({str(exc)[:80]}); weapons disabled", exc_info=True)
            model_failures.append(f"weapons detector configured but its model failed to load: {str(exc)[:90]}")
            # Saying 'weapons disabled' must MAKE IT TRUE: cameras kept their
            # weapons flag with a None model and threw on every frame — 509
            # detector.cam1 errors in 28 minutes on the pilot's install while
            # health claimed the detector was merely 'configured but failed'.
            for c in cams_cfg:
                c["weapons"] = False
    # Load ONE shared video-action model iff any camera enables it (best-effort).
    video_action_model = None
    if any(c.get("video_action") for c in cams_cfg):
        try:
            from cvti.video_action_model import VideoMAEActionModel
            video_action_model = VideoMAEActionModel(video_action_model_path)
            log.info(f"[site] shared video-action model loaded ({video_action_model_path})")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[site] video-action model unavailable ({str(exc)[:80]}); disabled", exc_info=True)
            model_failures.append(f"video-action detector configured but its model failed to load: {str(exc)[:90]}")
    cams = build_camera_states(site, pose_model=pose_model, weapon_model=weapon_model,
                               video_action_model=video_action_model,
                               baseline_config=baseline_config,
                               scene_contexts=mapping_preflight.contexts,
                               monitoring_scopes=monitoring_scopes_from_preflight(
                                   mapping_preflight
                               ))
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
    # Watch is pure glass (user, 25 Aug): the live wall ships RAW frames — no
    # box drawing, no per-frame copy. Boxes belong to the alert, where the
    # sink already writes the annotated subject shot; a live overlay trailing
    # the person by a detection interval looked broken and bought nothing.
    publisher = FramePublisher(draw_boxes=False).start(output_dir) if publish_frames else None

    def _on_link_change(cam_id: str, previous: str, state: str, held: float) -> None:
        """A camera going offline raises its own alert through normal routing.

        Without this the customer believes they have coverage they do not have:
        an unreachable camera produces no alerts, which is indistinguishable
        from a camera watching a quiet area.
        """
        from cvti.serving.streams import CONNECTED, OFFLINE
        if state not in (OFFLINE, CONNECTED) or (state == CONNECTED and previous == "reconnecting"
                                                 and held < 1.0):
            return
        offline = state == OFFLINE
        queue.add(QueuedAlert(
            camera_id=cam_id,
            rule_name="camera_offline" if offline else "camera_recovered",
            priority="high" if offline else "info",
            title=(f"CAMERA OFFLINE — unreachable for {held:.0f}s"
                   if offline else "CAMERA RECOVERED — stream is back"),
            timestamp=time.time(),
            payload={"candidate": _LinkEvent(
                detector="camera_offline",
                rule_name="camera_offline" if offline else "camera_recovered",
                priority="high" if offline else "info",
                title=(f"Camera {cam_id} unreachable for {held:.0f}s" if offline
                       else f"Camera {cam_id} is back online")),
                     "frames": [], "scene": {}, "enqueued_at": time.time()}))

    pipe = MultiStreamPipeline(sources, weights=weights, target_fps=target_fps, imgsz=imgsz,
                               conf=conf, device=device, half=half, camera_states=states,
                               alert_queue=queue, publisher=publisher,
                               on_link_change=_on_link_change, publish_fps=publish_fps)

    def _fast_path(alert) -> None:
        """Two-tier alerting (EP-06-T4): criticals are shown provisionally the
        moment the detector fires; the verdict updates the same row in place."""
        if alert.priority != "critical":
            return
        candidate = (alert.payload or {}).get("candidate")
        if candidate is not None and getattr(candidate, "detector", "") == "camera_offline":
            return                      # deterministic alerts confirm instantly anyway
        event_id = sink.provisional(alert)
        if event_id is not None and alert.payload is not None:
            alert.payload["provisional_event_id"] = event_id

    pipe.on_queued = _fast_path
    pipe.start()

    # Cameras start immediately in critical-only mode while one persistent,
    # bounded worker maps missing/stale scenes in the background. This keeps a
    # 100-camera commissioning run from becoming 100 blocking Ollama calls.
    from cvti.scene.coordinator import SceneMappingCoordinator
    from cvti.scene.context_store import SceneContextStore

    mapping_coordinator = SceneMappingCoordinator(
        mapping_service, site_config_path, output_dir, max_workers=1,
        policy=site.get("scene_context_policy", "auto"),
    )
    initial_mapping_ids = [
        camera_id for camera_id, status in mapping_preflight.statuses.items()
        if status.get("status") in {"pending", "stale", "failed"}
    ]
    if initial_mapping_ids:
        mapping_coordinator.enqueue(initial_mapping_ids)
    _mapping_stop = threading.Event()

    def _mapping_worker() -> None:
        retry_due: dict[str, float] = {}
        retry_delays = (120.0, 300.0, 600.0)
        while not _mapping_stop.wait(0.25):
            if not mapping_coordinator.run_next():
                now = time.monotonic()
                due = []
                for job in mapping_coordinator.jobs:
                    if job.state != "failed" or job.attempts >= 4:
                        retry_due.pop(job.camera_id, None)
                        continue
                    deadline = retry_due.setdefault(
                        job.camera_id,
                        now + retry_delays[min(job.attempts - 1, len(retry_delays) - 1)],
                    )
                    if now >= deadline:
                        due.append(job.camera_id)
                if due:
                    mapping_coordinator.requeue_failed(due, max_attempts=4)
                    for camera_id in due:
                        retry_due.pop(camera_id, None)
                    continue
                if _mapping_stop.wait(2.0):
                    return
                stale_ids = []
                for camera_id in states:
                    status = SceneContextStore(
                        Path(output_dir) / "context", camera_id
                    ).load_status()
                    if status.status == "stale":
                        stale_ids.append(camera_id)
                if stale_ids:
                    mapping_coordinator.enqueue(stale_ids, force=True)
                continue
            for camera_id, state in states.items():
                store = SceneContextStore(Path(output_dir) / "context", camera_id)
                status = store.load_status()
                context = store._load_context()
                if context is not None:
                    state.scene_context = context
                for index, row in enumerate(mapping_health):
                    if str(row.get("camera_id")) == str(camera_id):
                        mapping_health[index] = {
                            "camera_id": camera_id,
                            **status.to_dict(),
                            "usable": status.status == "ready_reviewed",
                            "review_required": status.status != "ready_reviewed",
                            "environment_type": (
                                context or {}
                            ).get("environment_type", "unknown"),
                        }

    threading.Thread(
        target=_mapping_worker, name="scene-mapping", daemon=True
    ).start()
    custom_scanner = None
    watch_runner = None      # defined here: teardown references it on every path
    if gate_provider in ("ollama", "local"):
        # Watches: plain-English subjects to FOLLOW. Binds a description to a
        # tracked person (via numbered boxes) and keeps a case open for them.
        if any(c.get("watches") for c in cams_cfg):
            from cvti.serving.watch_runner import WatchRunner

            def _latest_frame(cam_id: str):
                # peek, never read: read_latest() CONSUMES the frame, and this
                # helper was quietly stealing frames from detection on every
                # watch check (same family as the scanner's double-decode).
                d = pipe._decoders.get(cam_id)
                if d is None:
                    return None
                f, _seq = d.peek_latest()
                return f.image if f is not None else None

            watch_runner = WatchRunner(
                cams_cfg, states, sink, model=gate_model or "gemma3:4b",
                base_url=gate_base_url or "http://localhost:11434/v1",
                frame_source=_latest_frame).start()

        # Customer-written English rules run as a slow VLM scan — the VLM IS the
        # detector, no person-trigger required (an aeroplane on an empty apron
        # counts). Always started: it watches the site file and begins scanning
        # a camera within one cycle of a sentence being typed in the app.
        from cvti.serving.custom_rules import CustomRuleScanner

        def _scanner_frame(cam_id: str):
            d = pipe._decoders.get(cam_id)
            if d is None:
                return None
            f, _seq = d.peek_latest()
            return f.image if f is not None else None

        custom_scanner = CustomRuleScanner(
            cams_cfg, sink, model=gate_model or "gemma3:4b",
            base_url=gate_base_url or "http://localhost:11434/v1",
            site_config_path=site_config_path,
            frame_source=_scanner_frame,
            context_provider=lambda camera_id: getattr(
                states.get(camera_id), "scene_context", None
            ))
        custom_scanner.status_path = Path(output_dir) / "english_rules_status.json"
        custom_scanner.start()
    # Retention. Storage limitation is not optional, and an edge box with no
    # purge fills its disk and stops recording evidence exactly when it matters.
    from cvti.serving.onboarding import get_site_meta
    from cvti.serving.retention import RetentionManager, RetentionPolicy
    try:
        _policy = RetentionPolicy.from_site(get_site_meta(site_config_path))
    except Exception:  # noqa: BLE001 - a bad setting must not stop monitoring
        log.warning("retention: could not read site policy; using defaults", exc_info=True)
        _policy = RetentionPolicy()
    retention = RetentionManager(output_dir, _policy).start()

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

    _started_at = time.time()
    assurance = None            # constructed below, once the decoders exist
    heartbeat = None            # constructed below, iff the site opted in

    def _gate_reachable(stats: dict):
        return gate_reachable(stats)

    def _build_health() -> dict:
        """The /health document (EP-04-T1): six signal classes, one status.
        Written to gate_health.json for the app, served over HTTP by the
        publisher, and the future heartbeat sends exactly this."""
        from cvti.health import snapshot as health_snapshot
        from cvti.serving.health_doc import build_health_doc
        from cvti.serving.memory_guard import sample_memory
        stats = gate_pool.stats()
        # Link state per camera, so the UI shows coverage rather than inferring
        # it from the absence of alerts.
        cameras = [d.link_status() for d in pipe._decoders.values()]
        # Per-component counters. Without these, "this detector found nothing"
        # and "this detector has thrown on every frame for a week" are the same
        # silence.
        components = health_snapshot()
        ret = retention.status()
        mem = sample_memory()
        gate_doc = {"provider": gate_provider, "model": gate_model,
                    "reachable": _gate_reachable(stats), **stats}
        doc = build_health_doc(
            started_at=_started_at, cameras=cameras, gate=gate_doc,
            disk=ret.get("disk") or {},
            memory={"available_gb": round(mem.available_gb, 2),
                    "rss_gb": round(mem.rss_gb, 2),
                    "level": mem.level(memory_warn_gb, memory_critical_gb)},
            components=components,
            engine={"frames_processed": pipe.frames_processed,
                    "alerts_queued": pipe.alerts_queued,
                    "target_fps": target_fps, "cameras": len(pipe._decoders),
                    "context_suppressions": sum(
                        state.context_suppression_count for state in states.values()
                    )},
            self_test=(assurance.last_result if assurance else {}),
            scene_mapping=mapping_health)
        if model_failures:
            # Configured coverage that is NOT running is at least degraded,
            # and the reason is named — never inferred from silence.
            doc["reasons"] = list(doc.get("reasons") or []) + model_failures
            if doc.get("status") == "ok":
                doc["status"] = "degraded"
        # Legacy keys the System panel already reads — kept at the top level so
        # the app needs no migration.
        doc.update(stats)
        doc.update({"provider": gate_provider, "model": gate_model,
                    "mock": mock_gate, "banner": MOCK_GATE_BANNER if mock_gate else "",
                    "updated_at": time.time(), "health": components,
                    "retention": ret,
                    # Detection -> operator-visible latency per priority tier.
                    # "Criticals alert in under a second" carries its measurement.
                    "alert_latency": sink.latency_stats(),
                    "heartbeat": heartbeat.status() if heartbeat else {"enabled": False}})
        return doc

    def _write_health() -> None:
        st = _build_health()
        try:
            from cvti.serving.health_history import record as _record_health
            _record_health(output_dir, st)
        except Exception:  # noqa: BLE001 - history must never hurt monitoring
            log.debug("health history write failed", exc_info=True)
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
            activated = activate_reviewed_scene_contexts(states, output_dir)
            if activated:
                log.info("[agent-map] full monitoring activated for: %s",
                         ", ".join(activated))
            _write_health()

    _starting_stop.set()          # the real heartbeat owns the file from here
    _write_health()
    _threading.Thread(target=_health_loop, name="gate-health", daemon=True).start()

    if publisher is not None:
        # /health on the publisher's authenticated server (EP-04-T1).
        publisher.health_provider = _build_health

    # Mobile response view (EP-06-T3): the guard's phone, on the site network.
    # Authenticated on every route; Telegram alerts deep-link into it.
    mobile = None
    if mobile_port:
        from cvti.serving.mobile import MobileServer
        # Port taken? Walk forward a few before giving up — and if it still
        # fails, say so in /health instead of only the log: a dead mobile view
        # silently strips the response link from every notification.
        # (Audit 23 Aug, #11.)
        for _try_port in range(mobile_port, mobile_port + 5):
            try:
                mobile = MobileServer(output_dir, port=_try_port,
                                      security_dir=security_dir).start()
                sink.mobile_base = mobile.base_url()
                if _try_port != mobile_port:
                    log.warning("mobile view: port %s taken, serving on %s instead",
                                mobile_port, _try_port)
                break
            except OSError as exc:
                _mobile_err = exc
        if mobile is None:
            log.warning("mobile view could not start on ports %s-%s: %s — alerts "
                        "will not carry a response link", mobile_port,
                        mobile_port + 4, _mobile_err)
            model_failures.append(
                f"mobile response view failed to start (ports {mobile_port}-{mobile_port + 4} busy) "
                "— notifications carry no respond link")

    # Heartbeat (EP-04-T2): OFF unless the site configured a URL. Sends the
    # whitelisted health payload outbound only; docs/HEARTBEAT.md is the schema.
    _site_meta = get_site_meta(site_config_path)
    if _site_meta.get("heartbeat_url"):
        from cvti.serving.heartbeat import Heartbeat
        heartbeat = Heartbeat(
            url=_site_meta["heartbeat_url"], site_key=_site_meta.get("heartbeat_key", ""),
            site_id=(_site_meta.get("name") or "site").strip().lower().replace(" ", "-"),
            health_provider=_build_health, output_dir=output_dir).start()

    # Live re-read of operator-editable site meta (audit 23 Aug, #5/#6/#13).
    # The UI edits notify / retention / heartbeat in the site file and implies
    # immediate effect; the engine used to read them once at spawn, so a test
    # alert would arrive on the NEW channel while real alerts kept going to the
    # old one until a restart nobody knew to do — and the UI claimed a
    # retention period the purger was not enforcing. The engine now watches
    # the file (mtime, 5s) and applies those three live. Cameras/zones/rules
    # still require a restart: they rebuild models, which is a real restart.
    def _rule_fingerprint():
        """mtimes of everything that shapes rules per camera: the site file's
        camera entries plus each camera's rules + zones files."""
        out = {}
        try:
            fresh = load_site_config(site_config_path)
        except Exception:  # noqa: BLE001 - half-written file: keep the old view
            log.debug("site re-read failed", exc_info=True)
            return None, None
        for cam in fresh.get("cameras", []):
            sig = []
            for key in ("config", "zones"):
                path = cam.get(key)
                try:
                    sig.append((key, path, Path(path).stat().st_mtime if path else 0))
                except OSError:
                    sig.append((key, path, 0))
            out[cam.get("id")] = tuple(sig)
        return fresh, out

    def _watch_site_meta():
        state = {"mtime": 0.0, "notify": notify,
                 "hb_url": _site_meta.get("heartbeat_url", ""),
                 "hb_key": _site_meta.get("heartbeat_key", "")}
        try:
            state["mtime"] = Path(site_config_path).stat().st_mtime
        except OSError:
            pass
        _, state["rules_fp"] = _rule_fingerprint()
        while True:
            time.sleep(5)
            # English rules and zones are JSON, not models: hot-swap them into
            # the running per-camera states so "describe it in English" takes
            # effect in seconds, not at the next restart. (User feedback,
            # 23 Aug: "it should kick off automatically".)
            fresh_site, fp = _rule_fingerprint()
            if fp is not None and fp != state.get("rules_fp"):
                from cvti.serving.camera import refresh_camera_rules
                for cam in fresh_site.get("cameras", []):
                    cid = cam.get("id")
                    if cid in states and fp.get(cid) != (state.get("rules_fp") or {}).get(cid):
                        try:
                            refresh_camera_rules(states[cid], cam, baseline_config)
                            log.info("[site] rules/zones hot-reloaded for camera %s", cid)
                        except Exception:  # noqa: BLE001 - keep the old rules over none
                            log.error("[site] rules hot-reload failed for %s; keeping "
                                      "current rules", cid, exc_info=True)
                state["rules_fp"] = fp
            try:
                mtime = Path(site_config_path).stat().st_mtime
                if mtime == state["mtime"]:
                    continue
                state["mtime"] = mtime
                meta = get_site_meta(site_config_path)
            except Exception:  # noqa: BLE001 - a half-written file is retried next tick
                log.debug("site meta re-read failed; retrying", exc_info=True)
                continue
            new_notify = (meta.get("notify") or "console").strip()
            if new_notify != state["notify"]:
                try:
                    sink.notifier = build_notifier(new_notify)
                    state["notify"] = new_notify
                    log.info("[site] notifier changed live -> %s", new_notify)
                except Exception:  # noqa: BLE001 - keep the old channel over none
                    log.error("[site] new notifier %r failed to build; keeping the old one",
                              new_notify, exc_info=True)
            try:
                new_policy = RetentionPolicy.from_site(meta)
                if new_policy != retention.policy:
                    retention.policy = new_policy
                    log.info("[site] retention policy changed live -> %s days",
                             new_policy.days)
            except Exception:  # noqa: BLE001 - a bad setting must not stop the watcher
                log.warning("[site] new retention setting unreadable; keeping current",
                            exc_info=True)
            hb_url = (meta.get("heartbeat_url") or "").strip()
            hb_key = (meta.get("heartbeat_key") or "").strip()
            if (hb_url, hb_key) != (state["hb_url"], state["hb_key"]):
                nonlocal heartbeat
                if heartbeat:
                    heartbeat.stop()
                    heartbeat = None
                if hb_url:
                    from cvti.serving.heartbeat import Heartbeat
                    heartbeat = Heartbeat(
                        url=hb_url, site_key=hb_key,
                        site_id=(meta.get("name") or "site").strip().lower().replace(" ", "-"),
                        health_provider=_build_health, output_dir=output_dir).start()
                    log.info("[site] heartbeat %s live", "reconfigured" if state["hb_url"] else "enabled")
                else:
                    log.info("[site] heartbeat disabled live")
                state["hb_url"], state["hb_key"] = hb_url, hb_key

    threading.Thread(target=_watch_site_meta, name="site-meta-watch", daemon=True).start()

    # Weekly owner summary (EP-08-T2): automatic, no action required. Checked
    # hourly; fires Monday 08:00+ at most once per ISO week, survives restarts
    # via summaries/state.json, and delivers through the same notifier alerts
    # use — the person deciding renewal hears from the product weekly.
    def _weekly_summary_loop():
        from cvti.owner_summary import due, mark_sent, weekly_summary
        while True:
            time.sleep(3600)
            try:
                if not due(output_dir):
                    continue
                meta = get_site_meta(site_config_path)
                s = weekly_summary(Path(output_dir) / "events.db", meta, output_dir)
                w = s["week"]
                sink.notifier.notify({
                    "ts": time.time(), "iso": s["generated_at"],
                    "camera_id": "weekly_summary", "rule": "weekly_summary",
                    "priority": "low", "confidence": None, "zone": None,
                    "track_id": None, "object_label": None, "evidence_dir": None,
                    "reason": (f"Weekly summary {s['window']['from']}->{s['window']['to']}: "
                               f"{w['incidents']} incidents ({w['real']} real), "
                               f"{w['noise_removed']} false alarms filtered, "
                               f"~{s['hours_saved']}h of attention given back. "
                               f"Full report: {s['pdf']}")})
                mark_sent(output_dir)
            except Exception:  # noqa: BLE001 - the summary must never hurt monitoring
                log.warning("weekly summary failed; will retry next hour", exc_info=True)

    threading.Thread(target=_weekly_summary_loop, name="weekly-summary", daemon=True).start()

    # Daily proof of life (EP-04-T4): the self-test exercises a real frame ->
    # the real gate -> a real notification, and the all-normal message makes
    # silence stop being the success signal. Skipped entirely for a mock gate —
    # a self-test against a gate that confirms everything proves nothing.
    if not mock_gate:
        from cvti.serving.assurance import Assurance

        def _any_latest_frame():
            for d in pipe._decoders.values():
                f, _seq = d.peek_latest()      # peek: detection owns read_latest
                if f is not None:
                    return f.image
            return None

        assurance = Assurance(
            latest_frame=_any_latest_frame,
            gate_factory=lambda: VerificationGate(provider=gate_provider, model=gate_model,
                                                  base_url=gate_base_url,
                                                  sensitivity=gate_sensitivity),
            notifier=sink.notifier,
            status_provider=_build_health,
            daily_normal=bool(get_site_meta(site_config_path).get("daily_normal", True)),
        ).start()


    try:
        pipe.run(max_seconds=seconds)
    finally:
        _mapping_stop.set()
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
        retention.stop()
        if assurance is not None:
            assurance.stop()
        if heartbeat is not None:
            heartbeat.beat()      # a final send so "last seen" reflects shutdown time
            heartbeat.stop()
        if mobile is not None:
            mobile.stop()
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
    p.add_argument("--publish-fps", type=float, default=12.0,
                   help="live-wall frame rate, decoupled from detection; 0 = old coupled behavior")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--device", default="")
    p.add_argument("--half", action="store_true")
    p.add_argument("--seconds", type=float, default=90.0,
                   help="Run cap; file sources stop automatically at end-of-stream.")
    p.add_argument("--gate-provider", default="mock")
    p.add_argument("--gate-model", default="")
    p.add_argument("--gate-base-url", default="")
    p.add_argument("--mapper-provider", default="",
                   help="Scene mapper provider; empty inherits the local gate provider.")
    p.add_argument("--mapper-model", default="",
                   help="Scene mapper model; empty inherits the gate model when compatible.")
    p.add_argument("--mapper-base-url", default="",
                   help="Scene mapper API base URL; empty inherits the local gate URL.")
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
    p.add_argument("--security-dir", default="",
                   help="where auth.db/audit.db live — GLOBAL to the install, not "
                        "per-feed (defaults to the output dir for standalone runs)")
    p.add_argument("--mobile-port", type=int, default=8710,
                   help="Port for the phone response view on the site network; 0 disables.")
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
                 publish_fps=args.publish_fps,
                 memory_guard=not args.no_memory_guard,
                 memory_warn_gb=args.memory_warn_gb,
                 memory_critical_gb=args.memory_critical_gb,
                 mobile_port=args.mobile_port,
                 security_dir=args.security_dir or None,
                 gate_model=args.gate_model, gate_base_url=args.gate_base_url,
                 mapper_provider=args.mapper_provider,
                 mapper_model=args.mapper_model,
                 mapper_base_url=args.mapper_base_url,
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
