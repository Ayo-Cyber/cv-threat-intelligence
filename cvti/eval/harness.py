"""The evaluation runner.

Drives the REAL detection path (same PerCameraState the live engine uses) over a
labeled clip set, one clip at a time, and records both stages:

    Stage 1  every candidate alert the detectors raised
    Stage 2  the subset TrueSight confirmed

Deliberately sequential and headless — no decode threads, no live wall, no app —
so it is deterministic, restartable, and far lighter on RAM than a live run.
Results stream to disk per clip, so Ctrl+C never loses completed work.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cvti.eval.dataset import EvalClip


@dataclass
class ClipResult:
    name: str
    path: str
    is_threat: bool
    candidates: int = 0          # Stage 1
    confirmed: int = 0           # Stage 2
    rules_fired: tuple = ()
    seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "is_threat": self.is_threat,
                "candidates": self.candidates, "confirmed": self.confirmed,
                "rules_fired": list(self.rules_fired), "seconds": round(self.seconds, 1),
                "error": self.error}


class EvalHarness:
    def __init__(self, *, config: str = "configs/all_threats_video_v1.json",
                 baseline: str | None = "configs/baseline_critical_v1.json",
                 weights: str = "models/yolov8n.pt",
                 pose_weights: str = "models/yolov8n-pose.pt",
                 video_model: str = "runs/video_finetune/videomae",
                 detectors: tuple = ("concealment", "video_action"),
                 gate: Any = None, target_fps: float = 4.0,
                 max_seconds_per_clip: float = 30.0,
                 out_dir: str = "runs/eval") -> None:
        self.config = config
        self.baseline = baseline
        self.weights = weights
        self.pose_weights = pose_weights
        self.video_model = video_model
        self.detectors = detectors
        self.gate = gate                      # None = Stage 1 only
        self.target_fps = target_fps
        self.max_seconds_per_clip = max_seconds_per_clip
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._names = None
        self._threat_classes = None
        self._pose = None
        self._video = None

    # --- model loading (once, shared across every clip) ---
    def _load_models(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        from cvti.detector.core import normalize_threat_classes
        self._model = YOLO(self.weights)
        self._names = self._model.names
        self._threat_classes = normalize_threat_classes("gun,knife")
        if any(d in self.detectors for d in ("concealment", "violence", "theft")):
            from cvti.detector.core import load_ultralytics_model
            self._pose = load_ultralytics_model(self.pose_weights)
        if "video_action" in self.detectors and Path(self.video_model).exists():
            try:
                from cvti.video_action_model import VideoMAEActionModel
                self._video = VideoMAEActionModel(self.video_model)
            except Exception as exc:  # noqa: BLE001
                print(f"[eval] video-action model unavailable ({str(exc)[:70]}) — skipping")

    def _state_for(self, clip: EvalClip):
        from cvti.rules.customization import CustomizationEngine
        from cvti.serving.camera import PerCameraState
        engine = CustomizationEngine(self.config, baseline_path=self.baseline)
        kwargs = {d: True for d in self.detectors}
        return PerCameraState(clip.name, engine, pose_model=self._pose,
                              video_action_model=self._video,
                              scene_context={"environment_type": "retail",
                                             "scene_description": "A shop interior monitored for theft."},
                              **kwargs)

    # --- one clip ---
    def run_clip(self, clip: EvalClip) -> ClipResult:
        import cv2
        import supervision as sv

        from cvti.detector.core import extract_detections
        self._load_models()
        res = ClipResult(clip.name, clip.path, clip.is_threat)
        t0 = time.time()
        cap = cv2.VideoCapture(clip.path)
        if not cap.isOpened():
            res.error = "could not open clip"
            return res
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(src_fps / max(self.target_fps, 0.1))))
        state = self._state_for(clip)
        rules: list[str] = []
        idx = 0
        try:
            while True:
                ok = cap.grab()
                if not ok:
                    break
                idx += 1
                if idx % stride:
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                ts = idx / src_fps
                if ts > self.max_seconds_per_clip:
                    break
                result = self._model(frame, verbose=False)[0]
                dets = sv.Detections.from_ultralytics(result)
                objs = extract_detections(result, self._names, self._threat_classes)
                for alert in state.process(dets, frame, ts, object_detections=objs):
                    res.candidates += 1
                    rules.append(alert.rule_name)
                    if self.gate is not None and self._confirm(alert):
                        res.confirmed += 1
        except Exception as exc:  # noqa: BLE001 - one bad clip must not sink the run
            res.error = str(exc)[:200]
        finally:
            cap.release()
        res.rules_fired = tuple(sorted(set(rules)))
        res.seconds = time.time() - t0
        return res

    def _confirm(self, alert: Any) -> bool:
        payload = alert.payload or {}
        try:
            verdict = self.gate.verify(payload.get("frames"), payload.get("candidate"),
                                       payload.get("scene"))
            return bool(verdict is not None and verdict.confirmed)
        except Exception as exc:  # noqa: BLE001 - a gate error is not a confirmation
            print(f"[eval] gate error on {alert.rule_name}: {str(exc)[:90]}")
            return False

    # --- the whole set, resumable ---
    def run(self, clips: list[EvalClip], *, resume: bool = True,
            progress: bool = True) -> list[ClipResult]:
        results_path = self.out_dir / "clip_results.jsonl"
        done: dict[str, dict] = {}
        if resume and results_path.exists():
            for line in results_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    done[d["path"]] = d
                except Exception:  # noqa: BLE001
                    continue
            if done and progress:
                print(f"[eval] resuming — {len(done)} clip(s) already done")

        out: list[ClipResult] = []
        with results_path.open("a") as fh:
            for i, clip in enumerate(clips, 1):
                if clip.path in done:
                    d = done[clip.path]
                    out.append(ClipResult(d["name"], d["path"], d["is_threat"],
                                          d["candidates"], d["confirmed"],
                                          tuple(d.get("rules_fired", [])),
                                          d.get("seconds", 0.0), d.get("error", "")))
                    continue
                if progress:
                    print(f"[eval] {i}/{len(clips)}  {clip.name} "
                          f"({'threat' if clip.is_threat else 'normal'}) …", flush=True)
                r = self.run_clip(clip)
                fh.write(json.dumps(r.to_dict()) + "\n")
                fh.flush()
                out.append(r)
                if progress:
                    print(f"        candidates={r.candidates} confirmed={r.confirmed} "
                          f"{r.seconds:.0f}s {r.error}", flush=True)
        return out
