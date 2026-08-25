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

from cvti.eval.dataset import ROOT, EvalClip
from cvti.logging_setup import get_logger

log = get_logger(__name__)


class GateUnavailable(RuntimeError):
    """The verification gate is unreachable — abort rather than report fake numbers."""


@dataclass
class ClipResult:
    name: str
    path: str
    is_threat: bool
    candidates: int = 0          # Stage 1: everything the detectors proposed
    verified: int = 0            # Stage 2: how many actually reached the VLM
    confirmed: int = 0           # Stage 2: how many it confirmed
    capped: bool = False         # a per-clip cap skipped some candidates
    rules_fired: tuple = ()
    seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "is_threat": self.is_threat,
                "candidates": self.candidates, "verified": self.verified,
                "confirmed": self.confirmed, "capped": self.capped,
                "rules_fired": list(self.rules_fired), "seconds": round(self.seconds, 1),
                "error": self.error}


class EvalHarness:
    def __init__(self, *, config: str = "configs/all_threats_video_v1.json",
                 baseline: str | None = "configs/baseline_critical_v1.json",
                 weights: str = "models/yolov8n.pt",
                 pose_weights: str = "models/yolov8n-pose.pt",
                 weapon_weights: str = "models/weapon_best.pt",
                 video_model: str = "runs/video_finetune/videomae",
                 detectors: tuple = ("concealment", "video_action"),
                 gate: Any = None, target_fps: float = 4.0,
                 imgsz: int = 640, conf: float = 0.25,
                 max_seconds_per_clip: float = 30.0,
                 stop_on_first_confirm: bool = True,
                 max_candidates_per_clip: int = 0,
                 out_dir: str = "runs/eval", run_key: str = "default") -> None:
        self.config = config
        self.baseline = baseline
        self.weights = weights
        self.pose_weights = pose_weights
        self.weapon_weights = weapon_weights
        self.video_model = video_model
        self.detectors = detectors
        self.gate = gate                      # None = Stage 1 only
        # Dense crowds need more pixels: at 640 the person detector resolves only
        # 0-2 individuals in a packed scene, starving any count-based detector.
        self.imgsz = imgsz
        self.conf = conf
        self.target_fps = target_fps
        self.max_seconds_per_clip = max_seconds_per_clip
        # Lossless: a clip is flagged if ANY candidate confirms.
        self.stop_on_first_confirm = stop_on_first_confirm
        # LOSSY and reported: caps VLM calls per clip. Biases recall DOWN (a
        # confirmation might have come from a skipped candidate), never up —
        # so a capped run understates the detector, which is the safe
        # direction for a claim.
        self.max_candidates_per_clip = max_candidates_per_clip
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Checkpoints are per RUN KEY (gate + detectors). Resuming a real ollama
        # run from mock results would silently report fake numbers.
        self.run_key = run_key
        self.gate_errors = 0            # consecutive gate failures
        self.max_gate_errors = 5        # abort past this — see _confirm()
        # Lazily-loaded shared models. Must be initialised HERE: preflight() returns
        # early when there is no gate, so leaving them there broke --gate none.
        self._model = None
        self._names = None
        self._threat_classes = None
        self._pose = None
        self._weapon = None
        self._video = None

    def preflight(self) -> None:
        """Fail fast if the gate can't answer, instead of producing a whole run of
        'rejections' that are really connection errors."""
        if self.gate is None:
            return
        import numpy as np

        from cvti.contracts import CandidateAlert
        probe = CandidateAlert(
            rule_name="preflight", detector="video_action", title="preflight probe",
            person_id=None, object_label=None, priority="low", timestamp=0.0)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self.gate.verify([frame], probe, {"environment_type": "test",
                                              "scene_description": "preflight"})
        except Exception as exc:  # noqa: BLE001
            raise GateUnavailable(
                f"verification gate unreachable: {str(exc)[:160]}\n"
                "  If you meant to use the local VLM, start it first:  ollama serve\n"
                "  (and make sure the model is pulled:  ollama pull gemma3:4b)"
            ) from exc

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
        if "weapons" in self.detectors:
            # Without this the weapons flag was set but no model was passed —
            # the detector never fired and an eval would have scored 0% recall
            # against a detector that was never actually running. Same failure
            # class as "detectors nothing listens to are silently discarded".
            from cvti.detector.core import load_detection_model
            self._weapon = load_detection_model(
                self.weapon_weights, str(ROOT / "external" / "yolov5"),
                preferred_kind="yolov5")
        if "video_action" in self.detectors and Path(self.video_model).exists():
            try:
                from cvti.video_action_model import VideoMAEActionModel
                self._video = VideoMAEActionModel(self.video_model)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[eval] video-action model unavailable ({str(exc)[:70]}) — skipping", exc_info=True)

    def _state_for(self, clip: EvalClip):
        from cvti.rules.customization import CustomizationEngine
        from cvti.serving.camera import PerCameraState
        engine = CustomizationEngine(self.config, baseline_path=self.baseline)
        kwargs = {d: True for d in self.detectors}
        return PerCameraState(clip.name, engine, pose_model=self._pose,
                              weapon_model=self._weapon,
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
                result = self._model(frame, imgsz=self.imgsz, conf=self.conf,
                                     verbose=False)[0]
                dets = sv.Detections.from_ultralytics(result)
                objs = extract_detections(result, self._names, self._threat_classes)
                for alert in state.process(dets, frame, ts, object_detections=objs):
                    res.candidates += 1
                    rules.append(alert.rule_name)
                    # Clip-level scoring asks ONE question: did anything on this
                    # clip confirm? Once something has, every further VLM call
                    # is ~12s spent to learn nothing. Lossless early exit — the
                    # 9.5h/detector estimate was mostly this. (25 Aug.)
                    if res.confirmed and self.stop_on_first_confirm:
                        continue
                    if (self.max_candidates_per_clip
                            and res.verified >= self.max_candidates_per_clip):
                        res.capped = True
                        continue
                    if self.gate is not None:
                        res.verified += 1
                        if self._confirm(alert):
                            res.confirmed += 1
        except Exception as exc:  # noqa: BLE001 - one bad clip must not sink the run
            log.warning("clip evaluation failed; recorded on the result", exc_info=True)
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
            self.gate_errors = 0        # a success clears the streak
            return bool(verdict is not None and verdict.confirmed)
        except Exception as exc:  # noqa: BLE001
            # A gate error is NOT a rejection. Counting it as one would report
            # "TrueSight suppressed everything" — fake numbers that look real.
            self.gate_errors += 1
            log.error(f"[eval] gate error on {alert.rule_name}: {str(exc)[:90]}", exc_info=True)
            if self.gate_errors >= self.max_gate_errors:
                raise GateUnavailable(
                    f"{self.gate_errors} consecutive gate failures — aborting so the run "
                    f"cannot report bogus suppression numbers. Last error: {str(exc)[:120]}"
                ) from exc
            return False

    # --- the whole set, resumable ---
    def run(self, clips: list[EvalClip], *, resume: bool = True,
            progress: bool = True) -> list[ClipResult]:
        self.preflight()      # cheap: one probe verdict before committing to the set
        safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in self.run_key)
        results_path = self.out_dir / f"clip_results_{safe}.jsonl"
        done: dict[str, dict] = {}
        if resume and results_path.exists():
            for line in results_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    done[d["path"]] = d
                except Exception:  # noqa: BLE001
                    continue
            if done and progress:
                log.info(f"[eval] resuming — {len(done)} clip(s) already done")

        out: list[ClipResult] = []
        with results_path.open("a") as fh:
            for i, clip in enumerate(clips, 1):
                if clip.path in done:
                    d = done[clip.path]
                    # By keyword, and tolerant of older checkpoint rows: a new
                    # field used to shift every positional after it and silently
                    # load garbage on resume.
                    out.append(ClipResult(
                        name=d["name"], path=d["path"], is_threat=d["is_threat"],
                        candidates=d.get("candidates", 0),
                        verified=d.get("verified", d.get("candidates", 0)),
                        confirmed=d.get("confirmed", 0),
                        capped=bool(d.get("capped", False)),
                        rules_fired=tuple(d.get("rules_fired", [])),
                        seconds=d.get("seconds", 0.0), error=d.get("error", "")))
                    continue
                if progress:
                    log.info(f"[eval] {i}/{len(clips)}  {clip.name} "
                          f"({'threat' if clip.is_threat else 'normal'}) …")
                r = self.run_clip(clip)
                fh.write(json.dumps(r.to_dict()) + "\n")
                fh.flush()
                out.append(r)
                if progress:
                    log.error(f"        candidates={r.candidates} confirmed={r.confirmed} "
                          f"{r.seconds:.0f}s {r.error}")
        return out
