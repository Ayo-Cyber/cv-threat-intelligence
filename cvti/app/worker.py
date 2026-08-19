"""Inference worker — runs the full detection pipeline in a background QThread."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from cvti.utils import resource_path


class DetectionWorker(QThread):
    frame_ready   = pyqtSignal(np.ndarray)   # annotated BGR frame
    alert_fired   = pyqtSignal(dict)          # alert payload
    status_update = pyqtSignal(str)           # status bar text
    fps_update    = pyqtSignal(float)         # current FPS

    def __init__(
        self,
        source: str,
        config_path: str,
        zones_path: str = "",
        gate_provider: str = "mock",
        gate_model: str = "",
        scene_context: dict | None = None,
        pose_weights: str = "",
        object_weights: str = "",
        conf: float = 0.4,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.source        = source
        self.config_path   = config_path
        self.zones_path    = zones_path
        self.gate_provider = gate_provider
        self.gate_model    = gate_model
        self.scene_context = scene_context
        self.pose_weights   = pose_weights or str(resource_path("models/yolov8n-pose.pt"))
        self.object_weights = object_weights or str(resource_path("models/yolov8n.pt"))
        self.conf          = conf
        self._running      = False

    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        self.status_update.emit("Loading models…")

        try:
            from ultralytics import YOLO
            import supervision as sv
            from cvti.retail.concealment import ConcealmentDetector, PoseFrame, keypoints_from_ultralytics, COCO_BAG_IDS
            from cvti.retail.zones import RetailZoneMonitor, filter_person_detections, load_zone_config
            from cvti.rules.customization import CustomizationEngine
            from cvti.event_adapters import zone_states_to_events, concealment_to_events
            from cvti.verification.gate import (
                MOCK_GATE_BANNER, VerificationGate, assert_engine_gate_allowed)

            # Same guard as the serving engine: a mock gate confirms everything.
            self.mock_gate = assert_engine_gate_allowed(self.gate_provider)

            pose_model   = YOLO(self.pose_weights)
            object_model = YOLO(self.object_weights)
            zone_monitor = RetailZoneMonitor(load_zone_config(self.zones_path)) if self.zones_path else None
            concealment  = ConcealmentDetector()
            engine       = CustomizationEngine(self.config_path)
            gate         = VerificationGate(provider=self.gate_provider, model=self.gate_model)

        except Exception as exc:
            self.status_update.emit(f"Load error: {exc}")
            return

        if getattr(self, "mock_gate", False):
            self.status_update.emit(f"⚠ {MOCK_GATE_BANNER} — alerts are NOT being verified.")
        else:
            self.status_update.emit("Running…")
        source = int(self.source) if self.source.isdigit() else self.source

        last_rule_sig: str | None = None
        last_alert_t  = -1e9
        cooldown      = 5.0
        n             = 0
        t0            = time.time()

        for result in pose_model.track(
            source=source, stream=True, persist=True,
            tracker=str(resource_path("configs/bytetrack_retail.yaml")),
            conf=self.conf, classes=[0], verbose=False,
        ):
            if not self._running:
                break

            frame = result.orig_img
            ts    = n / max(result.speed.get("preprocess", 1), 1e-3) if n > 0 else 0.0
            n    += 1

            detections  = sv.Detections.from_ultralytics(result)
            detections  = filter_person_detections(detections, frame.shape[:2])
            zone_states = zone_monitor.update(detections, ts) if zone_monitor else []

            pose_frames: list[PoseFrame] = []
            ids = result.boxes.id if result.boxes is not None else None
            if ids is not None and result.keypoints is not None:
                for i in range(len(ids)):
                    pose_frames.append(PoseFrame(
                        track_id=int(ids[i]),
                        timestamp=ts,
                        keypoints=keypoints_from_ultralytics(result, i),
                        bbox=tuple(float(v) for v in result.boxes.xyxy[i]),
                    ))

            bag_bboxes = None
            obj = object_model(frame, classes=list(COCO_BAG_IDS), conf=0.35, verbose=False)[0]
            if obj.boxes is not None and len(obj.boxes) > 0:
                bag_bboxes = [tuple(float(v) for v in b) for b in obj.boxes.xyxy]

            conceal    = concealment.update(pose_frames, ts, bag_bboxes=bag_bboxes)
            raw_events = zone_states_to_events(zone_states, ts) + concealment_to_events(conceal, ts)
            alerts     = engine.evaluate(raw_events, scene_context=self.scene_context)
            top        = alerts[0] if alerts else None

            if top is not None:
                sig = f"{top.rule_name}:{top.person_id}"
                if sig != last_rule_sig:
                    last_rule_sig = sig
                    verdict = gate.verify(frame, top, self.scene_context)
                    if verdict.confirmed and (ts - last_alert_t) >= cooldown:
                        last_alert_t = ts
                        self.alert_fired.emit({
                            "rule":       top.rule_name,
                            "priority":   top.priority,
                            "person_id":  top.person_id,
                            "confidence": round(verdict.confidence, 2),
                            "reason":     verdict.reason,
                            "timestamp":  time.strftime("%H:%M:%S"),
                        })
            else:
                last_rule_sig = None

            # Annotate frame
            annotated = zone_monitor.annotate(frame.copy(), detections, zone_states) if zone_monitor else frame.copy()
            annotated = _draw_concealment(annotated, pose_frames, {a.track_id: a for a in conceal})

            self.frame_ready.emit(annotated)

            elapsed = time.time() - t0
            if elapsed > 0:
                self.fps_update.emit(n / elapsed)

        self.status_update.emit("Stopped.")

    def stop(self) -> None:
        self._running = False
        self.wait()


# ---------------------------------------------------------------------------

def _draw_concealment(frame: np.ndarray, pose_frames: list, conceal_by_id: dict) -> np.ndarray:
    for pf in pose_frames:
        a = conceal_by_id.get(pf.track_id)
        if a is None or pf.bbox is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in pf.bbox)
        color = (0, 0, 255) if a.candidate else (0, 170, 255) if a.score >= 0.4 else (0, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"#{pf.track_id} CONCEAL>{(a.destination or '?').upper()}" if a.candidate else f"#{pf.track_id} {a.score:.2f}"
        cv2.putText(frame, tag, (x1 + 3, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame
