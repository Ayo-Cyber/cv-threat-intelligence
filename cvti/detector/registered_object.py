"""Registered-region appearance monitoring; a change is not proof of removal."""
from __future__ import annotations
from dataclasses import dataclass
import math
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RegionAssessment:
    state: str
    changed: bool = False
    appearance_score: float | None = None
    edge_retention: float | None = None
    evidence_png: bytes | None = None
    reason: str | None = None


class RegisteredObjectMonitor:
    """One explicitly registered rectangle, with frozen reference and bounded state.

    Callers supply successful shared person detections (None means unavailable).
    Revalidation requires explicit registration after a gap or camera change.
    """

    def __init__(self, confirm_seconds=8.0, max_gap_seconds=2.5):
        for value in (confirm_seconds, max_gap_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("timers must be finite and positive")
        self.confirm_seconds = confirm_seconds
        self.max_gap_seconds = max_gap_seconds
        self._reference = None
        self._last_time = None
        self._suspect_since = None
        self._emitted = False
        self._valid = False
        self._scene_change_since = None
        self._invalid_reason = "reference_unavailable"

    @staticmethod
    def _coverage(box, region):
        x1, y1, x2, y2 = region
        overlap = max(0, min(x2, box[2]) - max(x1, box[0])) * max(
            0, min(y2, box[3]) - max(y1, box[1]))
        return overlap / ((x2 - x1) * (y2 - y1))

    def register(self, frames, region, person_boxes):
        """Require 3-5 same-sized, unobstructed frames; never auto-learn absence."""
        self._valid = False
        if not 3 <= len(frames) <= 5 or len(person_boxes) != len(frames):
            raise ValueError("registration needs 3-5 frames and their person detections")
        shape = frames[0].shape
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError("registration requires BGR frames")
        if len(region) != 4 or any(not math.isfinite(v) or int(v) != v for v in region):
            raise ValueError("region must contain four integer coordinates")
        x1, y1, x2, y2 = map(int, region)
        if not (0 <= x1 < x2 <= shape[1] and 0 <= y1 < y2 <= shape[0]):
            raise ValueError("region must be inside the frame")
        crops, backgrounds = [], []
        for frame, boxes in zip(frames, person_boxes):
            if frame.shape != shape or boxes is None:
                raise ValueError("registration requires consistent frames and available detections")
            if any(self._coverage(box, region) >= .1 for box in boxes):
                raise ValueError("registered region is occluded")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crops.append(gray[y1:y2, x1:x2].astype(np.float32))
            backgrounds.append(cv2.resize(gray, (64, 48)).astype(np.float32))
        reference = np.median(crops, axis=0).astype(np.uint8)
        edges = cv2.Canny(reference, 50, 120)
        if np.count_nonzero(edges) < 10:
            raise ValueError("region lacks sufficient edge detail")
        if any(np.mean(np.abs(crop - reference)) > 10 for crop in crops):
            raise ValueError("registration frames are not stable")
        self._shape, self._region = shape, (x1, y1, x2, y2)
        self._reference, self._edges = reference.copy(), edges > 0
        self._reference_frame = np.median(np.stack(frames), axis=0).astype(np.uint8)
        self._background = np.median(backgrounds, axis=0)
        self._outside = np.ones((48, 64), dtype=bool)
        self._outside[int(y1 * 48 / shape[0]):math.ceil(y2 * 48 / shape[0]),
                      int(x1 * 64 / shape[1]):math.ceil(x2 * 64 / shape[1])] = False
        if self._outside.sum() < 100:
            raise ValueError("region leaves insufficient background for camera-change checks")
        if any(np.mean(np.abs(background - self._background)[self._outside]) > 10
               for background in backgrounds):
            raise ValueError("registration background is not stable")
        self._valid = True
        self._last_time = self._suspect_since = None
        self._occluded_since = None
        self._emitted = False
        self._scene_change_since = None

    def save_reference(self, path, *, camera_id, source_fingerprint, name):
        """Persist non-executable arrays; the caller owns access control and paths."""
        if not self._valid:
            raise ValueError("a valid registration is required")
        if not all(isinstance(v, str) and v.strip()
                   for v in (camera_id, source_fingerprint, name)):
            raise ValueError("camera, source fingerprint and object name are required")
        metadata = json.dumps({"version": 1, "camera_id": camera_id,
                               "source_fingerprint": source_fingerprint, "name": name,
                               "region": self._region})
        # Write to a sibling temporary file then replace, never expose partial data.
        import os
        import tempfile
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".reference-")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, metadata=metadata, frame=self._reference_frame)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_reference(self, path, *, camera_id, source_fingerprint, approved=False):
        """Loading never resumes timers or bypasses explicit operator approval."""
        self._valid = False
        if approved is not True:
            raise ValueError("saved reference requires explicit revalidation approval")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            if (metadata.get("version") != 1 or metadata.get("camera_id") != camera_id
                    or metadata.get("source_fingerprint") != source_fingerprint):
                raise ValueError("reference does not match this camera source")
            frame = saved["frame"]
            if frame.dtype != np.uint8:
                raise ValueError("invalid reference image")
            self.register([frame] * 3, metadata["region"], [[], [], []])
        return metadata["name"]

    def _evidence(self, frame):
        """Label the registered reference and current view, preserving aspect ratio."""
        h, w = frame.shape[:2]
        scale = min(1.0, 480 / w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        panes = []
        for title, image in (("REGISTERED REFERENCE", self._reference_frame),
                             ("CURRENT - CHANGE REQUIRES REVIEW", frame)):
            pane = cv2.resize(image, size)
            x1, y1, x2, y2 = [round(v * scale) for v in self._region]
            cv2.rectangle(pane, (x1, y1), (x2, y2), (0, 180, 255), 2)
            pane = cv2.copyMakeBorder(pane, 48, 0, 0, 0, cv2.BORDER_CONSTANT)
            # Keep labels readable even for small source images.
            font_scale = min(.45, size[0] / max(1, len(title) * 12))
            cv2.putText(pane, title, (3, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), 1)
            panes.append(pane)
        ok, encoded = cv2.imencode(".png", np.concatenate(panes, axis=1))
        if not ok:
            raise RuntimeError("could not encode registered-object evidence")
        return encoded.tobytes()

    def _invalidate(self, reason):
        self._valid = False
        self._invalid_reason = reason
        return RegionAssessment("revalidation_required", reason=reason)

    def update(self, frame, timestamp, person_boxes):
        if not self._valid or self._reference is None:
            return RegionAssessment("revalidation_required", reason=self._invalid_reason)
        if not math.isfinite(timestamp):
            return self._invalidate("invalid_timestamp")
        if frame.shape != self._shape:
            return self._invalidate("frame_dimensions_changed")
        if person_boxes is None:
            return self._invalidate("person_detection_unavailable")
        if self._last_time is not None and not 0 < timestamp - self._last_time <= self.max_gap_seconds:
            return self._invalidate("frame_timing_discontinuity")
        self._last_time = timestamp
        if any(self._coverage(box, self._region) >= .1 for box in person_boxes):
            # Require fresh consecutive visible evidence after handling.
            self._suspect_since = None
            self._scene_change_since = None
            if self._occluded_since is None:
                self._occluded_since = timestamp
            if timestamp - self._occluded_since > 30:
                return self._invalidate("prolonged_occlusion")
            return RegionAssessment("occluded")
        self._occluded_since = None
        # Assess the background only once the registered region is visible.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 48)).astype(np.float32)
        delta = small - self._background
        # A bounded global exposure shift is not evidence of camera movement.
        # Estimate outside the object so disappearance cannot set the correction.
        exposure_shift = float(np.median(delta[self._outside]))
        change = np.abs(delta - exposure_shift) > 35
        if gray.std() < 3:
            return self._invalidate("insufficient_image_detail")
        if abs(exposure_shift) > 30 or np.mean(change[self._outside]) > .25:
            self._suspect_since = None
            if self._scene_change_since is None:
                self._scene_change_since = timestamp
            if timestamp - self._scene_change_since >= 3:
                return self._invalidate("scene_or_lighting_changed")
            return RegionAssessment("waiting_for_stable_view", reason="scene_change_pending")
        self._scene_change_since = None
        x1, y1, x2, y2 = self._region
        crop = gray[y1:y2, x1:x2]
        reference = self._reference.astype(np.float32)
        current = crop.astype(np.float32)
        current += np.median(reference) - np.median(current)
        appearance = max(0.0, 1.0 - float(np.mean(np.abs(current - reference))) / 64)
        current_edges = cv2.dilate(cv2.Canny(crop, 50, 120), np.ones((3, 3), np.uint8))
        retention = float(np.mean(current_edges[self._edges] > 0))
        if appearance < .5 and retention < .4:
            if self._suspect_since is None:
                self._suspect_since = timestamp
            if timestamp - self._suspect_since >= self.confirm_seconds:
                first = not self._emitted
                evidence = self._evidence(frame) if first else None
                self._emitted = True
                return RegionAssessment("change_detected", first, appearance, retention, evidence)
            return RegionAssessment("suspect", False, appearance, retention)
        self._suspect_since = None
        # Do not re-arm automatically: an operator must review/re-register.
        return RegionAssessment("review_required" if self._emitted else "present",
                                False, appearance, retention)
