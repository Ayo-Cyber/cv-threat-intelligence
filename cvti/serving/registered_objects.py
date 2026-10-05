"""Camera-owned registered regions. References are captured from shared inference."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from cvti.contracts import CandidateAlert
from cvti.detector.registered_object import RegisteredObjectMonitor


def source_key(source):
    return hashlib.sha256(str(source).encode()).hexdigest()


def candidate_current(candidate, site):
    metadata = getattr(candidate, "metadata", {}) or {}
    if metadata.get("state") != "registered_object_changed":
        return True
    for camera in site.get("cameras", []):
        if camera.get("id") != metadata.get("camera_id"):
            continue
        return any(entry["id"] == metadata.get("object_id") and
                   entry["reference_path"] == metadata.get("reference_revision") and
                   entry["source_fingerprint"] == source_key(camera.get("source"))
                   for entry in camera.get("registered_objects", []))
    return False


class RegisteredObjects:
    def __init__(self, camera_id, source, entries):
        self.camera_id, self.source = camera_id, source_key(source)
        self.entries = entries
        self.monitors = {}
        self.samples = {}
        self.last_sample = {}
        self.last_status = {}

    def status(self, entry, state, reason=None):
        if self.last_status.get(entry["id"]) == (state, reason):
            return
        path = Path(entry["reference_path"]).with_suffix(".status.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".status-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump({"state": state, "reason": reason, "updated_at": time.time()}, stream)
            os.replace(tmp, path)
            self.last_status[entry["id"]] = (state, reason)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def invalidate(self, reason="capture_reset"):
        for entry in self.entries:
            self.monitors[entry["id"]] = None
            self.status(entry, "revalidation_required", reason)
        self.samples.clear()

    def update(self, frame, detections, timestamp):
        results = []
        people = None if detections is None else [d.bbox for d in detections if d.label == "person"]
        for entry in self.entries:
            key = entry["id"]
            if entry["source_fingerprint"] != self.source or list(frame.shape[:2]) != entry["frame_hw"]:
                self.status(entry, "revalidation_required", "source_or_dimensions_changed")
                continue
            if key not in self.monitors:
                path = Path(entry["reference_path"])
                # Never resume a saved baseline silently after engine restart.
                if path.exists():
                    self.monitors[key] = None
                    self.status(entry, "revalidation_required", "engine_restarted")
                    continue
                samples = self.samples.setdefault(key, [])
                previous = self.last_sample.get(key)
                if previous is not None and (timestamp <= previous or timestamp - previous > 2.5):
                    samples.clear()
                if people is None or any(RegisteredObjectMonitor._coverage(p, entry["region"]) >= .1 for p in people):
                    samples.clear()
                    self.status(entry, "waiting_for_clear_view")
                    continue
                if previous is not None and 0 < timestamp - previous < .5:
                    continue
                self.last_sample[key] = timestamp
                samples.append(frame.copy())
                if len(samples) < 3:
                    self.status(entry, "capturing_reference")
                    continue
                monitor = RegisteredObjectMonitor(confirm_seconds=entry["confirm_seconds"])
                try:
                    monitor.register(samples[-3:], entry["region"], [[], [], []])
                except ValueError:
                    self.samples[key] = samples[-2:]
                    self.status(entry, "waiting_for_stable_view")
                    continue
                monitor.save_reference(path, camera_id=self.camera_id,
                                       source_fingerprint=self.source, name=entry["name"])
                self.monitors[key] = monitor
                self.samples.pop(key, None)
            monitor = self.monitors[key]
            if monitor is None:
                continue
            assessment = monitor.update(frame, timestamp, people)
            self.status(entry, assessment.state, assessment.reason)
            if not assessment.changed:
                continue
            panel = cv2.imdecode(np.frombuffer(assessment.evidence_png, np.uint8), cv2.IMREAD_COLOR)
            candidate = CandidateAlert(
                rule_name="registered_object_change", priority="medium", detector="object_state",
                title="Registered object changed", person_id=None, object_label=entry["name"],
                timestamp=timestamp,
                question="Compare the labelled registered reference and current view. Is there a visible change to the marked object, rather than occlusion, lighting or camera movement? Describe only visible evidence. Do not infer theft, ownership or the responsible person.",
                metadata={"zone": key, "state": "registered_object_changed", "bbox": entry["region"],
                          "camera_id": self.camera_id, "reference_revision": entry["reference_path"],
                          "object_id": key, "object_name": entry["name"], "evidence_kind": "before_after"},
            )
            results.append((candidate, panel))
        return results
