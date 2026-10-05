"""Conservative, zone-scoped object-change candidates for CHI KPI 9.

One monitored position per zone. This measures generic object occupancy, not
product identity, ownership or theft. The verification gate owns the verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from cvti.contracts import RawEvent


@dataclass(frozen=True)
class ObjectZonePolicy:
    zone: str
    labels: tuple[str, ...]
    mode: str
    dwell_seconds: float = 120.0
    absent_seconds: float = 5.0
    stable_seconds: float = 2.0
    max_gap_seconds: float = 2.5
    min_observations: int = 3
    min_confidence: float = 0.45

    def __post_init__(self):
        if not isinstance(self.zone, str) or not self.zone.strip():
            raise ValueError("object-state zone must be named")
        if self.mode not in {"left_behind", "removed"}:
            raise ValueError("object-state mode must be left_behind or removed")
        if not isinstance(self.labels, (list, tuple)) or not self.labels or any(
                not isinstance(label, str) or not label.strip() or label == "person"
                for label in self.labels):
            raise ValueError("object-state labels must name non-person detector classes")
        object.__setattr__(self, "labels", tuple(self.labels))
        for name in ("dwell_seconds", "absent_seconds", "stable_seconds", "max_gap_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                raise ValueError(f"object-state {name} must be finite and positive")
        if isinstance(self.min_observations, bool) or not isinstance(self.min_observations, int) \
                or self.min_observations < 3:
            raise ValueError("object-state min_observations must be at least three")
        if isinstance(self.min_confidence, bool) or not isinstance(self.min_confidence, (int, float)) \
                or not math.isfinite(self.min_confidence) or not 0 < self.min_confidence <= 1:
            raise ValueError("object-state min_confidence must be in (0, 1]")


@dataclass
class _Position:
    empty_since: float | None = None
    empty_samples: int = 0
    empty_frame: bytes | None = None
    cleared: bool = False
    label: str = ""
    bbox: tuple | None = None
    seen_since: float = 0.0
    seen_samples: int = 0
    qualified: bool = False
    occluded_since: float | None = None
    before: bytes | None = None
    appearance: np.ndarray | None = None
    missing_since: float | None = None
    missing_samples: int = 0
    emitted: bool = False


class ObjectZoneMonitor:
    def __init__(self, policies):
        if not isinstance(policies, (tuple, list)) or len(policies) > 16:
            raise ValueError("object_state_zones must be a list of at most 16 policies")
        self.policies = tuple(ObjectZonePolicy(**item) for item in policies)
        if len({p.zone for p in self.policies}) != len(self.policies):
            raise ValueError("object-state policies require distinct zones")
        self.reset()

    def reset(self):
        self._positions = {}
        self._last_timestamp = None
        self._signature = None
        self._background = None

    def update(self, frame, detections, timestamp, zones):
        """Return (RawEvent, labelled before/after panel) pairs for this frame.

        None detections means inference unavailable; [] means successful empty
        inference. Only consecutive, observable samples advance a decision.
        """
        now = float(timestamp)
        if not math.isfinite(now) or detections is None:
            self.reset()
            return []
        h, w = frame.shape[:2]
        polygons = {z.name: np.asarray(z.polygon, dtype=np.int32) for z in zones}
        signature = (h, w, tuple((name, tuple(map(tuple, polygon)))
                                for name, polygon in sorted(polygons.items())))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 48))
        if self._last_timestamp is not None and now <= self._last_timestamp:
            self.reset()
            return []
        if signature != self._signature:
            self.reset()
            self._signature = signature
        # Ignore policy regions when checking broad scene changes, so moving
        # the watched object itself does not count as a camera move.
        mask = np.ones((48, 64), dtype=np.uint8)
        for policy in self.policies:
            if policy.zone in polygons:
                scaled = polygons[policy.zone] * np.array([64 / w, 48 / h])
                cv2.fillPoly(mask, [scaled.astype(np.int32)], 0)
        visible = mask.astype(bool)
        scene_changed = (self._background is not None and visible.sum() >= 100 and
                         np.mean(np.abs(small.astype(float) - self._background)[visible]) > 20)
        if gray.mean() < 8 or gray.mean() > 247 or gray.std() < 3 or scene_changed:
            self._positions.clear()
            self._background = small.astype(float)
            self._last_timestamp = now
            return []
        self._background = small.astype(float)
        output = []
        for policy in self.policies:
            polygon = polygons.get(policy.zone)
            if polygon is None:
                self._positions.pop(policy.zone, None)
                continue
            if self._last_timestamp is not None and now - self._last_timestamp > policy.max_gap_seconds:
                self._positions.pop(policy.zone, None)
            position = self._positions.setdefault(policy.zone, _Position())
            x, y, width, height = cv2.boundingRect(polygon)
            zone_box = (max(0, x), max(0, y), min(w, x + width), min(h, y + height))
            # Pause during brief human handling; discard long occlusions.
            if any(d.label == "person" and _overlap(d.bbox, zone_box) for d in detections):
                if position.occluded_since is None:
                    position.occluded_since = now
                position.missing_since, position.missing_samples = None, 0
                position.seen_since, position.seen_samples = now, 0
                position.empty_since, position.empty_samples = None, 0
                if now - position.occluded_since > 30:
                    self._positions.pop(policy.zone, None)
                continue
            position.occluded_since = None
            targets = [d for d in detections if d.label in policy.labels
                       and d.confidence >= policy.min_confidence
                       and cv2.pointPolygonTest(polygon, _center(d.bbox), False) >= 0]
            if len(targets) > 1:
                self._positions.pop(policy.zone, None)
                continue
            if targets:
                target = targets[0]
                bbox = tuple(int(v) for v in target.bbox)
                if position.bbox is None or position.label != target.label or _moved(position.bbox, bbox):
                    position.label, position.bbox = target.label, bbox
                    position.seen_since, position.seen_samples = now, 0
                    position.before = _jpeg(frame)
                    position.appearance = _crop(gray, bbox)
                    position.emitted = False
                    position.qualified = False
                if position.seen_samples == 0:
                    position.seen_since = now
                position.seen_samples += 1
                if now - position.seen_since >= policy.stable_seconds \
                        and position.seen_samples >= policy.min_observations:
                    position.qualified = True
                position.missing_since, position.missing_samples = None, 0
                position.empty_since, position.empty_samples = None, 0
                if (policy.mode == "left_behind" and position.cleared and not position.emitted
                        and now - position.seen_since >= max(policy.dwell_seconds, policy.stable_seconds)
                        and position.qualified and _changed_from_empty(position, gray)):
                    output.append(_event(policy, position, frame, now, "object_left_behind",
                                         now - position.seen_since, position.empty_frame))
                    position.emitted = True
                continue
            if position.bbox is not None:
                # Restart placement dwell after a missed sample, retaining the
                # already qualified baseline needed for removal verification.
                position.seen_since, position.seen_samples = now, 0
                # An unmatched/low-confidence detection over the expected item
                # could be a missed classification or an occluder, not removal.
                if any(_overlap(d.bbox, position.bbox) for d in detections):
                    position.missing_since, position.missing_samples = None, 0
                    continue
                current = _crop(gray, position.bbox)
                changed = (current is not None and position.appearance is not None and
                           np.mean(np.abs(current - position.appearance)) >= 15)
                if not changed:
                    position.missing_since, position.missing_samples = None, 0
                    continue
                if position.missing_since is None:
                    position.missing_since = now
                position.missing_samples += 1
                if (now - position.missing_since >= policy.absent_seconds
                        and position.missing_samples >= policy.min_observations):
                    if policy.mode == "removed" and position.qualified:
                        output.append(_event(policy, position, frame, now, "object_removed",
                                             now - position.missing_since, position.before))
                    self._positions[policy.zone] = _Position(empty_since=now, empty_samples=1)
                continue
            if position.empty_since is None:
                position.empty_since = now
            position.empty_samples += 1
            if now - position.empty_since >= policy.stable_seconds \
                    and position.empty_samples >= policy.min_observations:
                position.cleared = True
                position.empty_frame = _jpeg(frame)
        self._last_timestamp = now
        return [item for item in output if item is not None]


def _center(bbox):
    return (float(bbox[0] + bbox[2]) / 2, float(bbox[1] + bbox[3]) / 2)


def _overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _moved(a, b):
    tolerance = max(3, math.hypot(a[2] - a[0], a[3] - a[1]) * 0.1)
    return math.dist(_center(a), _center(b)) > tolerance


def _crop(gray, bbox):
    h, w = gray.shape
    x1, y1, x2, y2 = bbox
    crop = gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    return cv2.resize(crop, (32, 32)).astype(float) if crop.size else None


def _jpeg(frame):
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv2.resize(frame, (960, max(1, round(h * 960 / w))))
    ok, encoded = cv2.imencode(".jpg", frame)
    return encoded.tobytes() if ok else None


def _changed_from_empty(position, gray):
    if position.empty_frame is None:
        return False
    before = cv2.imdecode(np.frombuffer(position.empty_frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if before is None:
        return False
    before = cv2.resize(before, (gray.shape[1], gray.shape[0]))
    a, b = _crop(before, position.bbox), _crop(gray, position.bbox)
    return a is not None and b is not None and np.mean(np.abs(a - b)) >= 15


def _event(policy, position, frame, now, state, seconds, before):
    if before is None:
        return None
    prior = cv2.imdecode(np.frombuffer(before, dtype=np.uint8), cv2.IMREAD_COLOR)
    if prior is None:
        return None
    h, w = frame.shape[:2]
    panes = []
    for label, image in (("BEFORE", prior), ("AFTER", frame)):
        pane = cv2.resize(image, (480, max(1, round(h * 480 / w))))
        a, b, c, d = [round(value * 480 / w) for value in position.bbox]
        cv2.rectangle(pane, (a, b), (c, d), (0, 200, 255), 2)
        pane = cv2.copyMakeBorder(pane, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(pane, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        panes.append(pane)
    event = RawEvent(
        detector="object_state", active=True, state=state,
        title="POSSIBLE OBJECT REMOVAL" if state == "object_removed" else "OBJECT LEFT IN MONITORED ZONE",
        level="medium", object_label=position.label, timestamp=now,
        extra={"zone": policy.zone, "state": state, "bbox": position.bbox,
               "dwell_seconds": seconds, "evidence_kind": "before_after",
               "identity_scope": "generic_detector_class",
               "reasons": [f"{position.label}: {state} in {policy.zone}; observed interval {seconds:.1f}s",
                           "Candidate only; removal does not imply theft or identify an owner."]},
    )
    return event, np.concatenate(panes, axis=1)
