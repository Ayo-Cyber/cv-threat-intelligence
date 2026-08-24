"""Lightweight situational HSE candidate detectors.

These are intentionally small candidate generators for demo/pilot use. They do
not replace the VLM gate; they create cheap temporal signals that the existing
Customization Engine and Verification Gate can confirm with scene context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _frame_diagonal(frame_shape: tuple[int, ...]) -> float:
    h, w = frame_shape[:2]
    return max((float(w) ** 2 + float(h) ** 2) ** 0.5, 1.0)


@dataclass
class _TrackMotion:
    center: tuple[float, float]
    timestamp: float
    fast_frames: int = 0
    latched: bool = False


@dataclass
class RunningPanicDetector:
    """Detect sustained fast person movement from tracked bbox centers."""

    min_speed_ratio: float = 0.18
    min_frames: int = 3
    reset_speed_ratio: float = 0.08
    _tracks: dict[int, _TrackMotion] = field(default_factory=dict)

    def update(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        timestamp: float,
        frame_shape: tuple[int, ...],
    ) -> dict[str, Any] | None:
        center = _bbox_center(bbox)
        # Sweep departed tracks: ByteTrack ids never return, so anything not
        # updated for 30s is a person who left — one leaked _TrackMotion per
        # visitor otherwise, forever. (RAM audit 24 Aug, #3.)
        stale = [t for t, m in self._tracks.items() if timestamp - m.timestamp > 30.0]
        for t in stale:
            self._tracks.pop(t, None)
        previous = self._tracks.get(track_id)
        if previous is None:
            self._tracks[track_id] = _TrackMotion(center=center, timestamp=timestamp)
            return None

        dt = max(timestamp - previous.timestamp, 1e-3)
        dx = center[0] - previous.center[0]
        dy = center[1] - previous.center[1]
        speed_px = ((dx * dx + dy * dy) ** 0.5) / dt
        speed_ratio = speed_px / _frame_diagonal(frame_shape)

        fast_frames = previous.fast_frames + 1 if speed_ratio >= self.min_speed_ratio else 0
        latched = previous.latched
        fired = None
        if fast_frames >= self.min_frames and not latched:
            latched = True
            fired = {
                "kind": "running",
                "track_id": track_id,
                "bbox": bbox,
                "speed_ratio": round(speed_ratio, 4),
                "fast_frames": fast_frames,
            }
        if speed_ratio <= self.reset_speed_ratio:
            latched = False

        self._tracks[track_id] = _TrackMotion(
            center=center,
            timestamp=timestamp,
            fast_frames=fast_frames,
            latched=latched,
        )
        return fired


@dataclass
class CrowdFormationDetector:
    """Detect a persistent tight cluster of people."""

    min_people: int = 4
    min_frames: int = 3
    max_cluster_ratio: float = 0.24
    _cluster_frames: int = 0
    _latched: bool = False

    def update(
        self,
        people: list[dict[str, Any]],
        timestamp: float,
        frame_shape: tuple[int, ...],
    ) -> dict[str, Any] | None:
        if len(people) < self.min_people:
            self._cluster_frames = 0
            self._latched = False
            return None

        centers = [_bbox_center(tuple(p["bbox"])) for p in people]
        best: list[int] = []
        frame_diag = _frame_diagonal(frame_shape)
        max_distance = self.max_cluster_ratio * frame_diag
        for idx, center in enumerate(centers):
            members = [
                j
                for j, other in enumerate(centers)
                if ((center[0] - other[0]) ** 2 + (center[1] - other[1]) ** 2) ** 0.5
                <= max_distance
            ]
            if len(members) > len(best):
                best = members

        if len(best) < self.min_people:
            self._cluster_frames = 0
            self._latched = False
            return None

        self._cluster_frames += 1
        if self._cluster_frames < self.min_frames or self._latched:
            return None

        self._latched = True
        member_people = [people[i] for i in best]
        xs: list[int] = []
        ys: list[int] = []
        for person in member_people:
            x1, y1, x2, y2 = tuple(person["bbox"])
            xs.extend([int(x1), int(x2)])
            ys.extend([int(y1), int(y2)])
        return {
            "kind": "crowd_formation",
            "people_count": len(member_people),
            "track_ids": [p.get("track_id") for p in member_people],
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "cluster_frames": self._cluster_frames,
            "timestamp": timestamp,
        }


@dataclass
class FireSmokeCandidateDetector:
    """Detect persistent flame-colored or smoke-like frame regions."""

    min_frames: int = 3
    min_hot_area_ratio: float = 0.012
    min_smoke_area_ratio: float = 0.08
    max_smoke_area_ratio: float = 0.65
    _candidate_frames: int = 0
    _latched: bool = False

    def update(self, frame: np.ndarray, timestamp: float) -> dict[str, Any] | None:
        if frame.size == 0:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hot_mask = cv2.inRange(hsv, np.array([0, 70, 130]), np.array([35, 255, 255]))
        smoke_mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 60, 230]))

        total = float(frame.shape[0] * frame.shape[1])
        hot_area_ratio = float(cv2.countNonZero(hot_mask)) / max(total, 1.0)
        smoke_area_ratio = float(cv2.countNonZero(smoke_mask)) / max(total, 1.0)
        smoke_candidate = self.min_smoke_area_ratio <= smoke_area_ratio <= self.max_smoke_area_ratio
        candidate = hot_area_ratio >= self.min_hot_area_ratio or smoke_candidate

        if not candidate:
            self._candidate_frames = 0
            self._latched = False
            return None
        self._candidate_frames += 1
        if self._candidate_frames < self.min_frames or self._latched:
            return None

        self._latched = True
        return {
            "kind": "fire_smoke",
            "hot_area_ratio": round(hot_area_ratio, 4),
            "smoke_area_ratio": round(smoke_area_ratio, 4),
            "candidate_frames": self._candidate_frames,
            "timestamp": timestamp,
        }
