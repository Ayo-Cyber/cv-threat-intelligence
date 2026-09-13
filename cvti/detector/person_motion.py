"""Reusable motion state for tracked person bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PersonMotion:
    track_id: int
    bbox: tuple[float, float, float, float]
    speed_ratio: float
    moving: bool
    moving_seconds: float
    zone_names: tuple[str, ...]


@dataclass
class _TrackState:
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    first_seen: float
    last_seen: float
    speed_ratio: float = 0.0
    moving: bool = False
    moving_since: float | None = None


class PersonMotionTracker:
    """Classify tracked people using smoothed, frame-normalized velocity."""

    def __init__(
        self,
        *,
        enter_speed_ratio: float = 0.05,
        exit_speed_ratio: float = 0.02,
        min_track_seconds: float = 0.4,
        ema_alpha: float = 0.20,
        track_expiry_seconds: float = 1.0,
    ) -> None:
        self.enter_speed_ratio = enter_speed_ratio
        self.exit_speed_ratio = exit_speed_ratio
        self.min_track_seconds = min_track_seconds
        self.ema_alpha = ema_alpha
        self.track_expiry_seconds = track_expiry_seconds
        self._tracks: dict[int, _TrackState] = {}

    def update(
        self,
        people: Sequence[tuple[int, float, float, float, float]],
        timestamp: float,
        frame_shape: tuple[int, int],
        zones_by_track: Mapping[int, Sequence[str]] | None = None,
    ) -> list[PersonMotion]:
        """Return current smoothed motion state and expire stale tracks."""
        stale_ids = [
            track_id
            for track_id, state in self._tracks.items()
            if timestamp - state.last_seen > self.track_expiry_seconds
        ]
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)

        frame_height, frame_width = frame_shape[:2]
        frame_diagonal = max(hypot(float(frame_width), float(frame_height)), 1.0)
        snapshots: list[PersonMotion] = []

        for track_id, x1, y1, x2, y2 in people:
            bbox = (float(x1), float(y1), float(x2), float(y2))
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            state = self._tracks.get(track_id)

            if state is None:
                state = _TrackState(
                    center=center,
                    bbox=bbox,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self._tracks[track_id] = state
            else:
                elapsed = timestamp - state.last_seen
                if elapsed > 0.0:
                    displacement = hypot(center[0] - state.center[0], center[1] - state.center[1])
                    measured_speed = displacement / elapsed / frame_diagonal
                    state.speed_ratio = (
                        self.ema_alpha * measured_speed
                        + (1.0 - self.ema_alpha) * state.speed_ratio
                    )

                track_age = max(0.0, timestamp - state.first_seen)
                if state.moving:
                    if state.speed_ratio <= self.exit_speed_ratio:
                        state.moving = False
                        state.moving_since = None
                elif (
                    track_age >= self.min_track_seconds
                    and state.speed_ratio >= self.enter_speed_ratio
                ):
                    state.moving = True
                    state.moving_since = timestamp

                state.center = center
                state.bbox = bbox
                state.last_seen = timestamp

            moving_seconds = (
                max(0.0, timestamp - state.moving_since)
                if state.moving and state.moving_since is not None
                else 0.0
            )
            snapshots.append(
                PersonMotion(
                    track_id=int(track_id),
                    bbox=state.bbox,
                    speed_ratio=state.speed_ratio,
                    moving=state.moving,
                    moving_seconds=moving_seconds,
                    zone_names=tuple((zones_by_track or {}).get(track_id, ())),
                )
            )

        return snapshots


def movement_event_metadata(motions: Sequence[PersonMotion]) -> dict:
    """Build one aggregate candidate from moving person snapshots."""
    x1 = min(motion.bbox[0] for motion in motions)
    y1 = min(motion.bbox[1] for motion in motions)
    x2 = max(motion.bbox[2] for motion in motions)
    y2 = max(motion.bbox[3] for motion in motions)
    return {
        "kind": "multiple_people_moving",
        "confidence": 1.0,
        "people_count": len(motions),
        "track_ids": [motion.track_id for motion in motions],
        "group_bbox": (x1, y1, x2, y2),
        "motions": [
            {
                "track_id": motion.track_id,
                "speed_ratio": motion.speed_ratio,
                "moving_seconds": motion.moving_seconds,
                "zone_names": list(motion.zone_names),
            }
            for motion in motions
        ],
    }


@dataclass
class SimultaneousMovementDetector:
    """Emit one candidate per sustained interval of simultaneous movement."""

    min_people: int = 2
    persistence_seconds: float = 0.5
    _active_since: float | None = field(default=None, init=False)
    _latched: bool = field(default=False, init=False)

    def update(self, motions: Sequence[PersonMotion], timestamp: float) -> dict | None:
        qualifying = [motion for motion in motions if motion.moving]
        if len(qualifying) < self.min_people:
            self._active_since = None
            self._latched = False
            return None
        if self._active_since is None:
            self._active_since = timestamp
        if self._latched or timestamp - self._active_since < self.persistence_seconds:
            return None
        self._latched = True
        return movement_event_metadata(qualifying)
