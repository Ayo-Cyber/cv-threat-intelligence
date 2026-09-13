"""Reusable motion state for tracked person bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PersonMotion:
    track_id: int
    bbox: tuple[float, float, float, float]
    speed_ratio: float
    moving: bool
    moving_seconds: float
    zone_names: tuple[str, ...]
    observed: bool = True


@dataclass
class _TrackState:
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    first_seen: float
    last_seen: float
    speed_ratio: float = 0.0
    moving: bool = False
    moving_since: float | None = None
    zone_names: tuple[str, ...] = ()


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
        values = {
            "enter_speed_ratio": enter_speed_ratio,
            "exit_speed_ratio": exit_speed_ratio,
            "min_track_seconds": min_track_seconds,
            "ema_alpha": ema_alpha,
            "track_expiry_seconds": track_expiry_seconds,
        }
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in values.values()):
            raise ValueError("motion tracker settings must be finite numbers")
        if not all(isfinite(float(value)) for value in values.values()):
            raise ValueError("motion tracker settings must be finite numbers")
        if enter_speed_ratio <= 0:
            raise ValueError("enter_speed_ratio must be positive")
        if exit_speed_ratio <= 0 or exit_speed_ratio >= enter_speed_ratio:
            raise ValueError("exit_speed_ratio must be positive and lower than enter_speed_ratio")
        if min_track_seconds < 0:
            raise ValueError("min_track_seconds must be nonnegative")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must satisfy 0 < ema_alpha <= 1")
        if track_expiry_seconds <= 0:
            raise ValueError("track_expiry_seconds must be positive")
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
        observed_ids: set[int] = set()

        for track_id, x1, y1, x2, y2 in people:
            observed_ids.add(int(track_id))
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
            state.zone_names = tuple((zones_by_track or {}).get(track_id, ()))

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
                    zone_names=state.zone_names,
                )
            )

        for track_id in sorted(self._tracks):
            if track_id in observed_ids:
                continue
            state = self._tracks[track_id]
            snapshots.append(
                PersonMotion(
                    track_id=track_id,
                    bbox=state.bbox,
                    speed_ratio=state.speed_ratio,
                    moving=state.moving,
                    moving_seconds=(
                        max(0.0, timestamp - state.moving_since)
                        if state.moving and state.moving_since is not None else 0.0
                    ),
                    zone_names=state.zone_names,
                    observed=False,
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
    _active_track_ids: tuple[int, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if isinstance(self.min_people, bool) or not isinstance(self.min_people, int):
            raise ValueError("min_people must be an integer")
        if self.min_people < 2:
            raise ValueError("min_people must be at least 2")
        if (isinstance(self.persistence_seconds, bool)
                or not isinstance(self.persistence_seconds, (int, float))
                or not isfinite(float(self.persistence_seconds))):
            raise ValueError("persistence_seconds must be a finite number")
        if self.persistence_seconds < 0:
            raise ValueError("persistence_seconds must be nonnegative")

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return self._active_track_ids

    def update(self, motions: Sequence[PersonMotion], timestamp: float) -> dict | None:
        qualifying = [motion for motion in motions if motion.moving]
        if len(qualifying) < self.min_people:
            self._active_since = None
            self._latched = False
            self._active_track_ids = ()
            return None
        self._active_track_ids = tuple(motion.track_id for motion in qualifying)
        if self._active_since is None:
            self._active_since = timestamp
        if self._latched or timestamp - self._active_since < self.persistence_seconds:
            return None
        self._latched = True
        return movement_event_metadata(qualifying)
