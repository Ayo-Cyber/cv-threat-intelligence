"""Track matched watched objects and derive business state transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cvti.object_watch.matcher import ObjectMatch


@dataclass(frozen=True)
class ObjectStateEvent:
    camera_id: str
    object_id: str
    object_label: str
    category: str
    state: str
    bbox: tuple[int, int, int, int]
    similarity: float
    timestamp: float
    track_id: int | None = None
    zone_id: str | None = None
    dwell_seconds: float = 0.0
    reasons: tuple[str, ...] = ()


@dataclass
class _TrackedObject:
    match: ObjectMatch
    first_seen: float
    last_seen: float
    stable_since: float
    stable_bbox: tuple[int, int, int, int]
    last_zone: str | None
    loaded_emitted: bool = False
    left_behind_emitted: bool = False


class ObjectStateTracker:
    def __init__(
        self,
        *,
        removed_grace_seconds: float = 3.0,
        stable_seen_seconds: float = 1.0,
        left_behind_seconds: float = 120.0,
        stationary_pixel_tolerance: float = 8.0,
    ) -> None:
        if removed_grace_seconds <= 0:
            raise ValueError("removed_grace_seconds must be positive")
        if stable_seen_seconds < 0:
            raise ValueError("stable_seen_seconds must be non-negative")
        if left_behind_seconds <= 0:
            raise ValueError("left_behind_seconds must be positive")
        if stationary_pixel_tolerance < 0:
            raise ValueError("stationary_pixel_tolerance must be non-negative")
        self.removed_grace_seconds = float(removed_grace_seconds)
        self.stable_seen_seconds = float(stable_seen_seconds)
        self.left_behind_seconds = float(left_behind_seconds)
        self.stationary_pixel_tolerance = float(stationary_pixel_tolerance)
        self._tracked: dict[tuple[Any, ...], _TrackedObject] = {}

    def update(
        self,
        matches: list[ObjectMatch],
        timestamp: float,
        vehicles: list[tuple[int, int, int, int]] = (),
    ) -> list[ObjectStateEvent]:
        timestamp = float(timestamp)
        events: list[ObjectStateEvent] = []
        seen_keys = set()
        for match in matches:
            key = self._key(match)
            seen_keys.add(key)
            state = self._tracked.get(key)
            if state is None:
                state = _TrackedObject(
                    match=match,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    stable_since=timestamp,
                    stable_bbox=match.bbox,
                    last_zone=match.zone_id,
                )
                self._tracked[key] = state
                events.append(self._event(match, "object_seen", timestamp))
                if match.zone_id:
                    events.append(self._event(
                        match, "object_entered_zone", timestamp,
                        reasons=(f"object entered {match.zone_id}",),
                    ))
            else:
                events.extend(self._update_existing(state, match, timestamp))
            events.extend(self._loaded_events(state, match, vehicles, timestamp))

        for key, state in list(self._tracked.items()):
            if key in seen_keys:
                continue
            if (timestamp - state.first_seen) < self.stable_seen_seconds:
                continue
            if (timestamp - state.last_seen) <= self.removed_grace_seconds:
                continue
            events.append(self._event(
                state.match,
                "object_removed",
                timestamp,
                dwell_seconds=max(0.0, state.last_seen - state.first_seen),
                reasons=("object disappeared after being stable",),
            ))
            del self._tracked[key]
        return events

    def _update_existing(
        self,
        state: _TrackedObject,
        match: ObjectMatch,
        timestamp: float,
    ) -> list[ObjectStateEvent]:
        events = [self._event(match, "object_seen", timestamp)]
        if state.last_zone != match.zone_id:
            if state.last_zone:
                events.append(self._event(
                    match,
                    "object_exited_zone",
                    timestamp,
                    zone_id=state.last_zone,
                    reasons=(f"object exited {state.last_zone}",),
                ))
            if match.zone_id:
                events.append(self._event(
                    match,
                    "object_entered_zone",
                    timestamp,
                    reasons=(f"object entered {match.zone_id}",),
                ))
            state.last_zone = match.zone_id

        if _center_distance(state.stable_bbox, match.bbox) > self.stationary_pixel_tolerance:
            state.stable_since = timestamp
            state.stable_bbox = match.bbox
            state.left_behind_emitted = False
        dwell = max(0.0, timestamp - state.stable_since)
        if dwell >= self.left_behind_seconds and not state.left_behind_emitted:
            events.append(self._event(
                match,
                "object_left_behind",
                timestamp,
                dwell_seconds=dwell,
                reasons=(f"object stationary for {dwell:.1f}s",),
            ))
            state.left_behind_emitted = True

        state.match = match
        state.last_seen = timestamp
        return events

    def _loaded_events(
        self,
        state: _TrackedObject,
        match: ObjectMatch,
        vehicles: list[tuple[int, int, int, int]],
        timestamp: float,
    ) -> list[ObjectStateEvent]:
        if state.loaded_emitted or match.category != "product" or match.zone_id != "loading_bay":
            return []
        if not any(_overlaps(match.bbox, vehicle) for vehicle in vehicles):
            return []
        state.loaded_emitted = True
        return [self._event(
            match,
            "object_loaded_near_vehicle",
            timestamp,
            reasons=("product overlaps a vehicle in the loading bay",),
        )]

    def _event(
        self,
        match: ObjectMatch,
        state: str,
        timestamp: float,
        *,
        zone_id: str | None = None,
        dwell_seconds: float = 0.0,
        reasons: tuple[str, ...] = (),
    ) -> ObjectStateEvent:
        return ObjectStateEvent(
            camera_id=match.camera_id,
            object_id=match.object_id,
            object_label=match.object_label,
            category=match.category,
            state=state,
            bbox=match.bbox,
            similarity=match.similarity,
            timestamp=timestamp,
            track_id=match.track_id,
            zone_id=match.zone_id if zone_id is None else zone_id,
            dwell_seconds=dwell_seconds,
            reasons=reasons,
        )

    @staticmethod
    def _key(match: ObjectMatch) -> tuple[Any, ...]:
        if match.track_id is not None:
            return (match.camera_id, match.object_id, match.track_id)
        cx, cy = _center(match.bbox)
        return (match.camera_id, match.object_id, round(cx / 32), round(cy / 32))


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _center_distance(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx, ly = _center(left)
    rx, ry = _center(right)
    return math.hypot(lx - rx, ly - ry)


def _overlaps(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    return min(lx2, rx2) > max(lx1, rx1) and min(ly2, ry2) > max(ly1, ry1)
