"""Bounded, per-class object tracking built on supervision's ByteTrack."""

from __future__ import annotations

import math
import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import supervision as sv


@dataclass
class _Track:
    counter: int
    public_id: str
    class_id: int
    label: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    first_seen: float
    last_seen: float
    state: str = "observed"
    ended_at: Optional[float] = None
    end_reason: Optional[str] = None
    trajectory: deque = field(default_factory=deque)


class GeneralObjectTracker:
    """Track non-person detector results without sharing association across classes."""

    def __init__(
        self,
        *,
        camera_id: str,
        session_id: str,
        names: Mapping[int, str],
        person_class_ids: Tuple[int, ...] = (0,),
        expected_fps: float = 5.0,
        lost_after_seconds: float = 2.0,
        overlay_max_age_seconds: float = 0.5,
        trajectory_points: int = 32,
        max_tracks: int = 128,
        max_detections: int = 256,
        ended_retention_seconds: float = 5.0,
        max_ended_tracks: int = 128,
    ) -> None:
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError("camera_id must be a non-empty string")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(names, Mapping):
            raise ValueError("names must be a mapping")

        copied_names: Dict[int, str] = {}
        for class_id, label in names.items():
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise ValueError("names keys must be integer class ids")
            if not isinstance(label, str):
                raise ValueError("names values must be strings")
            copied_names[int(class_id)] = label or f"class_{class_id}"
        if not isinstance(person_class_ids, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in person_class_ids
        ):
            raise ValueError("person_class_ids must be a tuple of integer class ids")

        self._positive_float(expected_fps, "expected_fps")
        self._positive_float(lost_after_seconds, "lost_after_seconds")
        self._positive_float(overlay_max_age_seconds, "overlay_max_age_seconds")
        self._positive_float(ended_retention_seconds, "ended_retention_seconds")
        for value, name in (
            (trajectory_points, "trajectory_points"),
            (max_tracks, "max_tracks"),
            (max_detections, "max_detections"),
            (max_ended_tracks, "max_ended_tracks"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.camera_id = camera_id
        self.session_id = session_id
        self.names = dict(copied_names)
        self.person_class_ids = tuple(person_class_ids)
        self.expected_fps = float(expected_fps)
        self.lost_after_seconds = float(lost_after_seconds)
        self.overlay_max_age_seconds = float(overlay_max_age_seconds)
        self.trajectory_points = trajectory_points
        self.max_tracks = max_tracks
        self.max_detections = max_detections
        self.ended_retention_seconds = float(ended_retention_seconds)
        self.max_ended_tracks = max_ended_tracks
        self._trackable_classes = tuple(
            sorted(class_id for class_id in self.names if class_id not in self.person_class_ids)
        )
        self._lost_track_buffer = max(1, math.ceil(expected_fps * lost_after_seconds))
        self._trackers = {class_id: self._new_tracker() for class_id in self._trackable_classes}
        self._associations: Dict[Tuple[int, int], int] = {}
        self._tracks: Dict[int, _Track] = {}
        self._next_counter = 1
        self._last_update_at: Optional[float] = None
        self._last_success_at: Optional[float] = None
        self._continuity_started_at: Optional[float] = None
        self._last_available: Optional[bool] = None
        self._invalid_detections = 0
        self._dropped_detections = 0
        self._capacity_resets = 0
        self._lock = threading.RLock()

    @staticmethod
    def _positive_float(value: float, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite positive number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")

    @staticmethod
    def _timestamp(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("timestamp must be finite")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("timestamp must be finite")
        return result

    def _new_tracker(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return sv.ByteTrack(frame_rate=30, lost_track_buffer=self._lost_track_buffer)

    def _reset_association(self) -> None:
        for tracker in self._trackers.values():
            tracker.reset()
        self._associations.clear()

    def _end(self, track: _Track, ended_at: float, reason: str) -> None:
        if track.state == "ended":
            return
        track.state = "ended"
        track.ended_at = float(ended_at)
        track.end_reason = reason
        stale_keys = [key for key, value in self._associations.items() if value == track.counter]
        for key in stale_keys:
            del self._associations[key]

    def _end_expired(self, timestamp: float) -> None:
        for track in self._tracks.values():
            if track.state != "ended" and timestamp - track.last_seen > self.lost_after_seconds:
                self._end(track, track.last_seen + self.lost_after_seconds, "lost_timeout")

    def _enforce_live_bound(self, timestamp: float) -> None:
        live = [track for track in self._tracks.values() if track.state != "ended"]
        if len(live) <= self.max_tracks:
            return
        live.sort(
            key=lambda track: (
                track.state == "observed",
                track.last_seen,
                track.counter,
            )
        )
        for track in live[: len(live) - self.max_tracks]:
            self._end(track, timestamp, "capacity_reset")

    def _trim_ended(self, timestamp: float) -> None:
        ended = sorted(
            (track for track in self._tracks.values() if track.state == "ended"),
            key=lambda track: ((track.ended_at or 0.0), track.counter),
        )
        removable = {
            track.counter
            for track in ended
            if track.ended_at is not None
            and timestamp - track.ended_at > self.ended_retention_seconds
        }
        retained = [
            track
            for track in ended
            if track.counter not in removable
        ]
        if len(retained) > self.max_ended_tracks:
            removable.update(
                track.counter for track in retained[: len(retained) - self.max_ended_tracks]
            )
        for counter in removable:
            self._tracks.pop(counter, None)

    def _valid_rows(self, detections: sv.Detections) -> List[Tuple[int, float, np.ndarray]]:
        rows: List[Tuple[int, float, np.ndarray]] = []
        confidence = detections.confidence
        class_ids = detections.class_id
        for index in range(len(detections)):
            try:
                bbox = np.asarray(detections.xyxy[index], dtype=float).copy()
                score = float(confidence[index]) if confidence is not None else float("nan")
                raw_class = class_ids[index] if class_ids is not None else None
                class_value = float(raw_class) if raw_class is not None else float("nan")
                class_id = int(class_value)
                valid_class = math.isfinite(class_value) and class_value == class_id
            except (IndexError, TypeError, ValueError, OverflowError):
                self._invalid_detections += 1
                continue
            if class_id in self.person_class_ids:
                continue
            valid = (
                bbox.shape == (4,)
                and bool(np.isfinite(bbox).all())
                and bbox[2] > bbox[0]
                and bbox[3] > bbox[1]
                and math.isfinite(score)
                and 0.0 <= score <= 1.0
                and valid_class
                and class_id in self.names
            )
            if not valid:
                self._invalid_detections += 1
                continue
            rows.append((class_id, score, bbox))

        rows.sort(key=lambda row: (-row[1], row[0], *row[2].tolist()))
        limit = min(self.max_detections, self.max_tracks)
        if len(rows) > limit:
            self._dropped_detections += len(rows) - limit
            rows = rows[:limit]
        return rows

    @staticmethod
    def _detections_for(rows: List[Tuple[int, float, np.ndarray]], class_id: int) -> sv.Detections:
        selected = [row for row in rows if row[0] == class_id]
        if not selected:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.asarray([row[2] for row in selected], dtype=float),
            confidence=np.asarray([row[1] for row in selected], dtype=float),
            class_id=np.full(len(selected), class_id, dtype=int),
        )

    def _run_trackers(
        self, rows: List[Tuple[int, float, np.ndarray]]
    ) -> Dict[int, sv.Detections]:
        return {
            class_id: self._trackers[class_id].update_with_detections(
                self._detections_for(rows, class_id)
            )
            for class_id in self._trackable_classes
        }

    def _internal_track_count(self) -> int:
        identities = set()
        for tracker in self._trackers.values():
            for attribute in ("tracked_tracks", "lost_tracks", "removed_tracks"):
                identities.update(id(track) for track in getattr(tracker, attribute, ()))
        return len(identities)

    def _publish_outputs(self, outputs: Dict[int, sv.Detections], timestamp: float) -> None:
        observed = set()
        for class_id in self._trackable_classes:
            tracked = outputs[class_id]
            tracker_ids = tracked.tracker_id
            if tracker_ids is None:
                continue
            for index, raw_tracker_id in enumerate(tracker_ids):
                tracker_id = int(raw_tracker_id)
                if tracker_id < 0:
                    continue
                key = (class_id, tracker_id)
                counter = self._associations.get(key)
                raw_bbox = tracked.xyxy[index]
                bbox = (
                    float(raw_bbox[0]),
                    float(raw_bbox[1]),
                    float(raw_bbox[2]),
                    float(raw_bbox[3]),
                )
                confidence = float(tracked.confidence[index])
                if counter is None:
                    counter = self._next_counter
                    self._next_counter += 1
                    track = _Track(
                        counter=counter,
                        public_id=f"{self.camera_id}/{self.session_id}/{counter}",
                        class_id=class_id,
                        label=self.names[class_id],
                        bbox=bbox,
                        confidence=confidence,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        trajectory=deque(maxlen=self.trajectory_points),
                    )
                    self._tracks[counter] = track
                    self._associations[key] = counter
                else:
                    track = self._tracks[counter]
                    track.bbox = bbox
                    track.confidence = confidence
                    track.last_seen = timestamp
                    track.state = "observed"
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                track.trajectory.append((timestamp, cx, cy))
                observed.add(counter)

        for track in self._tracks.values():
            if track.state != "ended" and track.counter not in observed:
                track.state = "lost"

    def update(self, detections: Optional[sv.Detections], timestamp: float) -> None:
        timestamp = self._timestamp(timestamp)
        with self._lock:
            if self._last_update_at is not None:
                if timestamp < self._last_update_at:
                    raise ValueError("timestamp cannot move backwards; call reset before replay")
                if timestamp == self._last_update_at:
                    return

            if detections is not None and not isinstance(detections, sv.Detections):
                raise TypeError("detections must be supervision.Detections or None")

            long_gap = (
                self._last_success_at is not None
                and timestamp - self._last_success_at > self.lost_after_seconds
            )
            self._end_expired(timestamp)
            if long_gap:
                self._reset_association()

            self._last_update_at = timestamp
            if self._continuity_started_at is None:
                self._continuity_started_at = timestamp
            self._last_available = detections is not None
            if detections is None:
                for track in self._tracks.values():
                    if track.state != "ended":
                        track.state = "lost"
                self._trim_ended(timestamp)
                return

            rows = self._valid_rows(detections)
            outputs = self._run_trackers(rows)
            if self._internal_track_count() > self.max_tracks:
                for track in self._tracks.values():
                    if track.state != "ended":
                        self._end(track, timestamp, "capacity_reset")
                self._capacity_resets += 1
                self._reset_association()
                outputs = self._run_trackers(rows)

            self._publish_outputs(outputs, timestamp)
            self._last_success_at = timestamp
            self._end_expired(timestamp)
            self._enforce_live_bound(timestamp)
            self._trim_ended(timestamp)

    def _project(self, track: _Track, timestamp: float) -> dict:
        state = track.state
        ended_at = track.ended_at
        end_reason = track.end_reason
        if state != "ended" and timestamp - track.last_seen > self.lost_after_seconds:
            state = "ended"
            ended_at = track.last_seen + self.lost_after_seconds
            end_reason = "lost_timeout"
        return {
            "id": track.public_id,
            "class_id": track.class_id,
            "label": track.label,
            "state": state,
            "bbox": [float(value) for value in track.bbox],
            "confidence": float(track.confidence),
            "first_seen": float(track.first_seen),
            "last_seen": float(track.last_seen),
            "ended_at": None if ended_at is None else float(ended_at),
            "end_reason": end_reason,
            "trajectory": [[float(t), float(x), float(y)] for t, x, y in track.trajectory],
        }

    def _status(self, timestamp: float) -> str:
        if self._last_update_at is None:
            return "starting"
        continuity_at = self._last_success_at
        if continuity_at is None:
            continuity_at = self._continuity_started_at
        assert continuity_at is not None
        if timestamp - continuity_at > self.lost_after_seconds:
            return "stale"
        if self._last_available is False:
            return "unavailable"
        return "ok"

    def snapshot(self, timestamp: float) -> dict:
        timestamp = self._timestamp(timestamp)
        with self._lock:
            projected_tracks = []
            for track in sorted(self._tracks.values(), key=lambda item: item.counter):
                projected = self._project(track, timestamp)
                ended_at = projected["ended_at"]
                if (
                    projected["state"] == "ended"
                    and ended_at is not None
                    and timestamp - ended_at > self.ended_retention_seconds
                ):
                    continue
                projected_tracks.append((track.counter, projected))

            live = [item for item in projected_tracks if item[1]["state"] != "ended"]
            if len(live) > self.max_tracks:
                live = live[-self.max_tracks :]
            ended = [item for item in projected_tracks if item[1]["state"] == "ended"]
            if len(ended) > self.max_ended_tracks:
                ended.sort(
                    key=lambda item: ((item[1]["ended_at"] or 0.0), item[0])
                )
                ended = ended[-self.max_ended_tracks :]
            tracks = [projected for _, projected in sorted(live + ended)]
            return {
                "schema_version": 1,
                "camera_id": self.camera_id,
                "session_id": self.session_id,
                "timestamp": timestamp,
                "last_update_at": self._last_update_at,
                "last_success_at": self._last_success_at,
                "status": self._status(timestamp),
                "tracks": tracks,
                "counters": {
                    "invalid_detections": self._invalid_detections,
                    "dropped_detections": self._dropped_detections,
                    "capacity_resets": self._capacity_resets,
                },
            }

    def overlays(self, timestamp: float) -> List[dict]:
        timestamp = self._timestamp(timestamp)
        with self._lock:
            result = []
            for track in sorted(self._tracks.values(), key=lambda item: item.counter):
                if track.state != "observed":
                    continue
                if timestamp - track.last_seen > self.overlay_max_age_seconds:
                    continue
                result.append(
                    {
                        "track_id": -track.counter,
                        "bbox": tuple(int(round(value)) for value in track.bbox),
                        "label": f"{track.label} O{track.counter}",
                        "colour": (255, 180, 0),
                        "namespace": "object",
                    }
                )
            return result

    def reset(self, timestamp: float, reason: str = "source_reset") -> None:
        timestamp = self._timestamp(timestamp)
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")
        with self._lock:
            for track in self._tracks.values():
                if track.state != "ended":
                    self._end(track, timestamp, reason)
            self._reset_association()
            self._last_update_at = None
            self._last_success_at = None
            self._continuity_started_at = None
            self._last_available = None
            self._trim_ended(timestamp)
