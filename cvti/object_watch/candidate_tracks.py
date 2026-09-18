"""Bounded geometric association for object-watch proposal candidates."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Iterable

import numpy as np
import supervision as sv

from cvti.detector.object_tracks import GeneralObjectTracker
from cvti.object_watch.matcher import ObjectCandidate


class CandidateTrackAssociator:
    """Associate one generic object class while preserving supplied track IDs."""

    def __init__(self, *, camera_id: str, source_generation: int | str,
                 expected_fps: float, max_tracks: int) -> None:
        self._tracker = GeneralObjectTracker(
            camera_id=camera_id,
            session_id=str(source_generation),
            names={1: "object"},
            person_class_ids=(),
            expected_fps=expected_fps,
            max_tracks=max_tracks,
            max_detections=max_tracks,
        )
        namespace = hashlib.sha256(
            f"{camera_id}\0{source_generation}".encode("utf-8")
        ).digest()
        # Keep generated IDs negative and within JavaScript's exact integer
        # range while reserving a large, stable block per camera/generation.
        self._next_local_id = -((int.from_bytes(namespace[:3], "big") + 1) * 1_000_000)
        self._external_by_internal: dict[int, int] = {}
        self._local_by_internal: dict[int, int] = {}

    def assign(self, candidates: Iterable[ObjectCandidate],
               timestamp: float) -> tuple[ObjectCandidate, ...]:
        rows = tuple(candidates)
        detections = (sv.Detections.empty() if not rows else sv.Detections(
            xyxy=np.asarray([item.bbox for item in rows], dtype=float).reshape((-1, 4)),
            confidence=np.asarray([item.confidence for item in rows], dtype=float),
            class_id=np.ones(len(rows), dtype=int),
        ))
        self._tracker.update(detections, timestamp)
        snapshot = self._tracker.snapshot(timestamp)
        tracks = [item for item in snapshot["tracks"] if item["state"] == "observed"]
        assignments = _one_to_one(rows, tracks)

        retained = {_internal_id(item) for item in snapshot["tracks"]}
        self._external_by_internal = {
            key: value for key, value in self._external_by_internal.items() if key in retained
        }
        self._local_by_internal = {
            key: value for key, value in self._local_by_internal.items() if key in retained
        }

        inherited = {int(item.track_id) for item in rows if item.track_id is not None}
        for index, internal in assignments.items():
            supplied = rows[index].track_id
            if supplied is not None:
                self._external_by_internal[internal] = int(supplied)

        used = set(inherited)
        result: list[ObjectCandidate] = []
        for index, candidate in enumerate(rows):
            if candidate.track_id is not None:
                result.append(candidate)
                continue
            internal = assignments.get(index)
            if internal is None:
                result.append(candidate)
                continue
            track_id = self._external_by_internal.get(internal)
            if track_id is None or track_id in used:
                track_id = self._local_by_internal.get(internal)
                if track_id is None or track_id in used:
                    track_id = self._allocate(used)
                    self._local_by_internal[internal] = track_id
            used.add(track_id)
            result.append(replace(candidate, track_id=track_id))
        return tuple(result)

    def _allocate(self, occupied: set[int]) -> int:
        while self._next_local_id in occupied:
            self._next_local_id -= 1
        value = self._next_local_id
        self._next_local_id -= 1
        return value


def _internal_id(track: dict) -> int:
    return int(str(track["id"]).rsplit("/", 1)[-1])


def _one_to_one(candidates: tuple[ObjectCandidate, ...], tracks: list[dict]) -> dict[int, int]:
    ranked = []
    for candidate_index, candidate in enumerate(candidates):
        for track_index, track in enumerate(tracks):
            overlap = _iou(candidate.bbox, track["bbox"])
            if overlap > 0:
                ranked.append((-overlap, candidate_index, track_index))
    assigned_candidates: set[int] = set()
    assigned_tracks: set[int] = set()
    result: dict[int, int] = {}
    for _negative_iou, candidate_index, track_index in sorted(ranked):
        if candidate_index in assigned_candidates or track_index in assigned_tracks:
            continue
        assigned_candidates.add(candidate_index)
        assigned_tracks.add(track_index)
        result[candidate_index] = _internal_id(tracks[track_index])
    return result


def _iou(left, right) -> float:
    x1, y1, x2, y2 = left
    a1, b1, a2, b2 = right
    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    union = (max(0, x2 - x1) * max(0, y2 - y1)
             + max(0, a2 - a1) * max(0, b2 - b1) - intersection)
    return intersection / union if union else 0.0
