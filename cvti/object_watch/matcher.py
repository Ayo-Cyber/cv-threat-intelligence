"""Match detected crops against reviewed object-watch targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cvti.object_watch.embeddings import EmbeddingBackend
from cvti.object_watch.store import ObjectTarget, load_embeddings, load_targets


NEGATIVE_MARGIN = 0.05


@dataclass(frozen=True)
class ObjectCandidate:
    bbox: tuple[int, int, int, int]
    label_hint: str = ""
    confidence: float = 0.0
    track_id: int | None = None
    zone_id: str | None = None


@dataclass(frozen=True)
class ObjectMatch:
    camera_id: str
    object_id: str
    object_label: str
    category: str
    bbox: tuple[int, int, int, int]
    similarity: float
    timestamp: float
    track_id: int | None = None
    zone_id: str | None = None


class ObjectMatcher:
    def __init__(
        self,
        root: str | Path,
        backend: EmbeddingBackend,
        *,
        max_candidates_per_frame: int = 24,
    ) -> None:
        if max_candidates_per_frame <= 0:
            raise ValueError("max_candidates_per_frame must be positive")
        self.root = Path(root)
        self.backend = backend
        self.max_candidates_per_frame = int(max_candidates_per_frame)
        self.last_stats = {
            "processed_candidates": 0,
            "skipped_over_budget": 0,
            "invalid_candidates": 0,
            "matches": 0,
        }

    def match(
        self,
        camera_id: str,
        frame: Any,
        candidates: list[ObjectCandidate],
        timestamp: float,
    ) -> list[ObjectMatch]:
        ordered = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        selected = ordered[: self.max_candidates_per_frame]
        self.last_stats = {
            "processed_candidates": 0,
            "skipped_over_budget": max(0, len(ordered) - len(selected)),
            "invalid_candidates": 0,
            "matches": 0,
        }
        targets = [t for t in load_targets(self.root) if t.review_state == "active"]
        matches: list[ObjectMatch] = []
        for candidate in selected:
            crop = self._crop_bytes(frame, candidate.bbox)
            if crop is None:
                self.last_stats["invalid_candidates"] += 1
                continue
            self.last_stats["processed_candidates"] += 1
            vector = self.backend.embed_image(crop)
            best = self._best_match(targets, vector, candidate)
            if best is not None:
                matches.append(ObjectMatch(
                    camera_id=camera_id,
                    object_id=best[0].id,
                    object_label=best[0].label,
                    category=best[0].category,
                    bbox=self._clamp_bbox(frame, candidate.bbox),
                    similarity=best[1],
                    timestamp=float(timestamp),
                    track_id=candidate.track_id,
                    zone_id=candidate.zone_id,
                ))
        self.last_stats["matches"] = len(matches)
        return matches

    def _best_match(
        self,
        targets: list[ObjectTarget],
        vector: tuple[float, ...],
        candidate: ObjectCandidate,
    ) -> tuple[ObjectTarget, float] | None:
        accepted: list[tuple[ObjectTarget, float]] = []
        for target in targets:
            if target.allowed_zone_ids and candidate.zone_id not in target.allowed_zone_ids:
                continue
            records = load_embeddings(self.root, target.id, self.backend.fingerprint)
            positive_ids = {example.id for example in target.examples}
            negative_ids = {example.id for example in target.negative_examples}
            positives = [
                _cosine(vector, record.vector)
                for example_id, record in records.items() if example_id in positive_ids
            ]
            if not positives:
                continue
            positive = max(positives)
            negatives = [
                _cosine(vector, record.vector)
                for example_id, record in records.items() if example_id in negative_ids
            ]
            negative = max(negatives) if negatives else -1.0
            if positive >= target.min_similarity and positive >= negative + NEGATIVE_MARGIN:
                accepted.append((target, positive))
        if not accepted:
            return None
        return max(accepted, key=lambda item: item[1])

    def _crop_bytes(
        self,
        frame: Any,
        bbox: tuple[int, int, int, int],
    ) -> bytes | None:
        x1, y1, x2, y2 = self._clamp_bbox_or_none(frame, bbox) or (0, 0, 0, 0)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if getattr(crop, "size", 0) <= 0:
            return None
        return bytes(crop.tobytes())

    def _clamp_bbox(
        self,
        frame: Any,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        clamped = self._clamp_bbox_or_none(frame, bbox)
        if clamped is None:
            return (0, 0, 0, 0)
        return clamped

    @staticmethod
    def _clamp_bbox_or_none(
        frame: Any,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        return -1.0
    denom = math.sqrt(sum(v * v for v in left)) * math.sqrt(sum(v * v for v in right))
    if denom == 0:
        return -1.0
    return sum(a * b for a, b in zip(left, right)) / denom
