"""In-memory reference index and semantic crop matching decisions."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cvti.object_watch.embeddings import EmbeddingBackend, normalize_vector
from cvti.object_watch.images import encode_rgb_array
from cvti.object_watch.store import (
    ObjectTarget,
    library_lock,
    library_revision,
    load_embeddings,
    load_targets,
    target_readiness,
)


NEGATIVE_MARGIN = 0.05
RUNNER_UP_MARGIN = 0.03
_INDEX_CACHE: dict[tuple[str, int, str, int], "RecognitionIndex"] = {}
_INDEX_CACHE_LOCK = threading.Lock()


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
    target_revision: int = 0
    library_revision: int = 0
    reference_example_ids: tuple[str, ...] = ()
    model_fingerprint: str = ""


@dataclass(frozen=True)
class RecognitionDecision:
    status: str  # matched | ambiguous | rejected
    candidate: ObjectCandidate
    reason: str
    object_id: str | None = None
    best_similarity: float | None = None
    runner_up_similarity: float | None = None


@dataclass(frozen=True)
class IndexedTarget:
    target: ObjectTarget
    positives: tuple[tuple[str, tuple[float, ...]], ...]
    negatives: tuple[tuple[str, tuple[float, ...]], ...]


@dataclass(frozen=True)
class RecognitionIndex:
    library_revision: int
    model_name: str
    model_fingerprint: str
    preprocessing_version: int
    dimensions: int
    targets: tuple[IndexedTarget, ...]


def build_recognition_index(root: str | Path, backend: EmbeddingBackend) -> RecognitionIndex:
    """Load one immutable index for the current library/backend revision."""
    with library_lock(root):
        return _build_recognition_index_locked(root, backend)


def _build_recognition_index_locked(
    root: str | Path, backend: EmbeddingBackend,
) -> RecognitionIndex:
    revision = library_revision(root)
    key = (str(Path(root).resolve()), revision, backend.fingerprint,
           backend.preprocessing_version)
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None:
            return cached
    indexed: list[IndexedTarget] = []
    dimensions: set[int] = set()
    for target in load_targets(root):
        if target.review_state != "active":
            continue
        readiness = target_readiness(root, target, backend)
        if not readiness.ready:
            continue
        records = load_embeddings(root, target.id, backend.fingerprint)
        positives = tuple(
            (item.id, records[item.id].vector) for item in target.examples if item.reviewed
        )
        negatives = tuple(
            (item.id, records[item.id].vector) for item in target.negative_examples if item.reviewed
        )
        dimensions.update(len(vector) for _, vector in positives + negatives)
        indexed.append(IndexedTarget(target, positives, negatives))
    if len(dimensions) > 1:
        raise ValueError("recognition index contains mixed embedding dimensions")
    result = RecognitionIndex(
        library_revision=revision, model_name=backend.name,
        model_fingerprint=backend.fingerprint,
        preprocessing_version=backend.preprocessing_version,
        dimensions=next(iter(dimensions), int(getattr(backend, "dimensions", 0))),
        targets=tuple(indexed),
    )
    # Detect unsupported direct filesystem writes despite the snapshot lock.
    if library_revision(root) != revision:
        raise RuntimeError("object library changed while recognition index was built")
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE[key] = result
        if len(_INDEX_CACHE) > 16:
            _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
    return result


class ObjectMatcher:
    """Match candidates against a preloaded immutable library index.

    Similarities and default margins are uncalibrated starting thresholds, not
    probabilities. Rebuild/swap the index after a library revision changes.
    """

    def __init__(
        self,
        root: str | Path,
        backend: EmbeddingBackend,
        *,
        max_candidates_per_frame: int = 24,
        index: RecognitionIndex | None = None,
        runner_up_margin: float = RUNNER_UP_MARGIN,
    ) -> None:
        if max_candidates_per_frame <= 0:
            raise ValueError("max_candidates_per_frame must be positive")
        if not math.isfinite(runner_up_margin) or runner_up_margin < 0:
            raise ValueError("runner_up_margin must be finite and non-negative")
        self.root = Path(root)
        self.backend = backend
        self.index = index if index is not None else build_recognition_index(root, backend)
        if (self.index.model_fingerprint != backend.fingerprint
                or self.index.preprocessing_version != backend.preprocessing_version):
            raise ValueError("recognition index is incompatible with embedding backend")
        self.max_candidates_per_frame = int(max_candidates_per_frame)
        self.runner_up_margin = float(runner_up_margin)
        self.last_stats = _stats()
        self.last_decisions: tuple[RecognitionDecision, ...] = ()

    @classmethod
    def from_index(
        cls, backend: EmbeddingBackend, index: RecognitionIndex, *,
        max_candidates_per_frame: int = 24,
        runner_up_margin: float = RUNNER_UP_MARGIN,
    ) -> "ObjectMatcher":
        return cls(".", backend, max_candidates_per_frame=max_candidates_per_frame,
                   index=index, runner_up_margin=runner_up_margin)

    def match(
        self,
        camera_id: str,
        frame: Any,
        candidates: list[ObjectCandidate],
        timestamp: float,
    ) -> list[ObjectMatch]:
        ordered = sorted(candidates, key=lambda item: item.confidence, reverse=True)
        selected = ordered[:self.max_candidates_per_frame]
        self.last_stats = _stats(skipped=max(0, len(ordered) - len(selected)))
        matches: list[ObjectMatch] = []
        decisions: list[RecognitionDecision] = []
        for candidate in selected:
            encoded, bbox = self._crop_png(frame, candidate.bbox)
            if encoded is None:
                self.last_stats["invalid_candidates"] += 1
                decisions.append(RecognitionDecision("rejected", candidate, "invalid_bbox"))
                continue
            self.last_stats["processed_candidates"] += 1
            vector = normalize_vector(self.backend.embed_image(encoded))
            if self.index.dimensions and len(vector) != self.index.dimensions:
                raise ValueError("runtime embedding dimension does not match recognition index")
            ranked = self._rank(vector, candidate)
            if not ranked:
                decisions.append(RecognitionDecision("rejected", candidate, "no_plausible_target"))
                continue
            best_target, best_score, refs, negative_veto = ranked[0]
            runner = ranked[1][1] if len(ranked) > 1 else None
            if best_score < best_target.min_similarity:
                decisions.append(RecognitionDecision(
                    "rejected", candidate, "best_below_own_threshold",
                    best_target.id, best_score, runner,
                ))
                continue
            if negative_veto:
                decisions.append(RecognitionDecision(
                    "rejected", candidate, "best_negative_veto",
                    best_target.id, best_score, runner,
                ))
                continue
            if runner is not None and best_score - runner < self.runner_up_margin:
                decisions.append(RecognitionDecision(
                    "ambiguous", candidate, "runner_up_margin", best_target.id, best_score, runner
                ))
                continue
            matches.append(ObjectMatch(
                camera_id=camera_id, object_id=best_target.id,
                object_label=best_target.label, category=best_target.category,
                bbox=bbox, similarity=best_score, timestamp=float(timestamp),
                track_id=candidate.track_id, zone_id=candidate.zone_id,
                target_revision=best_target.revision,
                library_revision=self.index.library_revision,
                reference_example_ids=refs,
                model_fingerprint=self.index.model_fingerprint,
            ))
            decisions.append(RecognitionDecision(
                "matched", candidate, "positive_margin", best_target.id, best_score, runner
            ))
        self.last_stats["matches"] = len(matches)
        self.last_decisions = tuple(decisions)
        return matches

    def _rank(self, vector: tuple[float, ...], candidate: ObjectCandidate):
        ranked = []
        for indexed in self.index.targets:
            target = indexed.target
            if target.allowed_zone_ids and candidate.zone_id not in target.allowed_zone_ids:
                continue
            scores = [(example_id, _cosine(vector, reference))
                      for example_id, reference in indexed.positives]
            if not scores:
                continue
            positive = max(score for _, score in scores)
            negative = max((_cosine(vector, reference) for _, reference in indexed.negatives),
                           default=-1.0)
            refs = tuple(example_id for example_id, score in scores if score == positive)
            ranked.append((target, positive, refs, positive < negative + NEGATIVE_MARGIN))
        ranked.sort(key=lambda row: (-row[1], row[0].id))
        return ranked

    @staticmethod
    def _crop_png(frame: Any, bbox: tuple[int, int, int, int]):
        try:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = (int(value) for value in bbox)
        except (AttributeError, TypeError, ValueError):
            return None, (0, 0, 0, 0)
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None, (0, 0, 0, 0)
        crop = frame[y1:y2, x1:x2]
        try:
            return encode_rgb_array(crop, colour_space="bgr"), (x1, y1, x2, y2)
        except ValueError:
            return None, (0, 0, 0, 0)


def _stats(skipped: int = 0) -> dict[str, int]:
    return {"processed_candidates": 0, "skipped_over_budget": skipped,
            "invalid_candidates": 0, "matches": 0}


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        return -1.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator <= 0 or not math.isfinite(denominator):
        return -1.0
    return sum(a * b for a, b in zip(left, right)) / denominator
