from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from cvti.object_watch.embeddings import HashEmbeddingBackend, embed_examples
from cvti.object_watch.matcher import ObjectCandidate, ObjectMatcher
from cvti.object_watch.store import ObjectTarget, add_example, load_targets, save_target


def _frame(value: int, *, size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _active_target(tmp_path: Path, *, negative: bytes | None = None) -> None:
    save_target(tmp_path, ObjectTarget(
        id="chi-carton",
        label="Chi carton",
        category="product",
        aliases=(),
        review_state="draft",
        min_similarity=0.98,
        allowed_zone_ids=(),
        examples=(),
    ))
    add_example(tmp_path, "chi-carton", _frame(7).tobytes(), (0, 0, 16, 16), "upload")
    if negative is not None:
        add_example(tmp_path, "chi-carton", negative, (0, 0, 16, 16), "negative", negative=True)
    loaded = load_targets(tmp_path)[0]
    save_target(tmp_path, replace(loaded, review_state="active"))
    embed_examples(tmp_path, HashEmbeddingBackend(dimensions=8))


def test_matcher_returns_best_positive_above_threshold(tmp_path: Path):
    _active_target(tmp_path)
    matcher = ObjectMatcher(tmp_path, HashEmbeddingBackend(dimensions=8))

    matches = matcher.match("cam1", _frame(7), [
        ObjectCandidate(bbox=(0, 0, 16, 16), label_hint="box", confidence=0.8),
    ], timestamp=1.0)

    assert len(matches) == 1
    assert matches[0].object_id == "chi-carton"
    assert matches[0].object_label == "Chi carton"
    assert matches[0].similarity >= 0.98


def test_negative_example_can_veto_similar_plain_box(tmp_path: Path):
    _active_target(tmp_path, negative=_frame(7).tobytes())
    matcher = ObjectMatcher(tmp_path, HashEmbeddingBackend(dimensions=8))

    matches = matcher.match("cam1", _frame(7), [
        ObjectCandidate(bbox=(0, 0, 16, 16), label_hint="box", confidence=0.8),
    ], timestamp=1.0)

    assert matches == []


def test_max_candidates_per_frame_is_enforced(tmp_path: Path):
    _active_target(tmp_path)
    matcher = ObjectMatcher(
        tmp_path, HashEmbeddingBackend(dimensions=8), max_candidates_per_frame=1
    )

    matcher.match("cam1", _frame(7), [
        ObjectCandidate(bbox=(0, 0, 8, 8), confidence=0.2),
        ObjectCandidate(bbox=(0, 0, 16, 16), confidence=0.9),
    ], timestamp=1.0)

    assert matcher.last_stats["processed_candidates"] == 1
    assert matcher.last_stats["skipped_over_budget"] == 1


def test_zone_scoped_target_does_not_match_outside_zone(tmp_path: Path):
    save_target(tmp_path, ObjectTarget(
        id="chi-carton",
        label="Chi carton",
        category="product",
        aliases=(),
        review_state="draft",
        min_similarity=0.98,
        allowed_zone_ids=("storage",),
        examples=(),
    ))
    add_example(tmp_path, "chi-carton", _frame(7).tobytes(), (0, 0, 16, 16), "upload")
    loaded = load_targets(tmp_path)[0]
    save_target(tmp_path, replace(loaded, review_state="active"))
    embed_examples(tmp_path, HashEmbeddingBackend(dimensions=8))
    matcher = ObjectMatcher(tmp_path, HashEmbeddingBackend(dimensions=8))

    matches = matcher.match("cam1", _frame(7), [
        ObjectCandidate(bbox=(0, 0, 16, 16), zone_id="loading_bay", confidence=0.8),
    ], timestamp=1.0)

    assert matches == []


def test_zero_area_or_out_of_frame_candidate_is_skipped(tmp_path: Path):
    _active_target(tmp_path)
    matcher = ObjectMatcher(tmp_path, HashEmbeddingBackend(dimensions=8))

    matches = matcher.match("cam1", _frame(7), [
        ObjectCandidate(bbox=(20, 20, 25, 25), confidence=0.9),
        ObjectCandidate(bbox=(1, 1, 1, 8), confidence=0.8),
    ], timestamp=1.0)

    assert matches == []
    assert matcher.last_stats["invalid_candidates"] == 2
