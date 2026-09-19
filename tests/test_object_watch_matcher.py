from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from cvti.object_watch.embeddings import HashEmbeddingBackend, embed_examples
from cvti.object_watch.matcher import (
    IndexedTarget, ObjectCandidate, ObjectMatcher, RecognitionIndex, build_recognition_index,
)
from cvti.object_watch.store import ObjectTarget, activate_target, add_example, save_target


def _frame(value: int, *, size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _png(frame: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(frame[:, :, ::-1]).save(output, "PNG")
    return output.getvalue()


def _active_target(tmp_path: Path, backend, *, object_id: str = "chi-carton",
                   negative: np.ndarray | None = None, zones=(), threshold=0.98):
    save_target(tmp_path, ObjectTarget(object_id, object_id.title(), "product",
                                       min_similarity=threshold, allowed_zone_ids=zones))
    add_example(tmp_path, object_id, _png(_frame(7)), (0, 0, 1, 1),
                "upload", reviewed=True)
    if negative is not None:
        add_example(tmp_path, object_id, _png(negative), (0, 0, 1, 1),
                    "negative", negative=True, reviewed=True)
    embed_examples(tmp_path, backend)
    return activate_target(tmp_path, object_id, backend)


def test_matcher_returns_versioned_match_from_preloaded_index(tmp_path: Path):
    backend = HashEmbeddingBackend(8)
    target = _active_target(tmp_path, backend)
    index = build_recognition_index(tmp_path, backend)
    matcher = ObjectMatcher.from_index(backend, index)
    matches = matcher.match("cam1", _frame(7), [
        ObjectCandidate((0, 0, 16, 16), "box", 0.8)
    ], 1.0)
    assert len(matches) == 1
    assert matches[0].object_id == "chi-carton"
    assert matches[0].target_revision == target.revision
    assert matches[0].library_revision == index.library_revision
    assert matches[0].model_fingerprint == backend.fingerprint
    assert matches[0].reference_example_ids
    assert matcher.last_decisions[0].status == "matched"


def test_negative_example_vetoes_match(tmp_path: Path):
    backend = HashEmbeddingBackend(8)
    _active_target(tmp_path, backend, negative=_frame(7))
    matcher = ObjectMatcher(tmp_path, backend)
    assert matcher.match("cam1", _frame(7), [ObjectCandidate((0, 0, 16, 16))], 1.0) == []
    assert matcher.last_decisions[0].status == "rejected"


def test_runner_up_is_ambiguous(tmp_path: Path):
    backend = HashEmbeddingBackend(8)
    _active_target(tmp_path, backend, object_id="target-a", threshold=0.5)
    _active_target(tmp_path, backend, object_id="target-b", threshold=0.5)
    matcher = ObjectMatcher(tmp_path, backend)
    assert matcher.match("cam1", _frame(7), [ObjectCandidate((0, 0, 16, 16))], 1.0) == []
    assert matcher.last_decisions[0].status == "ambiguous"


def test_match_does_not_reread_library(tmp_path: Path):
    backend = HashEmbeddingBackend(8)
    _active_target(tmp_path, backend)
    matcher = ObjectMatcher(tmp_path, backend)
    (tmp_path / "object_library" / "targets.json").write_text("{broken")
    assert matcher.match("cam1", _frame(7), [ObjectCandidate((0, 0, 16, 16))], 1.0)


def test_candidate_cap_zone_and_invalid_boxes(tmp_path: Path):
    backend = HashEmbeddingBackend(8)
    _active_target(tmp_path, backend, zones=("storage",))
    matcher = ObjectMatcher(tmp_path, backend, max_candidates_per_frame=2)
    assert matcher.match("cam1", _frame(7), [
        ObjectCandidate((20, 20, 25, 25), confidence=0.9),
        ObjectCandidate((0, 0, 16, 16), confidence=0.8, zone_id="loading"),
        ObjectCandidate((0, 0, 16, 16), confidence=0.1, zone_id="storage"),
    ], 1.0) == []
    assert matcher.last_stats["processed_candidates"] == 1
    assert matcher.last_stats["invalid_candidates"] == 1
    assert matcher.last_stats["skipped_over_budget"] == 1


class _FixedBackend:
    name = "fixed"
    fingerprint = "fixed-fp"
    preprocessing_version = 1
    dimensions = 2

    def embed_image(self, image_bytes: bytes) -> tuple[float, ...]:
        return (1.0, 0.0)


def _indexed(*targets: IndexedTarget) -> RecognitionIndex:
    return RecognitionIndex(1, "fixed", "fixed-fp", 1, 2, targets)


def _candidate_target(object_id: str, threshold: float, vector, negative=None) -> IndexedTarget:
    target = ObjectTarget(object_id, object_id, "product", review_state="active",
                          min_similarity=threshold)
    negatives = () if negative is None else (("neg", negative),)
    return IndexedTarget(target, (("positive", vector),), negatives)


def test_higher_candidate_failing_own_threshold_blocks_lower_label():
    index = _indexed(
        _candidate_target("a", .95, (.93, (1 - .93 ** 2) ** .5)),
        _candidate_target("b", .80, (.92, (1 - .92 ** 2) ** .5)),
    )
    matcher = ObjectMatcher.from_index(_FixedBackend(), index)
    assert matcher.match("cam", _frame(1), [ObjectCandidate((0, 0, 16, 16))], 1) == []
    assert matcher.last_decisions[0].reason == "best_below_own_threshold"


def test_negative_veto_on_best_does_not_fall_through_to_lower_label():
    index = _indexed(
        _candidate_target("a", .7, (1.0, 0.0), negative=(1.0, 0.0)),
        _candidate_target("b", .7, (.8, .6)),
    )
    matcher = ObjectMatcher.from_index(_FixedBackend(), index)
    assert matcher.match("cam", _frame(1), [ObjectCandidate((0, 0, 16, 16))], 1) == []
    assert matcher.last_decisions[0].reason == "best_negative_veto"


def test_near_tie_considers_runner_before_its_own_threshold():
    index = _indexed(
        _candidate_target("a", .7, (.94, (1 - .94 ** 2) ** .5)),
        _candidate_target("b", .99, (.93, (1 - .93 ** 2) ** .5)),
    )
    matcher = ObjectMatcher.from_index(_FixedBackend(), index)
    assert matcher.match("cam", _frame(1), [ObjectCandidate((0, 0, 16, 16))], 1) == []
    assert matcher.last_decisions[0].status == "ambiguous"
