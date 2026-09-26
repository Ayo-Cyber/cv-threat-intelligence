from __future__ import annotations

import io
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from cvti.object_watch.embeddings import HashEmbeddingBackend, embed_examples
from cvti.object_watch.store import (
    EmbeddingRecord, ObjectExample, ObjectTarget, activate_target, add_example,
    library_revision, load_embeddings, load_targets, review_example, save_target,
    target_readiness, targets_needing_reembed, write_embedding,
)


def image_bytes(value: int = 40, size: int = 16) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (size, size), (value, value, value)).save(out, "PNG")
    return out.getvalue()


def target(**changes) -> ObjectTarget:
    base = ObjectTarget(
        id="chi-carton", label="Chi carton", category="product",
        aliases=("milk carton",), min_similarity=0.72,
        allowed_zone_ids=("loading_bay",),
    )
    return replace(base, **changes)


def test_active_state_requires_explicit_readiness_activation(tmp_path: Path):
    with pytest.raises(ValueError, match="use activate_target"):
        save_target(tmp_path, target(review_state="active"))
    save_target(tmp_path, target())
    with pytest.raises(ValueError, match="no reviewed positive"):
        activate_target(tmp_path, "chi-carton", HashEmbeddingBackend(8))


def test_embedding_model_fingerprint_controls_reembed(tmp_path: Path):
    save_target(tmp_path, target())
    example = add_example(tmp_path, "chi-carton", image_bytes(), (0, 0, 1, 1),
                          "upload", reviewed=True)
    write_embedding(tmp_path, "chi-carton", example.id, EmbeddingRecord(
        "test-embed", "fp-a", 1, example.sha256, (1.0, 0.0, 0.0)))
    assert targets_needing_reembed(tmp_path, "fp-a") == []
    assert [item.id for item in targets_needing_reembed(tmp_path, "fp-b")] == ["chi-carton"]


def test_examples_are_canonical_unreviewed_and_round_trip(tmp_path: Path):
    save_target(tmp_path, target())
    positive = add_example(tmp_path, "chi-carton", image_bytes(), (0, 0, 1, 1), "upload")
    negative = add_example(tmp_path, "chi-carton", image_bytes(80), (0, 0, 1, 1),
                           "negative_upload", negative=True)
    loaded = load_targets(tmp_path)[0]
    assert loaded.examples == (positive,)
    assert loaded.negative_examples == (negative,)
    assert not positive.reviewed
    assert positive.path.endswith(".png")
    assert (tmp_path / "object_library" / positive.path).read_bytes().startswith(b"\x89PNG")


def test_review_embedding_readiness_and_mutation_deactivates(tmp_path: Path):
    backend = HashEmbeddingBackend(8)
    save_target(tmp_path, target())
    example = add_example(tmp_path, "chi-carton", image_bytes(), (0, 0, 1, 1), "upload")
    assert not target_readiness(tmp_path, "chi-carton", backend).ready
    review_example(tmp_path, "chi-carton", example.id)
    assert embed_examples(tmp_path, backend) == 1
    active = activate_target(tmp_path, "chi-carton", backend)
    assert active.review_state == "active"
    revision = active.revision
    changed = save_target(tmp_path, replace(active, grounding_description="blue carton"))
    assert changed.review_state == "draft"
    assert changed.revision == revision + 1


def test_legacy_example_is_not_crop_ready(tmp_path: Path):
    save_target(tmp_path, target())
    library = tmp_path / "object_library"
    doc = json.loads((library / "targets.json").read_text(encoding="utf-8"))
    doc["targets"][0]["examples"] = [{
        "id": "ex-legacy", "source": "upload", "path": "examples/chi-carton/ex-legacy.jpg",
        "bbox": [0, 0, 1, 1], "sha256": "0" * 64, "reviewed": True,
    }]
    (library / "targets.json").write_text(json.dumps(doc))
    readiness = target_readiness(tmp_path, "chi-carton", HashEmbeddingBackend(8))
    assert not readiness.ready
    assert "canonical crop preprocessing" in " ".join(readiness.reasons)


def test_concurrent_mutations_do_not_lose_targets(tmp_path: Path):
    threads = [threading.Thread(target=save_target, args=(tmp_path, target(
        id=f"target-{index}", label=f"Target {index}"))) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(load_targets(tmp_path)) == 8
    assert library_revision(tmp_path) == 8


def test_concurrent_examples_on_one_target_are_serialized(tmp_path: Path):
    save_target(tmp_path, target())
    threads = [threading.Thread(
        target=add_example,
        args=(tmp_path, "chi-carton", image_bytes(index + 1), (0, 0, 1, 1), "upload"),
    ) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(load_targets(tmp_path)[0].examples) == 6


def test_add_example_rejects_invalid_data_and_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="unsafe object id"):
        add_example(tmp_path, "../escape", image_bytes(), (0, 0, 1, 1), "upload")
    save_target(tmp_path, target())
    with pytest.raises(ValueError, match="decodable"):
        add_example(tmp_path, "chi-carton", b"not-image", (0, 0, 1, 1), "upload")


def test_invalid_json_is_retained_and_atomic_temps_are_unique(tmp_path: Path):
    library = tmp_path / "object_library"
    library.mkdir()
    targets = library / "targets.json"
    targets.write_text("{not json")
    with pytest.raises(ValueError, match="invalid object target store"):
        load_targets(tmp_path)
    assert targets.read_text(encoding="utf-8") == "{not json"


def test_load_embeddings_rejects_mismatched_crop_hash(tmp_path: Path):
    save_target(tmp_path, target())
    example = add_example(tmp_path, "chi-carton", image_bytes(), (0, 0, 1, 1),
                          "upload", reviewed=True)
    write_embedding(tmp_path, "chi-carton", example.id, EmbeddingRecord(
        "test-embed", "fp-a", 1, "0" * 64, (1.0, 0.0)))
    with pytest.raises(ValueError, match="crop hash mismatch"):
        load_embeddings(tmp_path, "chi-carton", "fp-a")


def test_concurrent_embedding_commit_is_idempotent_and_publishes_revision(tmp_path: Path):
    save_target(tmp_path, target())
    example = add_example(tmp_path, "chi-carton", image_bytes(), (0, 0, 1, 1),
                          "upload", reviewed=True)
    target_revision = load_targets(tmp_path)[0].revision
    before = library_revision(tmp_path)
    record = EmbeddingRecord("test", "fp", 1, example.sha256, (1.0, 0.0))
    results = []

    def commit():
        results.append(write_embedding(
            tmp_path, "chi-carton", example.id, record,
            expected_crop_sha256=example.sha256, expected_reviewed=True,
            expected_target_revision=target_revision,
        ))

    threads = [threading.Thread(target=commit) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 1
    assert library_revision(tmp_path) == before + 1


def test_readiness_validates_backend_expected_dimension(tmp_path: Path):
    backend = HashEmbeddingBackend(8)
    save_target(tmp_path, target())
    example = add_example(tmp_path, "chi-carton", image_bytes(), (0, 0, 1, 1),
                          "upload", reviewed=True)
    write_embedding(tmp_path, "chi-carton", example.id, EmbeddingRecord(
        backend.name, backend.fingerprint, backend.preprocessing_version,
        example.sha256, (1.0, 0.0)))
    assert "dimension mismatch" in " ".join(
        target_readiness(tmp_path, "chi-carton", backend).reasons)
