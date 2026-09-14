from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvti.object_watch.store import (
    EmbeddingRecord,
    ObjectExample,
    ObjectTarget,
    add_example,
    load_embeddings,
    load_targets,
    save_target,
    targets_needing_reembed,
    write_embedding,
)


JPEG_BYTES = b"\xff\xd8example-image\xff\xd9"


def target(
    *,
    id: str = "chi-carton",
    label: str = "Chi carton",
    category: str = "product",
    review_state: str = "draft",
    examples: tuple[ObjectExample, ...] = (),
    negative_examples: tuple[ObjectExample, ...] = (),
) -> ObjectTarget:
    return ObjectTarget(
        id=id,
        label=label,
        category=category,
        aliases=("milk carton",),
        review_state=review_state,
        min_similarity=0.72,
        allowed_zone_ids=("loading_bay",),
        examples=examples,
        negative_examples=negative_examples,
    )


def test_save_target_requires_reviewed_examples_before_activation(tmp_path: Path):
    with pytest.raises(ValueError, match="active target requires at least one example"):
        save_target(tmp_path, target(review_state="active"))


def test_embedding_model_fingerprint_controls_reembed(tmp_path: Path):
    save_target(tmp_path, target())
    example = add_example(tmp_path, "chi-carton", JPEG_BYTES, (0, 0, 10, 10), "upload")
    write_embedding(tmp_path, "chi-carton", example.id, EmbeddingRecord(
        model_name="test-embed",
        model_fingerprint="fp-a",
        preprocessing_version=1,
        crop_sha256=example.sha256,
        vector=(1.0, 0.0, 0.0),
    ))

    assert targets_needing_reembed(tmp_path, "fp-a") == []
    assert [t.id for t in targets_needing_reembed(tmp_path, "fp-b")] == ["chi-carton"]


def test_load_targets_round_trips_examples_and_negative_examples(tmp_path: Path):
    save_target(tmp_path, target())
    positive = add_example(tmp_path, "chi-carton", JPEG_BYTES, (0, 0, 10, 10), "upload")
    negative = add_example(
        tmp_path, "chi-carton", b"\xff\xd8plain-box\xff\xd9",
        (1, 1, 11, 11), "negative_upload", negative=True
    )

    loaded = load_targets(tmp_path)

    assert loaded == (
        target(examples=(positive,), negative_examples=(negative,)),
    )
    assert (tmp_path / "object_library" / positive.path).is_file()
    assert (tmp_path / "object_library" / negative.path).is_file()


def test_add_example_rejects_path_traversal_object_id(tmp_path: Path):
    with pytest.raises(ValueError, match="unsafe object id"):
        add_example(tmp_path, "../escape", JPEG_BYTES, (0, 0, 10, 10), "upload")


def test_load_targets_rejects_invalid_json_without_deleting_file(tmp_path: Path):
    library = tmp_path / "object_library"
    library.mkdir()
    targets = library / "targets.json"
    targets.write_text("{not json")

    with pytest.raises(ValueError, match="invalid object target store"):
        load_targets(tmp_path)

    assert targets.read_text() == "{not json"


def test_save_target_replaces_targets_atomically(tmp_path: Path):
    save_target(tmp_path, target(label="Old label"))
    save_target(tmp_path, target(label="New label"))

    saved = json.loads((tmp_path / "object_library" / "targets.json").read_text())

    assert [row["id"] for row in saved["targets"]] == ["chi-carton"]
    assert saved["targets"][0]["label"] == "New label"
    assert not (tmp_path / "object_library" / "targets.json.tmp").exists()


def test_unknown_category_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown object category"):
        save_target(tmp_path, target(category="sku"))


def test_load_embeddings_rejects_mismatched_crop_hash(tmp_path: Path):
    save_target(tmp_path, target())
    example = add_example(tmp_path, "chi-carton", JPEG_BYTES, (0, 0, 10, 10), "upload")
    write_embedding(tmp_path, "chi-carton", example.id, EmbeddingRecord(
        model_name="test-embed",
        model_fingerprint="fp-a",
        preprocessing_version=1,
        crop_sha256="0" * 64,
        vector=(1.0, 0.0),
    ))

    with pytest.raises(ValueError, match="crop hash mismatch"):
        load_embeddings(tmp_path, "chi-carton", "fp-a")
