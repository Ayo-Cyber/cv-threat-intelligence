from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cvti.object_watch.embeddings import (
    HashEmbeddingBackend,
    embed_examples,
    load_embedding_backend,
)
from cvti.object_watch.store import (
    ObjectTarget,
    add_example,
    load_embeddings,
    save_target,
)


def test_hash_backend_is_deterministic_and_normalized():
    backend = HashEmbeddingBackend(dimensions=8)

    first = backend.embed_image(b"chi carton")
    second = backend.embed_image(b"chi carton")

    assert first == second
    assert abs(sum(v * v for v in first) - 1.0) < 1e-6
    assert backend.name == "hash"
    assert backend.fingerprint == "hash-8-v1"


def test_hash_backend_rejects_empty_bytes():
    backend = HashEmbeddingBackend(dimensions=8)

    with pytest.raises(ValueError, match="image bytes are required"):
        backend.embed_image(b"")


def test_embed_examples_writes_records_for_missing_embeddings(tmp_path: Path):
    save_target(tmp_path, ObjectTarget(
        id="chi-carton",
        label="Chi carton",
        category="product",
        aliases=(),
        review_state="draft",
        min_similarity=0.72,
        allowed_zone_ids=(),
        examples=(),
    ))
    add_example(tmp_path, "chi-carton", b"\xff\xd8carton\xff\xd9", (0, 0, 12, 12), "upload")

    written = embed_examples(tmp_path, HashEmbeddingBackend(dimensions=8))

    records = load_embeddings(tmp_path, "chi-carton", "hash-8-v1")
    assert written == 1
    assert len(records) == 1
    assert next(iter(records.values())).model_name == "hash"


def test_embed_examples_skips_existing_matching_embedding(tmp_path: Path):
    save_target(tmp_path, ObjectTarget(
        id="chi-carton",
        label="Chi carton",
        category="product",
        aliases=(),
        review_state="draft",
        min_similarity=0.72,
        allowed_zone_ids=(),
        examples=(),
    ))
    add_example(tmp_path, "chi-carton", b"\xff\xd8carton\xff\xd9", (0, 0, 12, 12), "upload")
    backend = HashEmbeddingBackend(dimensions=8)

    assert embed_examples(tmp_path, backend) == 1
    assert embed_examples(tmp_path, backend) == 0


def test_siglip_backend_failure_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(RuntimeError, match="SigLIP embedding backend unavailable"):
        load_embedding_backend("siglip")


def test_unknown_embedding_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown embedding backend"):
        load_embedding_backend("mystery")
