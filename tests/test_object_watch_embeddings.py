from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from cvti.object_watch.embeddings import (
    HashEmbeddingBackend, embed_examples, load_embedding_backend, normalize_vector,
)
from cvti.object_watch.store import ObjectTarget, add_example, load_embeddings, save_target


def png(value: int = 20) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 12), (value, value, value)).save(output, "PNG")
    return output.getvalue()


def test_hash_backend_is_explicit_deterministic_test_double():
    backend = HashEmbeddingBackend(dimensions=8)
    assert backend.embed_image(b"chi carton") == backend.embed_image(b"chi carton")
    assert abs(sum(value * value for value in backend.embed_image(b"chi carton")) - 1) < 1e-6
    assert load_embedding_backend("hash").name == "hash"


def test_vector_validation_rejects_nan_zero_and_empty():
    for vector in ((), (0.0, 0.0), (float("nan"), 1.0)):
        with pytest.raises(ValueError):
            normalize_vector(vector)


def test_embed_examples_only_embeds_reviewed_canonical_crops(tmp_path: Path):
    save_target(tmp_path, ObjectTarget("chi-carton", "Chi carton", "product"))
    pending = add_example(tmp_path, "chi-carton", png(10), (0, 0, 1, 1), "upload")
    reviewed = add_example(tmp_path, "chi-carton", png(20), (0, 0, 1, 1),
                           "upload", reviewed=True)
    backend = HashEmbeddingBackend(dimensions=8)
    assert embed_examples(tmp_path, backend) == 1
    records = load_embeddings(tmp_path, "chi-carton", backend.fingerprint)
    assert set(records) == {reviewed.id}
    assert pending.id not in records
    assert embed_examples(tmp_path, backend) == 0


def test_siglip_requires_explicit_local_model_path():
    with pytest.raises(RuntimeError, match="local model_path is required"):
        load_embedding_backend("siglip")


def test_unknown_embedding_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown embedding backend"):
        load_embedding_backend("mystery")
