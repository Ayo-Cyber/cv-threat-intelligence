"""Local embedding backends for object-watch targets."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol

from cvti.object_watch.store import (
    EmbeddingRecord,
    load_embeddings,
    load_targets,
    write_embedding,
)


class EmbeddingBackend(Protocol):
    name: str
    fingerprint: str
    preprocessing_version: int

    def embed_image(self, image_bytes: bytes) -> tuple[float, ...]:
        ...


class HashEmbeddingBackend:
    """Deterministic no-dependency backend for tests and offline plumbing.

    This is not a semantic vision model. It exists so storage, matching, and
    scoring can be tested without downloading weights.
    """

    name = "hash"
    preprocessing_version = 1

    def __init__(self, dimensions: int = 32) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = int(dimensions)
        self.fingerprint = f"hash-{self.dimensions}-v1"

    def embed_image(self, image_bytes: bytes) -> tuple[float, ...]:
        if not image_bytes:
            raise ValueError("image bytes are required")
        values: list[float] = []
        counter = 0
        while len(values) < self.dimensions:
            digest = hashlib.sha256(
                image_bytes + counter.to_bytes(4, "big")
            ).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        vector = values[: self.dimensions]
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            raise ValueError("embedding norm is zero")
        return tuple(v / norm for v in vector)


class SiglipEmbeddingBackend:
    """Lazy optional semantic embedding backend.

    The constructor imports optional packages only when explicitly requested.
    It never downloads model weights implicitly; callers must provide a local
    model path or a pre-cached model name according to their environment.
    """

    name = "siglip"
    preprocessing_version = 1

    def __init__(self, model: str = "google/siglip-base-patch16-224") -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel  # type: ignore
            import torch  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover - exact dependency varies
            raise RuntimeError(f"SigLIP embedding backend unavailable: {exc}") from exc
        self._torch = torch
        self._image_cls = Image
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                model, local_files_only=True
            )
            self._model = AutoModel.from_pretrained(model, local_files_only=True)
        except Exception as exc:  # pragma: no cover - requires local weights
            raise RuntimeError(f"SigLIP embedding backend unavailable: {exc}") from exc
        self._model.eval()
        self.model_name = model
        self.fingerprint = "siglip-" + hashlib.sha256(model.encode()).hexdigest()[:12]

    def embed_image(self, image_bytes: bytes) -> tuple[float, ...]:
        if not image_bytes:
            raise ValueError("image bytes are required")
        import io

        image = self._image_cls.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
        vector = features[0].float()
        vector = vector / vector.norm()
        return tuple(float(v) for v in vector.tolist())


def load_embedding_backend(name: str = "hash") -> EmbeddingBackend:
    normalized = name.strip().lower()
    if normalized == "hash":
        return HashEmbeddingBackend()
    if normalized == "siglip":
        return SiglipEmbeddingBackend()
    raise ValueError(f"unknown embedding backend: {name}")


def _example_bytes(root: str | Path, relative_path: str) -> bytes:
    path = Path(root) / "object_library" / relative_path
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"example crop is unreadable: {relative_path}") from exc


def embed_examples(root: str | Path, backend: EmbeddingBackend) -> int:
    written = 0
    for target in load_targets(root):
        existing = load_embeddings(root, target.id, backend.fingerprint)
        for example in target.examples + target.negative_examples:
            record = existing.get(example.id)
            if record is not None and record.crop_sha256 == example.sha256:
                continue
            image_bytes = _example_bytes(root, example.path)
            vector = backend.embed_image(image_bytes)
            write_embedding(
                root,
                target.id,
                example.id,
                EmbeddingRecord(
                    model_name=backend.name,
                    model_fingerprint=backend.fingerprint,
                    preprocessing_version=backend.preprocessing_version,
                    crop_sha256=example.sha256,
                    vector=vector,
                ),
            )
            written += 1
    return written
