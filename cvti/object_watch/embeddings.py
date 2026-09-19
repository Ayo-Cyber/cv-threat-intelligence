"""Local semantic embedding backends for canonical object-watch crops."""

from __future__ import annotations

import hashlib
import io
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cvti.object_watch.images import IMAGE_PREPROCESSING_VERSION
from cvti.object_watch.store import EmbeddingRecord, load_embeddings, load_targets, write_embedding


class EmbeddingBackend(Protocol):
    name: str
    fingerprint: str
    preprocessing_version: int
    dimensions: int

    def embed_image(self, image_bytes: bytes) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class BackendMetadata:
    """Lightweight compatibility contract; creating it never loads model code."""

    name: str
    fingerprint: str
    preprocessing_version: int
    dimensions: int
    structurally_available: bool = True
    executable_verified: bool = False


def normalize_vector(values) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding vector must be nonzero")
    return tuple(value / norm for value in vector)


class HashEmbeddingBackend:
    """Explicit test double only; production config rejects this backend."""

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
            digest = hashlib.sha256(image_bytes + counter.to_bytes(4, "big")).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        return normalize_vector(values[:self.dimensions])

    @property
    def metadata(self) -> BackendMetadata:
        return BackendMetadata(self.name, self.fingerprint, self.preprocessing_version,
                               self.dimensions, executable_verified=True)


_MODEL_CACHE: "OrderedDict[tuple[str, str], tuple[str, tuple, Any, Any, Any, threading.Lock]]" = OrderedDict()
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE_LIMIT = 2


class SiglipEmbeddingBackend:
    """Local-files-only SigLIP image encoder, shared and serialized per process."""

    name = "siglip"
    preprocessing_version = 1

    def __init__(
        self,
        model: str,
        *,
        device: str = "cpu",
        fingerprint: str | None = None,
        inference_lock_root: str | Path | None = None,
    ) -> None:
        path = Path(model).expanduser().resolve()
        if not path.is_dir():
            raise RuntimeError("SigLIP embedding backend unavailable: local model directory is missing")
        from cvti.object_watch.runtime_config import (
            _artifact_fingerprint, _model_artifact_files, _model_dimensions,
        )
        artifacts = _model_artifact_files(path)
        if not artifacts:
            raise RuntimeError("SigLIP embedding backend unavailable: model artifacts are missing")
        before = _artifact_signature(artifacts)
        loaded_fingerprint = _artifact_fingerprint(artifacts, IMAGE_PREPROCESSING_VERSION)
        if fingerprint is not None and fingerprint != loaded_fingerprint:
            raise RuntimeError("SigLIP embedding backend unavailable: model fingerprint changed")
        dimensions = _model_dimensions(path / "config.json")
        if dimensions is None:
            raise RuntimeError("SigLIP embedding backend unavailable: embedding dimension is missing")
        try:
            from transformers import AutoImageProcessor, AutoModel  # type: ignore
            import torch  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"SigLIP embedding backend unavailable: {exc}") from exc
        key = (str(path), device)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is None or cached[0] != loaded_fingerprint or cached[1] != before:
                try:
                    processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
                    model_instance = AutoModel.from_pretrained(str(path), local_files_only=True)
                    model_instance.to(device)
                    model_instance.eval()
                except Exception as exc:
                    raise RuntimeError(f"SigLIP embedding backend unavailable: {exc}") from exc
                after = _artifact_signature(artifacts)
                if after != before:
                    raise RuntimeError("SigLIP embedding backend unavailable: model artifacts changed during load")
                cached = (loaded_fingerprint, after, processor, model_instance, torch, threading.Lock())
                _MODEL_CACHE[key] = cached
                while len(_MODEL_CACHE) > _MODEL_CACHE_LIMIT:
                    _MODEL_CACHE.popitem(last=False)
            else:
                _MODEL_CACHE.move_to_end(key)
        cached_fingerprint, cached_signature, self._processor, self._model, self._torch, self._lock = cached
        if cached_fingerprint != loaded_fingerprint or cached_signature != _artifact_signature(artifacts):
            raise RuntimeError("SigLIP embedding backend unavailable: cached model artifacts changed")
        self._image_cls = Image
        self.device = device
        self.model_name = str(path)
        self.fingerprint = loaded_fingerprint
        self.dimensions = dimensions
        self.inference_lock_root = Path(inference_lock_root) if inference_lock_root else None

    def embed_image(self, image_bytes: bytes) -> tuple[float, ...]:
        if not image_bytes:
            raise ValueError("image bytes are required")
        try:
            image = self._image_cls.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError("image bytes are not a decodable canonical image") from exc
        from cvti.object_watch.runtime_config import advisory_inference_lock

        root = self.inference_lock_root or Path(self.model_name).parent
        with advisory_inference_lock(root, blocking=True) as acquired:
            if not acquired:  # blocking locks always acquire; retained for a single contract.
                raise RuntimeError("object watch inference is busy")
            with self._lock:
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self._torch.no_grad():
                    features = self._model.get_image_features(**inputs)
                vector = normalize_vector(features[0].float().cpu().tolist())
                if len(vector) != self.dimensions:
                    raise RuntimeError("SigLIP runtime embedding dimension does not match model metadata")
                return vector

    @property
    def metadata(self) -> BackendMetadata:
        return BackendMetadata(self.name, self.fingerprint, self.preprocessing_version,
                               self.dimensions, executable_verified=True)


def load_embedding_backend(
    name: str = "hash", *, model_path: str | Path | None = None,
    device: str = "cpu", fingerprint: str | None = None,
    inference_lock_root: str | Path | None = None,
) -> EmbeddingBackend:
    """Compatibility factory; hash remains available only when explicitly asked."""
    normalized = name.strip().lower()
    if normalized == "hash":
        return HashEmbeddingBackend()
    if normalized == "siglip":
        if model_path is None:
            raise RuntimeError("SigLIP embedding backend unavailable: local model_path is required")
        return SiglipEmbeddingBackend(str(model_path), device=device, fingerprint=fingerprint,
                                      inference_lock_root=inference_lock_root)
    raise ValueError(f"unknown embedding backend: {name}")


def embed_examples(root: str | Path, backend: EmbeddingBackend) -> int:
    """Embed reviewed canonical crops only; legacy whole snapshots are skipped."""
    written = 0
    snapshot = load_targets(root)
    for target in snapshot:
        existing = load_embeddings(root, target.id, backend.fingerprint)
        for example in target.examples + target.negative_examples:
            if not example.reviewed or example.crop_preprocessing_version != IMAGE_PREPROCESSING_VERSION:
                continue
            record = existing.get(example.id)
            if (record is not None and record.crop_sha256 == example.sha256
                    and record.preprocessing_version == backend.preprocessing_version
                    and record.model_name == backend.name):
                continue
            path = Path(root)
            library = path if path.name == "object_library" else path / "object_library"
            crop_path = library / example.path
            if ".." in Path(example.path).parts or not crop_path.is_file():
                raise ValueError(f"example crop is unreadable: {example.path}")
            vector = normalize_vector(backend.embed_image(crop_path.read_bytes()))
            committed = write_embedding(root, target.id, example.id, EmbeddingRecord(
                model_name=backend.name,
                model_fingerprint=backend.fingerprint,
                preprocessing_version=backend.preprocessing_version,
                crop_sha256=example.sha256,
                vector=vector,
            ), expected_crop_sha256=example.sha256, expected_reviewed=True,
                expected_target_revision=target.revision)
            written += int(committed)
    return written


def _fallback_local_fingerprint(path: Path) -> str:
    from cvti.object_watch.runtime_config import _model_artifact_files
    files = _model_artifact_files(path)
    if not files:
        raise RuntimeError("SigLIP embedding backend unavailable: model artifacts are missing")
    from cvti.object_watch.runtime_config import _artifact_fingerprint
    return _artifact_fingerprint(sorted(files), IMAGE_PREPROCESSING_VERSION)


def _artifact_signature(files: list[Path]) -> tuple:
    return tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                 for path in files)
