"""Persisted, local-only runtime configuration and readiness checks."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterator

from cvti.object_watch.locking import file_lock


@dataclass(frozen=True)
class ObjectWatchConfig:
    backend: str = "siglip"
    model_path: Path | None = None
    device: str = "cpu"
    proposal_provider: str = "none"
    world_weights: Path | None = None
    clip_weights: Path | None = None
    sample_fps: float = 1.0
    max_candidates: int = 24
    result_ttl_seconds: float = 2.0
    library_path: Path | None = None


@dataclass(frozen=True)
class Readiness:
    status: str
    reasons: tuple[str, ...]
    backend: str
    model_fingerprint: str | None
    dimensions: int | None = None
    structurally_available: bool = False
    executable_verified: bool = False

    @property
    def ready(self) -> bool:
        return self.status == "ready" and self.executable_verified


_FINGERPRINT_CACHE: dict[tuple, str] = {}
_FINGERPRINT_LOCK = threading.Lock()


def resolve_config(site_output_dir: Path) -> ObjectWatchConfig:
    site = Path(site_output_dir).resolve()
    library = site / "object_library"
    path = library / "runtime.json"
    raw = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid object watch runtime config: {path}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"invalid object watch runtime config: {path}")
    backend = str(raw.get("backend", "siglip")).strip().lower()
    if backend == "hash":
        raise ValueError("hash backend is test-only and cannot be resolved for production")
    if backend != "siglip":
        raise ValueError(f"unsupported object watch backend: {backend}")
    proposal_provider = str(raw.get("proposal_provider", "none")).strip().lower()
    if proposal_provider not in {"none", "yolo_world"}:
        raise ValueError(f"unsupported object proposal provider: {proposal_provider}")
    model_path = _local_path(raw.get("model_path"), library / "models" / "siglip", site)
    raw_max_candidates = raw.get("max_candidates", 24)
    if (isinstance(raw_max_candidates, bool)
            or not isinstance(raw_max_candidates, int)):
        raise ValueError("object watch max_candidates must be an integer")
    config = ObjectWatchConfig(
        backend=backend, model_path=model_path,
        device=str(raw.get("device", "cpu")).strip() or "cpu",
        proposal_provider=proposal_provider,
        world_weights=_optional_local_path(raw.get("world_weights"), site),
        clip_weights=_optional_local_path(raw.get("clip_weights"), site),
        sample_fps=float(raw.get("sample_fps", 1.0)),
        max_candidates=raw_max_candidates,
        result_ttl_seconds=float(raw.get("result_ttl_seconds", 2.0)),
        library_path=library,
    )
    if (not math.isfinite(config.sample_fps) or config.sample_fps <= 0
            or isinstance(config.max_candidates, bool) or config.max_candidates <= 0
            or not math.isfinite(config.result_ttl_seconds)
            or config.result_ttl_seconds <= 0):
        raise ValueError("object watch runtime limits must be positive")
    return config


def write_config(site_output_dir: Path, config: ObjectWatchConfig) -> Path:
    if config.backend.lower() == "hash":
        raise ValueError("hash backend is test-only and cannot be persisted for production")
    if config.proposal_provider not in {"none", "yolo_world"}:
        raise ValueError(f"unsupported object proposal provider: {config.proposal_provider}")
    library = Path(site_output_dir).resolve() / "object_library"
    library.mkdir(parents=True, exist_ok=True)
    path = library / "runtime.json"
    doc = asdict(replace(config, library_path=None))
    for key in ("model_path", "world_weights", "clip_weights", "library_path"):
        if doc.get(key) is not None:
            doc[key] = str(doc[key])
    fd, raw = tempfile.mkstemp(prefix=".runtime.json.", suffix=".tmp", dir=library)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(doc, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return path


def preflight(config: ObjectWatchConfig) -> Readiness:
    reasons: list[str] = []
    if config.backend.lower() != "siglip":
        reasons.append("production backend must be siglip")
    model_path = config.model_path
    files: list[Path] = []
    if model_path is None or not model_path.is_dir():
        reasons.append("local SigLIP model directory is missing")
    else:
        config_file = model_path / "config.json"
        processor = model_path / "preprocessor_config.json"
        weights = sorted(model_path.glob("*.safetensors")) + sorted(model_path.glob("pytorch_model*.bin"))
        if not config_file.is_file():
            reasons.append("SigLIP config.json is missing")
        if not processor.is_file():
            reasons.append("SigLIP preprocessor_config.json is missing")
        if not weights:
            reasons.append("SigLIP weight artifacts are missing")
        files = _model_artifact_files(model_path)
    dimensions = None
    if not reasons and model_path:
        dimensions, config_reason = _siglip_model_dimensions(model_path / "config.json")
        if config_reason:
            reasons.append(config_reason)
    structural_reasons = tuple(reasons)
    fingerprint = _artifact_fingerprint(files, preprocessing_version=1) if not reasons else None
    optional_missing = [
        label for label, value in (("world_weights", config.world_weights),
                                   ("clip_weights", config.clip_weights))
        if config.proposal_provider == "yolo_world" and (value is None or not value.is_file())
    ]
    if reasons:
        status = "unavailable"
    elif optional_missing:
        status = "degraded"
        reasons.extend(f"optional {label} is missing" for label in optional_missing)
    else:
        status = "structurally_available"
    return Readiness(status, tuple(reasons), config.backend, fingerprint, dimensions,
                     structurally_available=not structural_reasons, executable_verified=False)


def load_configured_backend(config: ObjectWatchConfig):
    readiness = preflight(config)
    if readiness.status == "unavailable":
        raise RuntimeError("object watch unavailable: " + "; ".join(readiness.reasons))
    from cvti.object_watch.embeddings import SiglipEmbeddingBackend

    return SiglipEmbeddingBackend(
        str(config.model_path), device=config.device,
        fingerprint=readiness.model_fingerprint,
        inference_lock_root=config.library_path,
    )


def configured_backend_metadata(config: ObjectWatchConfig):
    """Return activation-safe metadata without importing torch or loading weights."""
    readiness = preflight(config)
    if not readiness.model_fingerprint or readiness.dimensions is None:
        raise RuntimeError("object watch unavailable: " + "; ".join(readiness.reasons))
    from cvti.object_watch.embeddings import BackendMetadata
    return BackendMetadata(
        name=config.backend, fingerprint=readiness.model_fingerprint,
        preprocessing_version=1, dimensions=readiness.dimensions,
        structurally_available=readiness.structurally_available,
        executable_verified=readiness.executable_verified,
    )


@contextmanager
def advisory_inference_lock(root: str | Path, *, blocking: bool = True) -> Iterator[bool]:
    """Cross-process advisory lock; nonblocking mode yields ``False`` if busy."""
    library = Path(root)
    if library.name != "object_library":
        library = library / "object_library"
    with file_lock(library / ".inference.lock", blocking=blocking) as acquired:
        yield acquired


def _model_dimensions(config_path: Path) -> int | None:
    """Return the image feature width for a supported original SigLIP config."""
    dimensions, _ = _siglip_model_dimensions(config_path)
    return dimensions


def _siglip_model_dimensions(config_path: Path) -> tuple[int | None, str | None]:
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, "SigLIP config.json is not valid JSON"
    if not isinstance(raw, dict):
        return None, "SigLIP config.json must contain an object"
    model_type = raw.get("model_type")
    if model_type != "siglip":
        if model_type == "siglip2":
            return None, "SigLIP2 model configs are not supported"
        return None, "SigLIP config.json model_type must be 'siglip'"
    vision = raw.get("vision_config")
    if not isinstance(vision, dict):
        return None, "SigLIP config.json vision_config must contain an object"
    if vision.get("model_type") not in (None, "siglip_vision_model"):
        return None, "SigLIP vision_config model_type must be 'siglip_vision_model'"
    try:
        SiglipConfig = importlib.import_module("transformers").SiglipConfig
    except ImportError:
        return None, "transformers is required to validate SigLIP config.json"
    try:
        config = SiglipConfig.from_dict(raw)
        dimensions = config.vision_config.hidden_size
    except (AttributeError, TypeError, ValueError) as exc:
        return None, f"SigLIP config.json is invalid: {exc}"
    if (not isinstance(dimensions, int) or isinstance(dimensions, bool)
            or dimensions <= 0):
        return None, "SigLIP vision hidden_size must be a positive integer"
    return dimensions, None


def _model_artifact_files(model_path: Path) -> list[Path]:
    """Files which define image preprocessing and model output semantics."""
    names = ("config.json", "preprocessor_config.json", "processor_config.json")
    files = [model_path / name for name in names if (model_path / name).is_file()]
    files.extend(sorted(model_path.glob("*.safetensors")))
    files.extend(sorted(model_path.glob("pytorch_model*.bin")))
    return files


def _local_path(value, default: Path, base: Path | None = None) -> Path:
    raw = str(value) if value not in (None, "") else str(default)
    if "://" in raw or raw.lower().startswith(("http:/", "https:/")):
        raise ValueError("model paths must be local filesystem paths")
    path = Path(raw)
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.expanduser().resolve()


def _optional_local_path(value, base: Path) -> Path | None:
    return None if value in (None, "") else _local_path(value, Path("."), base)


def _artifact_fingerprint(files: list[Path], preprocessing_version: int) -> str:
    stats = tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in files)
    key = (preprocessing_version, stats)
    with _FINGERPRINT_LOCK:
        cached = _FINGERPRINT_CACHE.get(key)
        if cached is not None:
            return cached
    digest = hashlib.sha256(f"preprocess:{preprocessing_version}".encode())
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    after = tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in files)
    if after != stats:
        raise RuntimeError("model artifacts changed while fingerprinting")
    result = "siglip-" + digest.hexdigest()[:24]
    with _FINGERPRINT_LOCK:
        _FINGERPRINT_CACHE[key] = result
        if len(_FINGERPRINT_CACHE) > 32:
            _FINGERPRINT_CACHE.pop(next(iter(_FINGERPRINT_CACHE)))
    return result
