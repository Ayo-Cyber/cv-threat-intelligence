"""Persistent object-watch target library.

The object library is deliberately plain JSON plus image crops. A target's
examples are customer-authored evidence; embeddings are model-versioned derived
artifacts that can be regenerated when the local embedding backend changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


VALID_CATEGORIES = {"product", "vehicle", "pallet", "ppe", "custom"}
VALID_REVIEW_STATES = {"draft", "active", "needs_reembed", "disabled"}
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class ObjectExample:
    id: str
    source: str
    path: str
    bbox: tuple[int, int, int, int]
    sha256: str
    reviewed: bool = True


@dataclass(frozen=True)
class EmbeddingRecord:
    model_name: str
    model_fingerprint: str
    preprocessing_version: int
    crop_sha256: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class ObjectTarget:
    id: str
    label: str
    category: str
    aliases: tuple[str, ...] = ()
    review_state: str = "draft"
    min_similarity: float = 0.72
    allowed_zone_ids: tuple[str, ...] = ()
    examples: tuple[ObjectExample, ...] = ()
    negative_examples: tuple[ObjectExample, ...] = ()


def _library(root: str | Path) -> Path:
    return Path(root) / "object_library"


def _targets_path(root: str | Path) -> Path:
    return _library(root) / "targets.json"


def _safe_id(value: str, kind: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe {kind}: {value!r}")
    return value


def _safe_object_id(value: str) -> str:
    try:
        return _safe_id(value, "object id")
    except ValueError as exc:
        raise ValueError(f"unsafe object id: {value!r}") from exc


def _validate_bbox(bbox: tuple[int, int, int, int] | list[int]) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError("bbox must have four integers")
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
    return (x1, y1, x2, y2)


def _example_from_dict(raw: dict[str, Any]) -> ObjectExample:
    return ObjectExample(
        id=_safe_id(str(raw.get("id", "")), "example id"),
        source=str(raw.get("source", "")).strip(),
        path=str(raw.get("path", "")).strip(),
        bbox=_validate_bbox(raw.get("bbox", ())),
        sha256=str(raw.get("sha256", "")).strip(),
        reviewed=bool(raw.get("reviewed", True)),
    )


def _target_from_dict(raw: dict[str, Any]) -> ObjectTarget:
    return ObjectTarget(
        id=str(raw.get("id", "")).strip(),
        label=str(raw.get("label", "")).strip(),
        category=str(raw.get("category", "")).strip(),
        aliases=tuple(str(v).strip() for v in raw.get("aliases", ()) if str(v).strip()),
        review_state=str(raw.get("review_state", "draft")).strip(),
        min_similarity=float(raw.get("min_similarity", 0.72)),
        allowed_zone_ids=tuple(
            str(v).strip() for v in raw.get("allowed_zone_ids", ()) if str(v).strip()
        ),
        examples=tuple(_example_from_dict(v) for v in raw.get("examples", ())),
        negative_examples=tuple(
            _example_from_dict(v) for v in raw.get("negative_examples", ())
        ),
    )


def _target_to_dict(target: ObjectTarget) -> dict[str, Any]:
    return {
        "id": target.id,
        "label": target.label,
        "category": target.category,
        "aliases": list(target.aliases),
        "review_state": target.review_state,
        "min_similarity": target.min_similarity,
        "allowed_zone_ids": list(target.allowed_zone_ids),
        "examples": [asdict(example) for example in target.examples],
        "negative_examples": [asdict(example) for example in target.negative_examples],
    }


def _validate_target(target: ObjectTarget) -> ObjectTarget:
    object_id = _safe_object_id(target.id)
    label = target.label.strip()
    if not label:
        raise ValueError("object label is required")
    if target.category not in VALID_CATEGORIES:
        raise ValueError(f"unknown object category: {target.category}")
    if target.review_state not in VALID_REVIEW_STATES:
        raise ValueError(f"unknown object review state: {target.review_state}")
    similarity = float(target.min_similarity)
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1")
    examples = tuple(target.examples)
    negatives = tuple(target.negative_examples)
    if target.review_state == "active" and not any(e.reviewed for e in examples):
        raise ValueError("active target requires at least one example")
    for example in examples + negatives:
        _safe_id(example.id, "example id")
        if not example.source.strip():
            raise ValueError("example source is required")
        if not example.path.strip() or ".." in Path(example.path).parts:
            raise ValueError("unsafe example path")
        if not re.fullmatch(r"[0-9a-f]{64}", example.sha256):
            raise ValueError("example sha256 must be lowercase sha256 hex")
        _validate_bbox(example.bbox)
    return ObjectTarget(
        id=object_id,
        label=label,
        category=target.category,
        aliases=tuple(v.strip() for v in target.aliases if v.strip()),
        review_state=target.review_state,
        min_similarity=similarity,
        allowed_zone_ids=tuple(v.strip() for v in target.allowed_zone_ids if v.strip()),
        examples=examples,
        negative_examples=negatives,
    )


def _read_doc(root: str | Path) -> dict[str, Any]:
    path = _targets_path(root)
    if not path.exists():
        return {"version": 1, "targets": []}
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid object target store: {path}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("targets"), list):
        raise ValueError(f"invalid object target store: {path}")
    return doc


def _write_doc(root: str | Path, doc: dict[str, Any]) -> None:
    library = _library(root)
    library.mkdir(parents=True, exist_ok=True)
    path = _targets_path(root)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_targets(root: str | Path) -> tuple[ObjectTarget, ...]:
    doc = _read_doc(root)
    targets = tuple(_validate_target(_target_from_dict(raw)) for raw in doc["targets"])
    ids = [target.id for target in targets]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate object target id")
    return targets


def save_target(root: str | Path, target: ObjectTarget) -> ObjectTarget:
    validated = _validate_target(target)
    existing = [t for t in load_targets(root) if t.id != validated.id]
    existing.append(validated)
    existing.sort(key=lambda row: row.id)
    _write_doc(root, {"version": 1, "targets": [_target_to_dict(t) for t in existing]})
    return validated


def add_example(
    root: str | Path,
    object_id: str,
    image_bytes: bytes,
    bbox: tuple[int, int, int, int],
    source: str,
    *,
    negative: bool = False,
) -> ObjectExample:
    object_id = _safe_object_id(object_id)
    if not bytes(image_bytes):
        raise ValueError("example image bytes are required")
    source = source.strip()
    if not source:
        raise ValueError("example source is required")
    bbox = _validate_bbox(bbox)
    targets = list(load_targets(root))
    index = next((i for i, t in enumerate(targets) if t.id == object_id), -1)
    if index < 0:
        raise ValueError(f"unknown object target: {object_id}")
    digest = hashlib.sha256(image_bytes).hexdigest()
    prefix = "neg" if negative else "ex"
    example_id = f"{prefix}-{digest[:12]}"
    rel_path = Path("examples") / object_id / f"{example_id}.jpg"
    target_path = _library(root) / rel_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(image_bytes)
    example = ObjectExample(
        id=example_id,
        source=source,
        path=rel_path.as_posix(),
        bbox=bbox,
        sha256=digest,
        reviewed=True,
    )
    target = targets[index]
    if negative:
        targets[index] = replace(
            target, negative_examples=target.negative_examples + (example,)
        )
    else:
        targets[index] = replace(target, examples=target.examples + (example,))
    _write_doc(root, {"version": 1, "targets": [_target_to_dict(_validate_target(t)) for t in targets]})
    return example


def _embedding_path(root: str | Path, model_fingerprint: str, object_id: str) -> Path:
    _safe_id(model_fingerprint, "model fingerprint")
    object_id = _safe_object_id(object_id)
    return _library(root) / "embeddings" / model_fingerprint / f"{object_id}.json"


def _record_to_dict(record: EmbeddingRecord) -> dict[str, Any]:
    return {
        "model_name": record.model_name,
        "model_fingerprint": record.model_fingerprint,
        "preprocessing_version": int(record.preprocessing_version),
        "crop_sha256": record.crop_sha256,
        "vector": list(record.vector),
    }


def _record_from_dict(raw: dict[str, Any]) -> EmbeddingRecord:
    return EmbeddingRecord(
        model_name=str(raw.get("model_name", "")).strip(),
        model_fingerprint=str(raw.get("model_fingerprint", "")).strip(),
        preprocessing_version=int(raw.get("preprocessing_version", 0)),
        crop_sha256=str(raw.get("crop_sha256", "")).strip(),
        vector=tuple(float(v) for v in raw.get("vector", ())),
    )


def write_embedding(
    root: str | Path,
    object_id: str,
    example_id: str,
    record: EmbeddingRecord,
) -> None:
    object_id = _safe_object_id(object_id)
    example_id = _safe_id(example_id, "example id")
    path = _embedding_path(root, record.model_fingerprint, object_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        doc = json.loads(path.read_text())
    else:
        doc = {"version": 1, "embeddings": {}}
    doc["embeddings"][example_id] = _record_to_dict(record)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_embeddings(
    root: str | Path,
    object_id: str,
    model_fingerprint: str,
) -> dict[str, EmbeddingRecord]:
    object_id = _safe_object_id(object_id)
    path = _embedding_path(root, model_fingerprint, object_id)
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid embedding store: {path}") from exc
    raw_embeddings = doc.get("embeddings")
    if not isinstance(raw_embeddings, dict):
        raise ValueError(f"invalid embedding store: {path}")
    records = {
        _safe_id(example_id, "example id"): _record_from_dict(raw)
        for example_id, raw in raw_embeddings.items()
    }
    examples = {
        example.id: example
        for target in load_targets(root) if target.id == object_id
        for example in target.examples + target.negative_examples
    }
    for example_id, record in records.items():
        example = examples.get(example_id)
        if example is not None and record.crop_sha256 != example.sha256:
            raise ValueError(f"crop hash mismatch for {object_id}:{example_id}")
    return records


def targets_needing_reembed(
    root: str | Path,
    model_fingerprint: str,
) -> list[ObjectTarget]:
    needing: list[ObjectTarget] = []
    for target in load_targets(root):
        examples = target.examples + target.negative_examples
        if not examples:
            continue
        records = load_embeddings(root, target.id, model_fingerprint)
        for example in examples:
            record = records.get(example.id)
            if record is None or record.crop_sha256 != example.sha256:
                needing.append(target)
                break
    return needing
