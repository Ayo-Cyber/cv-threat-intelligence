"""Versioned local object library with serialized, atomic mutations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from cvti.object_watch.images import IMAGE_PREPROCESSING_VERSION, decode_crop_encode
from cvti.object_watch.locking import file_lock


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
    reviewed: bool = False
    bbox_format: str = "legacy"
    crop_preprocessing_version: int = 0
    source_sha256: str = ""
    source_width: int = 0
    source_height: int = 0


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
    grounding_description: str = ""
    revision: int = 0


@dataclass(frozen=True)
class TargetReadiness:
    ready: bool
    status: str
    reasons: tuple[str, ...]
    target_revision: int
    library_revision: int
    model_fingerprint: str


def _library(root: str | Path) -> Path:
    path = Path(root)
    return path if path.name == "object_library" else path / "object_library"


def _safe_id(value: str, kind: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe {kind}: {value!r}")
    return value


def _safe_object_id(value: str) -> str:
    try:
        return _safe_id(value, "object id")
    except ValueError as exc:
        raise ValueError(f"unsafe object id: {value!r}") from exc


@contextmanager
def library_lock(root: str | Path) -> Iterator[None]:
    """Serialize a coherent library snapshot or mutation across processes."""
    library = _library(root)
    with file_lock(library / ".library.lock") as acquired:
        if not acquired:
            raise RuntimeError("object library mutation lock timed out")
        yield


_mutation_lock = library_lock


def _targets_path(root: str | Path) -> Path:
    return _library(root) / "targets.json"


def _read_doc(root: str | Path) -> dict[str, Any]:
    path = _targets_path(root)
    if not path.exists():
        return {"version": 2, "revision": 0, "targets": []}
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid object target store: {path}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("targets"), list):
        raise ValueError(f"invalid object target store: {path}")
    doc.setdefault("revision", 0)
    return doc


def _atomic_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _example_from_dict(raw: dict[str, Any]) -> ObjectExample:
    bbox = raw.get("bbox", ())
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must have four integers")
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return ObjectExample(
        id=_safe_id(str(raw.get("id", "")), "example id"),
        source=str(raw.get("source", "")).strip(),
        path=str(raw.get("path", "")).strip(),
        bbox=(x1, y1, x2, y2),
        sha256=str(raw.get("sha256", "")).strip(),
        reviewed=bool(raw.get("reviewed", False)),
        bbox_format=str(raw.get("bbox_format", "legacy")),
        crop_preprocessing_version=int(raw.get("crop_preprocessing_version", 0)),
        source_sha256=str(raw.get("source_sha256", "")),
        source_width=int(raw.get("source_width", 0)),
        source_height=int(raw.get("source_height", 0)),
    )


def _target_from_dict(raw: dict[str, Any]) -> ObjectTarget:
    return ObjectTarget(
        id=str(raw.get("id", "")).strip(),
        label=str(raw.get("label", "")).strip(),
        category=str(raw.get("category", "")).strip(),
        aliases=tuple(str(v).strip() for v in raw.get("aliases", ()) if str(v).strip()),
        review_state=str(raw.get("review_state", "draft")).strip(),
        min_similarity=float(raw.get("min_similarity", 0.72)),
        allowed_zone_ids=tuple(str(v).strip() for v in raw.get("allowed_zone_ids", ()) if str(v).strip()),
        examples=tuple(_example_from_dict(v) for v in raw.get("examples", ())),
        negative_examples=tuple(_example_from_dict(v) for v in raw.get("negative_examples", ())),
        grounding_description=str(raw.get("grounding_description", "")).strip(),
        revision=int(raw.get("revision", 0)),
    )


def _target_to_dict(target: ObjectTarget) -> dict[str, Any]:
    row = asdict(target)
    row["aliases"] = list(target.aliases)
    row["allowed_zone_ids"] = list(target.allowed_zone_ids)
    row["examples"] = [asdict(example) for example in target.examples]
    row["negative_examples"] = [asdict(example) for example in target.negative_examples]
    return row


def _validate_target(target: ObjectTarget) -> ObjectTarget:
    object_id = _safe_object_id(target.id)
    if not target.label.strip():
        raise ValueError("object label is required")
    if target.category not in VALID_CATEGORIES:
        raise ValueError(f"unknown object category: {target.category}")
    if target.review_state not in VALID_REVIEW_STATES:
        raise ValueError(f"unknown object review state: {target.review_state}")
    if not math.isfinite(float(target.min_similarity)) or not 0 <= target.min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1")
    for example in target.examples + target.negative_examples:
        _safe_id(example.id, "example id")
        example_path = Path(example.path)
        if (not example.source or not example.path or example_path.is_absolute()
                or ".." in example_path.parts or len(example_path.parts) < 3
                or example_path.parts[:2] != ("examples", object_id)):
            raise ValueError("unsafe example metadata")
        if not re.fullmatch(r"[0-9a-f]{64}", example.sha256):
            raise ValueError("example sha256 must be lowercase sha256 hex")
        x1, y1, x2, y2 = example.bbox
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("example bbox must have bounded positive area")
        if example.crop_preprocessing_version == IMAGE_PREPROCESSING_VERSION:
            if (example.source_width <= 0 or example.source_height <= 0
                    or x2 > example.source_width or y2 > example.source_height):
                raise ValueError("canonical example bbox exceeds source dimensions")
    return replace(
        target, id=object_id, label=target.label.strip(),
        grounding_description=target.grounding_description.strip(),
        aliases=tuple(v.strip() for v in target.aliases if v.strip()),
        allowed_zone_ids=tuple(v.strip() for v in target.allowed_zone_ids if v.strip()),
    )


def load_targets(root: str | Path) -> tuple[ObjectTarget, ...]:
    targets = tuple(_validate_target(_target_from_dict(row)) for row in _read_doc(root)["targets"])
    if len({target.id for target in targets}) != len(targets):
        raise ValueError("duplicate object target id")
    return targets


def library_revision(root: str | Path) -> int:
    return int(_read_doc(root).get("revision", 0))


def _write_targets(root: str | Path, targets: list[ObjectTarget], revision: int) -> None:
    targets.sort(key=lambda item: item.id)
    _atomic_json(_targets_path(root), {
        "version": 2, "revision": revision,
        "targets": [_target_to_dict(item) for item in targets],
    })


def save_target(root: str | Path, target: ObjectTarget) -> ObjectTarget:
    candidate = _validate_target(target)
    with _mutation_lock(root):
        doc = _read_doc(root)
        targets = [_validate_target(_target_from_dict(row)) for row in doc["targets"]]
        old = next((item for item in targets if item.id == candidate.id), None)
        if candidate.review_state == "active" and (old is None or old.review_state != "active"):
            raise ValueError("use activate_target after embedding readiness succeeds")
        old_revision = old.revision if old is not None else 0
        if old is not None:
            retained = {item.id for item in candidate.examples + candidate.negative_examples}
            historical = {item.id for item in old.examples + old.negative_examples}
            if not historical.issubset(retained):
                raise ValueError("historical object examples cannot be deleted")
        recognition_changed = old is None or replace(candidate, revision=old_revision) != old
        state = candidate.review_state
        if (old is not None and old.review_state == "active" and recognition_changed
                and state == "active"):
            state = "draft"
        saved = replace(candidate, revision=old_revision + (1 if recognition_changed else 0), review_state=state)
        targets = [item for item in targets if item.id != saved.id] + [saved]
        revision = int(doc.get("revision", 0)) + (1 if recognition_changed else 0)
        _write_targets(root, targets, revision)
        return saved


def add_example(
    root: str | Path,
    object_id: str,
    image_bytes: bytes,
    bbox: tuple[int, int, int, int] | tuple[float, float, float, float],
    source: str,
    *,
    negative: bool = False,
    bbox_format: str = "legacy",
    reviewed: bool = False,
) -> ObjectExample:
    object_id = _safe_object_id(object_id)
    source = str(source).strip()
    if not source:
        raise ValueError("example source is required")
    canonical = decode_crop_encode(image_bytes, bbox, bbox_format)
    digest = hashlib.sha256(canonical.png_bytes).hexdigest()
    prefix = "neg" if negative else "ex"
    example = ObjectExample(
        id=f"{prefix}-{digest[:12]}", source=source,
        path=(Path("examples") / object_id / f"{prefix}-{digest[:12]}.png").as_posix(),
        bbox=canonical.pixel_bbox, sha256=digest, reviewed=bool(reviewed),
        bbox_format="pixel_xyxy",
        crop_preprocessing_version=canonical.preprocessing_version,
        source_sha256=canonical.source_sha256,
        source_width=canonical.source_width, source_height=canonical.source_height,
    )
    with _mutation_lock(root):
        doc = _read_doc(root)
        targets = [_validate_target(_target_from_dict(row)) for row in doc["targets"]]
        index = next((i for i, item in enumerate(targets) if item.id == object_id), -1)
        if index < 0:
            raise ValueError(f"unknown object target: {object_id}")
        path = _library(root) / example.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            _atomic_bytes(path, canonical.png_bytes)
        target = targets[index]
        sequence = target.negative_examples if negative else target.examples
        sequence = tuple(item for item in sequence if item.id != example.id) + (example,)
        targets[index] = replace(
            target,
            negative_examples=sequence if negative else target.negative_examples,
            examples=target.examples if negative else sequence,
            review_state="draft" if target.review_state == "active" else target.review_state,
            revision=target.revision + 1,
        )
        _write_targets(root, targets, int(doc.get("revision", 0)) + 1)
    return example


def review_example(root: str | Path, object_id: str, example_id: str, *, reviewed: bool = True) -> ObjectExample:
    object_id, example_id = _safe_object_id(object_id), _safe_id(example_id, "example id")
    with _mutation_lock(root):
        doc = _read_doc(root)
        targets = [_validate_target(_target_from_dict(row)) for row in doc["targets"]]
        pos = next((i for i, item in enumerate(targets) if item.id == object_id), -1)
        if pos < 0:
            raise ValueError(f"unknown object target: {object_id}")
        target = targets[pos]
        found: ObjectExample | None = None
        positives, negatives = list(target.examples), list(target.negative_examples)
        for sequence in (positives, negatives):
            for index, item in enumerate(sequence):
                if item.id == example_id:
                    found = replace(item, reviewed=bool(reviewed))
                    sequence[index] = found
        if found is None:
            raise ValueError(f"unknown object example: {example_id}")
        targets[pos] = replace(target, examples=tuple(positives), negative_examples=tuple(negatives),
                               review_state="draft" if target.review_state == "active" else target.review_state,
                               revision=target.revision + 1)
        _write_targets(root, targets, int(doc.get("revision", 0)) + 1)
        return found


def deactivate_target(root: str | Path, object_id: str) -> ObjectTarget:
    object_id = _safe_object_id(object_id)
    with _mutation_lock(root):
        doc = _read_doc(root)
        targets = [_validate_target(_target_from_dict(row)) for row in doc["targets"]]
        pos = next((i for i, item in enumerate(targets) if item.id == object_id), -1)
        if pos < 0:
            raise ValueError(f"unknown object target: {object_id}")
        targets[pos] = replace(targets[pos], review_state="disabled", revision=targets[pos].revision + 1)
        _write_targets(root, targets, int(doc.get("revision", 0)) + 1)
        return targets[pos]


def _embedding_path(root: str | Path, fingerprint: str, object_id: str) -> Path:
    _safe_id(fingerprint, "model fingerprint")
    return _library(root) / "embeddings" / fingerprint / f"{_safe_object_id(object_id)}.json"


def write_embedding(
    root: str | Path, object_id: str, example_id: str, record: EmbeddingRecord, *,
    expected_crop_sha256: str | None = None,
    expected_reviewed: bool | None = None,
    expected_target_revision: int | None = None,
    expected_library_revision: int | None = None,
) -> bool:
    """Conditionally commit one embedding and publish a new library revision.

    Expected values are optional for compatibility. Callers performing inference
    should provide them so stale results are rejected while holding the mutation
    lock. The return value is false for an idempotent or stale commit.
    """
    object_id, example_id = _safe_object_id(object_id), _safe_id(example_id, "example id")
    _validate_record(record)
    path = _embedding_path(root, record.model_fingerprint, object_id)
    with _mutation_lock(root):
        targets_doc = _read_doc(root)
        if (expected_library_revision is not None
                and int(targets_doc.get("revision", 0)) != expected_library_revision):
            return False
        targets = [_validate_target(_target_from_dict(row)) for row in targets_doc["targets"]]
        target = next((item for item in targets if item.id == object_id), None)
        if expected_target_revision is not None and (
                target is None or target.revision != expected_target_revision):
            return False
        examples = () if target is None else target.examples + target.negative_examples
        example = next((item for item in examples if item.id == example_id), None)
        if expected_crop_sha256 is not None and (
                example is None or example.sha256 != expected_crop_sha256
                or record.crop_sha256 != expected_crop_sha256):
            return False
        if expected_reviewed is not None and (
                example is None or example.reviewed is not expected_reviewed):
            return False
        if path.exists():
            try:
                doc = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid embedding store: {path}") from exc
        else:
            doc = {"version": 2, "embeddings": {}}
        row = asdict(record)
        row["vector"] = list(record.vector)
        if doc.setdefault("embeddings", {}).get(example_id) == row:
            return False
        doc["embeddings"][example_id] = row
        _atomic_json(path, doc)
        _write_targets(root, targets, int(targets_doc.get("revision", 0)) + 1)
        return True


def _validate_record(record: EmbeddingRecord) -> None:
    if not record.vector or not all(math.isfinite(float(value)) for value in record.vector):
        raise ValueError("embedding vector must contain finite values")
    if math.sqrt(sum(float(value) ** 2 for value in record.vector)) <= 0:
        raise ValueError("embedding vector must be nonzero")
    if record.preprocessing_version <= 0:
        raise ValueError("embedding preprocessing version must be positive")


def load_embeddings(root: str | Path, object_id: str, model_fingerprint: str) -> dict[str, EmbeddingRecord]:
    path = _embedding_path(root, model_fingerprint, object_id)
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid embedding store: {path}") from exc
    if not isinstance(doc.get("embeddings"), dict):
        raise ValueError(f"invalid embedding store: {path}")
    records = {
        _safe_id(key, "example id"): EmbeddingRecord(
            model_name=str(raw.get("model_name", "")),
            model_fingerprint=str(raw.get("model_fingerprint", "")),
            preprocessing_version=int(raw.get("preprocessing_version", 0)),
            crop_sha256=str(raw.get("crop_sha256", "")),
            vector=tuple(float(value) for value in raw.get("vector", ())),
        ) for key, raw in doc["embeddings"].items()
    }
    examples = {example.id: example for target in load_targets(root) if target.id == object_id
                for example in target.examples + target.negative_examples}
    for key, record in records.items():
        _validate_record(record)
        if record.model_fingerprint != model_fingerprint:
            raise ValueError(f"embedding fingerprint mismatch for {object_id}:{key}")
        if key in examples and record.crop_sha256 != examples[key].sha256:
            raise ValueError(f"crop hash mismatch for {object_id}:{key}")
    return records


def target_readiness(root: str | Path, target: ObjectTarget | str, backend: Any) -> TargetReadiness:
    if isinstance(target, str):
        object_id = _safe_object_id(target)
        found = next((item for item in load_targets(root) if item.id == object_id), None)
        if found is None:
            raise ValueError(f"unknown object target: {object_id}")
        target_obj = found
    else:
        target_obj = target
    fingerprint = str(backend.fingerprint)
    reasons: list[str] = []
    reviewed = [item for item in target_obj.examples if item.reviewed]
    reviewed_all = reviewed + [item for item in target_obj.negative_examples if item.reviewed]
    if not reviewed:
        reasons.append("no reviewed positive examples")
    if any(item.crop_preprocessing_version != IMAGE_PREPROCESSING_VERSION for item in reviewed_all):
        reasons.append("reviewed examples require canonical crop preprocessing")
    try:
        records = load_embeddings(root, target_obj.id, fingerprint)
    except ValueError as exc:
        records = {}
        reasons.append(f"invalid embedding store: {exc}")
    dimensions: set[int] = set()
    expected_dimensions = getattr(backend, "dimensions", None)
    for example in reviewed_all:
        record = records.get(example.id)
        if record is None:
            reasons.append(f"missing embedding for {example.id}")
            continue
        dimensions.add(len(record.vector))
        if expected_dimensions is not None and len(record.vector) != int(expected_dimensions):
            reasons.append(f"embedding dimension mismatch for {example.id}")
        if (record.crop_sha256 != example.sha256
                or record.model_fingerprint != fingerprint
                or record.preprocessing_version != int(backend.preprocessing_version)
                or record.model_name != str(backend.name)):
            reasons.append(f"incompatible embedding for {example.id}")
    if len(dimensions) > 1:
        reasons.append("embedding dimensions do not match")
    return TargetReadiness(not reasons, "ready" if not reasons else "unavailable", tuple(reasons),
                           target_obj.revision, library_revision(root), fingerprint)


def activate_target(root: str | Path, object_id: str, backend: Any) -> ObjectTarget:
    object_id = _safe_object_id(object_id)
    with _mutation_lock(root):
        doc = _read_doc(root)
        targets = [_validate_target(_target_from_dict(row)) for row in doc["targets"]]
        pos = next((i for i, item in enumerate(targets) if item.id == object_id), -1)
        if pos < 0:
            raise ValueError(f"unknown object target: {object_id}")
        readiness = target_readiness(root, targets[pos], backend)
        if not readiness.ready:
            raise ValueError("target is not ready: " + "; ".join(readiness.reasons))
        targets[pos] = replace(targets[pos], review_state="active",
                               revision=targets[pos].revision + 1)
        _write_targets(root, targets, int(doc.get("revision", 0)) + 1)
        return targets[pos]


def targets_needing_reembed(root: str | Path, model_fingerprint: str) -> list[ObjectTarget]:
    needing = []
    for target in load_targets(root):
        examples = [item for item in target.examples + target.negative_examples if item.reviewed]
        records = load_embeddings(root, target.id, model_fingerprint)
        if any(item.crop_preprocessing_version != IMAGE_PREPROCESSING_VERSION
               or item.id not in records or records[item.id].crop_sha256 != item.sha256
               for item in examples):
            needing.append(target)
    return needing
