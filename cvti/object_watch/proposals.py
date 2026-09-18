"""Offline-safe proposal helpers for reference object recognition."""

from __future__ import annotations

import importlib.util
import importlib
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cvti.object_watch.matcher import ObjectCandidate


@dataclass(frozen=True)
class ProposalReadiness:
    ready: bool
    reasons: tuple[str, ...] = ()


def yolo_world_preflight(world_weights: str | Path | None,
                         clip_weights: str | Path | None) -> ProposalReadiness:
    """Check every local artifact before any third-party model is constructed."""
    reasons: list[str] = []
    world = Path(world_weights).expanduser().resolve() if world_weights else None
    clip = Path(clip_weights).expanduser().resolve() if clip_weights else None
    if importlib.util.find_spec("ultralytics") is None:
        reasons.append("ultralytics is not installed")
    if importlib.util.find_spec("clip") is None:
        reasons.append("OpenAI CLIP is not installed")
    if world is None or not world.is_file():
        reasons.append("local YOLO-World weights are missing")
    elif not zipfile.is_zipfile(world):
        reasons.append("YOLO-World weights are not a valid local torch archive")
    if clip is None or not clip.is_file():
        reasons.append("local CLIP text weights are missing")
    return ProposalReadiness(not reasons, tuple(reasons))


class YoloWorldProposalProvider:
    """YOLO-World proposals using only explicitly provisioned local assets."""

    def __init__(self, world_weights: str | Path, clip_weights: str | Path, *,
                 device: str = "cpu", model_factory=None) -> None:
        readiness = yolo_world_preflight(world_weights, clip_weights)
        if not readiness.ready:
            raise RuntimeError("YOLO-World unavailable: " + "; ".join(readiness.reasons))
        self.device = device
        world_path = str(Path(world_weights).expanduser().resolve())
        clip_path = str(Path(clip_weights).expanduser().resolve())
        self._model = (model_factory(world_path, clip_path) if model_factory is not None
                       else _OfflineYoloWorldAdapter(world_path, clip_path, device))
        self._phrases: tuple[str, ...] = ()

    def propose(self, frame: Any, phrases: Iterable[str], *,
                limit_candidates: int | None = None) -> tuple[ObjectCandidate, ...]:
        wanted = _validated_phrases(phrases)
        if not wanted:
            return ()
        if (limit_candidates is not None
                and (isinstance(limit_candidates, bool)
                     or not isinstance(limit_candidates, int) or limit_candidates <= 0)):
            raise ValueError("limit_candidates must be a positive integer")
        if wanted != self._phrases:
            configure = getattr(self._model, "configure_classes", None)
            if not callable(configure):
                raise RuntimeError("YOLO-World model factory must provide configure_classes()")
            configure(wanted)
            self._phrases = wanted
        results = self._model.predict(frame, device=self.device, verbose=False)
        rows = sorted(_result_candidates(results, wanted),
                      key=lambda item: item.confidence, reverse=True)
        if limit_candidates is not None:
            rows = rows[:limit_candidates]
        return tuple(rows)


class _OfflineYoloWorldAdapter:
    """Patch local CLIP text features into Ultralytics without its auto-installer."""

    _TEXT_BATCH_SIZE = 64

    def __init__(self, world_path: str, clip_path: str, device: str) -> None:
        # Importing ultralytics.nn.text_model triggers an attempted `pip install
        # clip`; only public top-level APIs and the separately installed CLIP
        # package are used here.
        try:
            YOLOWorld = getattr(importlib.import_module("ultralytics"), "YOLOWorld")
            clip = importlib.import_module("clip")
            torch = importlib.import_module("torch")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(f"YOLO-World offline dependencies are unavailable: {exc}") from exc

        try:
            self._world = YOLOWorld(world_path)
            self._world.to(device)
        except Exception as exc:
            raise RuntimeError(f"failed to load local YOLO-World weights: {exc}") from exc
        try:
            self._text_model, _ = clip.load(clip_path, device=device, jit=False)
        except Exception as exc:
            raise RuntimeError(f"failed to load local CLIP text weights: {exc}") from exc
        self._clip = clip
        self._torch = torch
        self._device = device
        self._world.model.eval()
        self._text_model.eval()
        self._validate_world_head()

    def _validate_world_head(self):
        try:
            head = self._world.model.model[-1]
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "incompatible Ultralytics YOLO-World model: detection head is unavailable"
            ) from exc
        if not hasattr(head, "nc") or not hasattr(self._world.model, "txt_feats"):
            raise RuntimeError(
                "incompatible Ultralytics YOLO-World model: expected nc and txt_feats; "
                "install the validated Ultralytics version"
            )
        return head

    def configure_classes(self, phrases: tuple[str, ...]) -> None:
        chunks = []
        try:
            with self._torch.no_grad():
                for start in range(0, len(phrases), self._TEXT_BATCH_SIZE):
                    texts = list(phrases[start:start + self._TEXT_BATCH_SIZE])
                    tokens = self._clip.tokenize(texts, truncate=True).to(self._device)
                    chunks.append(self._text_model.encode_text(tokens).float())
                features = self._torch.cat(chunks, dim=0)
                if features.ndim != 2 or features.shape[0] != len(phrases) or features.shape[1] <= 0:
                    raise ValueError(f"unexpected CLIP text feature shape {tuple(features.shape)}")
                norms = features.norm(dim=-1, keepdim=True)
                if not bool(self._torch.isfinite(features).all()) or not bool((norms > 0).all()):
                    raise ValueError("CLIP produced non-finite or zero text features")
                features = (features / norms).unsqueeze(0)
        except Exception as exc:
            raise RuntimeError(f"failed to encode YOLO-World descriptions locally: {exc}") from exc

        head = self._validate_world_head()
        names = _model_names(self._world.model, phrases)
        self._world.model.txt_feats = features
        head.nc = len(phrases)
        self._world.model.names = names
        predictor = getattr(self._world, "predictor", None)
        predictor_model = getattr(predictor, "model", None)
        if predictor_model is not None:
            predictor_model.names = names

    def predict(self, frame: Any, **kwargs):
        return self._world.predict(frame, **kwargs)


def _model_names(model: Any, phrases: tuple[str, ...]):
    current = getattr(model, "names", None)
    return list(phrases) if isinstance(current, list) else dict(enumerate(phrases))


def _validated_phrases(phrases: Iterable[str]) -> tuple[str, ...]:
    if isinstance(phrases, (str, bytes)):
        raise ValueError("YOLO-World descriptions must be an iterable of strings")
    result: list[str] = []
    seen: set[str] = set()
    try:
        values = iter(phrases)
    except TypeError as exc:
        raise ValueError("YOLO-World descriptions must be an iterable of strings") from exc
    for value in values:
        if not isinstance(value, str):
            raise ValueError("YOLO-World descriptions must contain only strings")
        description = value.strip()
        if not description:
            continue
        if len(description) > 512:
            raise ValueError("YOLO-World descriptions must be at most 512 characters")
        if description not in seen:
            seen.add(description)
            result.append(description)
    return tuple(result)


def _as_list(value: Any) -> list[Any]:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    converted = tolist() if callable(tolist) else value
    if not isinstance(converted, (list, tuple)):
        raise TypeError("model output is not a sequence")
    return list(converted)


def _result_candidates(results: Any, phrases: tuple[str, ...]) -> list[ObjectCandidate]:
    candidates: list[ObjectCandidate] = []
    if results is None:
        return candidates
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        try:
            coordinates = _as_list(boxes.xyxy)
            confidences = _as_list(boxes.conf)
            classes = _as_list(boxes.cls)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("unexpected Ultralytics Results.boxes output") from exc
        if not (len(coordinates) == len(confidences) == len(classes)):
            raise RuntimeError("incompatible Ultralytics Results.boxes output shapes")
        for raw_box, raw_confidence, raw_class in zip(coordinates, confidences, classes):
            try:
                values = tuple(float(value) for value in raw_box)
                confidence = float(raw_confidence)
                class_value = float(raw_class)
            except (TypeError, ValueError):
                continue
            if (len(values) != 4 or not all(math.isfinite(value) for value in values)
                    or not math.isfinite(confidence) or not math.isfinite(class_value)
                    or not class_value.is_integer()):
                continue
            class_id = int(class_value)
            if class_id < 0 or class_id >= len(phrases):
                continue
            bbox = (int(round(values[0])), int(round(values[1])),
                    int(round(values[2])), int(round(values[3])))
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            candidates.append(ObjectCandidate(bbox, phrases[class_id], confidence))
    return candidates


def merge_proposals(candidates: Iterable[Any], proposals: Iterable[Any], *,
                    max_candidates: int, iou_threshold: float = 0.55) -> tuple[ObjectCandidate, ...]:
    """Confidence-order, class-agnostic NMS with one global cap."""
    rows = sorted((_candidate(item) for item in (*tuple(candidates), *tuple(proposals))),
                  key=lambda item: item.confidence, reverse=True)
    kept: list[ObjectCandidate] = []
    for row in rows:
        if all(_iou(row.bbox, old.bbox) < iou_threshold for old in kept):
            kept.append(row)
            if len(kept) >= max_candidates:
                break
    return tuple(kept)


def _candidate(value: Any) -> ObjectCandidate:
    if isinstance(value, ObjectCandidate):
        return value
    if isinstance(value, dict):
        bbox = value.get("bbox", value.get("box"))
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            raise ValueError("proposal bbox must contain four coordinates")
        label = value.get("class_name", value.get("label_hint", value.get("label", value.get("phrase", ""))))
        confidence = value.get("confidence", value.get("score", 0.0))
        box = tuple(int(round(v)) for v in bbox)
        return ObjectCandidate((box[0], box[1], box[2], box[3]), str(label), float(confidence),
                               value.get("track_id"), value.get("zone_id"))
    bbox = getattr(value, "bbox")
    if len(bbox) != 4:
        raise ValueError("proposal bbox must contain four coordinates")
    label = getattr(value, "class_name", getattr(value, "label_hint", getattr(value, "label", "")))
    box = tuple(int(round(v)) for v in bbox)
    return ObjectCandidate((box[0], box[1], box[2], box[3]), str(label),
                           float(getattr(value, "confidence", 0.0)),
                           getattr(value, "track_id", None), getattr(value, "zone_id", None))


def _iou(left, right) -> float:
    x1, y1, x2, y2 = left
    a1, b1, a2, b2 = right
    area = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    union = max(0, x2 - x1) * max(0, y2 - y1) + max(0, a2 - a1) * max(0, b2 - b1) - area
    return area / union if union else 0.0
