"""Hybrid bridge from video action classifiers into the threat rule engine.

VideoMAE/X3D outputs are treated as weak temporal evidence. They do not decide
alerts directly; they become `RawEvent`s that configs and the VLM can use.
"""

from __future__ import annotations

from typing import Iterable

from cvti.contracts import RawEvent
from cvti.video_action_model import VideoActionPrediction


DEFAULT_EVIDENCE_WEIGHT = 0.35
DEFAULT_RAW_CONFIDENCE_THRESHOLD = 0.05


_SIGNAL_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "violence_candidate",
        (
            "punching",
            "boxing",
            "wrestling",
            "kickboxing",
            "martial arts",
            "sword fighting",
            "fencing",
        ),
    ),
    (
        "panic_running_candidate",
        (
            "running",
            "jogging",
            "parkour",
        ),
    ),
    (
        "weapon_handling_candidate",
        (
            "sharpening knives",
            "archery",
            "shooting",
        ),
    ),
    (
        # Emitted by a CamNuvem-fine-tuned model whose classes are theft/normal
        # (the "normal" label matches nothing here, so it's correctly ignored).
        "theft_candidate",
        (
            "theft",
            "robbery",
            "shoplift",
            "stealing",
            "steal",
            "burglary",
        ),
    ),
)


def predictions_to_events(
    predictions: Iterable[VideoActionPrediction],
    *,
    backend: str,
    model_name: str,
    window_name: str,
    sampled_frame_indices: list[int],
    timestamp: float = 0.0,
    evidence_weight: float = DEFAULT_EVIDENCE_WEIGHT,
    raw_confidence_threshold: float = DEFAULT_RAW_CONFIDENCE_THRESHOLD,
) -> list[RawEvent]:
    events: list[RawEvent] = []
    for prediction in predictions:
        signal_type = classify_action_label(prediction.label)
        if signal_type is None:
            continue
        if prediction.confidence < raw_confidence_threshold:
            continue

        adjusted_confidence = prediction.confidence * evidence_weight
        level = _level_from_adjusted_confidence(adjusted_confidence)
        events.append(
            RawEvent(
                detector="video_action",
                active=True,
                title=f"VIDEO ACTION: {prediction.label}",
                level=level,
                timestamp=timestamp,
                extra={
                    "signal_type": signal_type,
                    "label": prediction.label,
                    "rank": prediction.rank,
                    "raw_confidence": round(prediction.confidence, 6),
                    "evidence_weight": evidence_weight,
                    "adjusted_confidence": round(adjusted_confidence, 6),
                    "backend": backend,
                    "model": model_name,
                    "window": window_name,
                    "sampled_frame_indices": sampled_frame_indices,
                },
            )
        )
    return events


def classify_action_label(label: str) -> str | None:
    normalized = label.lower()
    for signal_type, needles in _SIGNAL_LABELS:
        if any(needle in normalized for needle in needles):
            return signal_type
    return None


def _level_from_adjusted_confidence(confidence: float) -> str:
    if confidence >= 0.20:
        return "high"
    if confidence >= 0.10:
        return "medium"
    return "low"
