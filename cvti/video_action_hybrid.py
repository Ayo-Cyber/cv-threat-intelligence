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

# Per-signal bars, because one number cannot serve signals of very different
# quality. Measured 25 Sep over the KPI manifest's 170 normals + 162 theft
# clips (tools/video_action_sweep.py, runs/eval/video_action/scores.jsonl):
#
#   theft, at the shipped 0.05   20.6% of NORMAL clips raised a candidate,
#                                for 45.7% recall
#   theft, at 0.95                3.5% false positives, 28.4% recall
#   AUC                           0.659  (0.5 is a coin flip)
#
# An AUC of 0.66 means the score barely ranks theft above normal, so no
# threshold makes this signal good -- the curve only trades one failure for
# the other. It matters because theft was the sole source of every false
# positive in the 24 Sep scorecard, and each one costs a ~12s VLM
# verification on the pilot's 4-core box, starving the alerts that are real.
# 0.95 is the point where it stops flooding the queue; the honest option is
# to stop raising it at all and let concealment plus the object watchlist
# carry theft. A site can set either, and the rest of the signals -- violence
# especially, which row 9 measures at 97.2% -- keep the permissive bar that
# suits them.
SIGNAL_CONFIDENCE_THRESHOLDS: dict = {"theft_candidate": 0.95}


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
    signal_thresholds: dict | None = None,
) -> list[RawEvent]:
    events: list[RawEvent] = []
    for prediction in predictions:
        signal_type = classify_action_label(prediction.label)
        if signal_type is None:
            continue
        bar = (signal_thresholds or SIGNAL_CONFIDENCE_THRESHOLDS).get(
            signal_type, raw_confidence_threshold)
        if prediction.confidence < bar:
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
