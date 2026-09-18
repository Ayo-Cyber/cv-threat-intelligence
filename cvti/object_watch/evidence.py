"""Bounded visual evidence for reference-aware object-watch verification."""

from __future__ import annotations

import io
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from cvti.object_watch.images import decode_crop_encode, encode_rgb_array


def build_object_watch_evidence(
    observation_frame: Any,
    bbox: Sequence[float | int],
    reference_crop_bytes: bytes,
    label: str,
    *,
    include_context: bool = True,
    max_panel_dimension: int = 1280,
) -> list[np.ndarray]:
    """Return optional BGR context followed by one labelled comparison panel.

    The comparison is deliberately the final (and sufficient) image so a gate
    configured with ``max_frames=1`` still receives both the observed candidate
    and enrolled reference. Inputs are decoded through the canonical bounded
    object-watch image helpers; no reference path or source metadata is exposed.
    """
    if isinstance(max_panel_dimension, bool) or max_panel_dimension < 320:
        raise ValueError("max_panel_dimension must be at least 320")

    # Round-trip through the canonical image boundary to enforce dtype, source
    # dimensions, byte limits, and strict bbox bounds in one shared place.
    observation_bytes = encode_rgb_array(observation_frame, colour_space="bgr")
    candidate = decode_crop_encode(observation_bytes, bbox, "pixel_xyxy")
    reference = decode_crop_encode(reference_crop_bytes, (0, 0, 1, 1), "legacy")

    candidate_image = Image.open(io.BytesIO(candidate.png_bytes)).convert("RGB")
    reference_image = Image.open(io.BytesIO(reference.png_bytes)).convert("RGB")
    panel = _comparison_panel(
        candidate_image,
        reference_image,
        _safe_label(label),
        max_panel_dimension,
    )
    panel_bgr = np.asarray(panel, dtype=np.uint8)[:, :, ::-1].copy()

    if not include_context:
        return [panel_bgr]
    context = np.asarray(observation_frame)
    # Return owned evidence so later drawing on the live observation cannot
    # mutate what the gate sees.
    return [np.ascontiguousarray(context.copy()), panel_bgr]


def _safe_label(label: str) -> str:
    # Labels are display text, never paths. Flatten control characters and cap
    # their size so an accidental filename or payload cannot dominate a panel.
    return " ".join(str(label).split())[:80] or "unknown object"


def _comparison_panel(
    candidate: Image.Image,
    reference: Image.Image,
    label: str,
    max_dimension: int,
) -> Image.Image:
    header = 34
    gap = 8
    tile_width = max(1, (max_dimension - gap) // 2)
    tile_height = max(1, max_dimension - header)
    candidate = _fit(candidate, tile_width, tile_height)
    reference = _fit(reference, tile_width, tile_height)
    content_height = max(candidate.height, reference.height)
    width = candidate.width + gap + reference.width
    panel = Image.new("RGB", (width, header + content_height), "black")
    panel.paste(candidate, (0, header + (content_height - candidate.height) // 2))
    panel.paste(reference, (
        candidate.width + gap,
        header + (content_height - reference.height) // 2,
    ))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 10), "CANDIDATE", fill="white")
    draw.text((candidate.width + gap + 6, 10), f"ENROLLED REFERENCE: {label}", fill="white")
    return panel


def _fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    if scale >= 1.0:
        return image.copy()
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)
