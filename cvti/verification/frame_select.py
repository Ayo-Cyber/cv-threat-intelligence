"""Per-rule evidence-frame selection for the Verification Gate (plan.md Phase 4).

Different threats want different frames sent to the VLM:

- weapon:      1 clearest full frame (a crisp still beats motion blur)
- violence:    3-4 frames spanning the motion peak (the act needs temporal context)
- concealment: 3-4 frames around the reach/retract moment
- robbery:     4-6 frames around the compound event
- zone/time:   1 frame + the event metadata is usually enough

Instead of always sending the first (often blurry) frame, we pick from a rolling
buffer of recent frames: motion-peak anchored for multi-frame rules, sharpest
frame for single-frame rules. Ported from the offline tools/gate_bakeoff.py logic.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from cvti.logging_setup import get_logger

log = get_logger(__name__)

# rule-name / detector keyword -> how many frames to send the gate
_RULE_FRAMES = {
    "weapon": 1, "weapons": 1, "gun": 1, "knife": 1,
    "violence": 4, "assault": 4, "fight": 4,
    "concealment": 3, "shoplift": 3, "theft": 3,
    "robbery": 5, "armed_robbery": 5,
    "presence": 1, "loiter": 1, "zone": 1, "after_hours": 1, "intrusion": 1,
}
_DEFAULT_FRAMES = 3


def frames_for_rule(rule_name: str) -> int:
    """How many evidence frames this rule should send the gate."""
    name = (rule_name or "").lower()
    for key, n in _RULE_FRAMES.items():
        if key in name:
            return n
    return _DEFAULT_FRAMES


def _sharpness(frame: np.ndarray) -> float:
    """Variance-of-Laplacian: higher = crisper (less motion blur)."""
    try:
        import cv2
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception as exc:  # noqa: BLE001
        log.debug("blur score unavailable for a frame", exc_info=True)
        return 0.0


def _motion_scores(frames: list[np.ndarray]) -> list[float]:
    """Per-frame motion = mean abs diff vs the previous frame (downscaled+gray)."""
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        log.debug("blur scoring unavailable", exc_info=True)
        return [0.0] * len(frames)
    small = []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.int16)
        small.append(g)
    scores = [0.0]
    for i in range(1, len(small)):
        scores.append(float(np.abs(small[i] - small[i - 1]).mean()))
    return scores


def select_evidence_frames(recent: list[np.ndarray], rule_name: str,
                           *, count: int | None = None) -> tuple[list[np.ndarray], dict]:
    """Pick evidence frames for `rule_name` from a chronological `recent` buffer.

    Returns (frames, meta) where meta records the strategy + chosen indices so it
    can be written into the alert artifacts.
    """
    n = count if count is not None else frames_for_rule(rule_name)
    if not recent:
        return [], {"strategy": "none", "count": 0, "selected_indices": []}

    if n <= 1:
        # Single frame: send the sharpest (clearest) one available.
        idx = max(range(len(recent)), key=lambda i: _sharpness(recent[i]))
        return [recent[idx]], {"strategy": "sharpest", "count": 1,
                               "selected_indices": [idx], "buffer_len": len(recent)}

    # Multi-frame: spread n frames evenly across the whole recent buffer so the
    # gate sees the motion into the act (the buffer ends at the flagged moment).
    # The motion peak is recorded for reference/artifacts.
    motion = _motion_scores(recent)
    anchor = max(range(len(recent)), key=lambda i: motion[i])
    last = len(recent) - 1
    if len(recent) <= n:
        chosen = list(range(len(recent)))
    else:
        chosen = sorted({round(k * last / (n - 1)) for k in range(n)})
    return ([recent[i] for i in chosen],
            {"strategy": "motion_peak_span", "count": len(chosen),
             "selected_indices": chosen, "anchor_index": anchor, "buffer_len": len(recent)})


# ---------------------------------------------------------------------------
# Subject crop: the second half of the evidence upgrade (W5, 9 Sep 2026).
# A person in full-frame CCTV is often <50px tall — the gate's own rejections
# say "no people are visible" on clips where the detector tracked someone. A
# zoomed crop of the flagged subject, appended AFTER the full frames, lets the
# VLM inspect hands and held objects without losing scene context.

_CROP_MARGIN = 0.40      # context around the box, as a fraction of its size
_CROP_MIN_SIDE = 320     # upscale small crops so the VLM gets real pixels


def subject_crop(frame: np.ndarray, bbox: Any,
                 *, margin: float = _CROP_MARGIN,
                 min_side: int = _CROP_MIN_SIDE) -> np.ndarray | None:
    """A zoomed view of `bbox` (x1, y1, x2, y2) with context margin.

    Returns None when the box is degenerate or the crop fails — evidence then
    stays full-frame-only, which is the old behaviour."""
    try:
        import cv2
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        bw, bh = x2 - x1, y2 - y1
        if bw <= 2 or bh <= 2:
            return None
        mx, my = bw * margin, bh * margin
        cx1 = max(0, int(x1 - mx)); cy1 = max(0, int(y1 - my))
        cx2 = min(w, int(x2 + mx)); cy2 = min(h, int(y2 + my))
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        side = min(crop.shape[0], crop.shape[1])
        if side < min_side:
            scale = min_side / side
            crop = cv2.resize(crop, (int(crop.shape[1] * scale),
                                     int(crop.shape[0] * scale)),
                              interpolation=cv2.INTER_CUBIC)
        return crop
    except Exception as exc:  # noqa: BLE001 - a failed crop must never cost the alert
        log.debug("subject crop failed; sending full frames only", exc_info=True)
        return None


def append_subject_crop(frames: list[np.ndarray], moment: np.ndarray,
                        bbox: Any) -> list[np.ndarray]:
    """Evidence list = full frames + (when a subject is boxed) a zoomed crop.

    `moment` is the frame the rule fired on — the bbox belongs to IT, not to
    the buffered evidence frames, so the crop is always cut from the moment."""
    if bbox is None:
        return frames
    crop = subject_crop(moment, bbox)
    return frames if crop is None else frames + [crop]
