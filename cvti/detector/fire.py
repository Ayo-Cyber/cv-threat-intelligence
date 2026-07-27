"""Fire + smoke detection — cheap CV pre-filter, no model.

This is a *trigger*, not a verdict: it proposes a candidate and TrueSight (the VLM
gate) confirms whether it's really fire/smoke — so the pre-filter can be permissive
without spamming the operator (a sunset, a red sign, or a lava lamp gets rejected by
the gate).

Two signals:
  • fire  — flame-coloured pixels (orange/red/yellow, bright + saturated) covering a
            meaningful area AND *flickering* over time. The flicker (frame-to-frame
            oscillation of the coloured area) is what separates real flame from a
            static orange wall or a steady sunset.
  • smoke — low-saturation grey haze that is *moving* (differs from a slow background)
            and covers a growing area. Smoke has little colour and softens edges.

Per-kind cooldown so a persistent fire fires once, not every frame.
"""
from __future__ import annotations

from collections import deque


class FireSmokeDetector:
    def __init__(
        self,
        *,
        fire_area_frac: float = 0.006,   # min flame-coloured area (fraction of frame)
        flicker_window: int = 6,         # frames of history for the flicker test
        flicker_cv: float = 0.12,        # min coeff. of variation of area → "flickering"
        smoke_area_frac: float = 0.05,   # min moving-haze area
        smoke_motion: float = 6.0,       # min mean grey diff inside the haze region
        warmup: int = 6,
        cooldown_seconds: float = 60.0,
        detect_smoke: bool = True,
    ) -> None:
        self.fire_area_frac = fire_area_frac
        self.flicker_window = flicker_window
        self.flicker_cv = flicker_cv
        self.smoke_area_frac = smoke_area_frac
        self.smoke_motion = smoke_motion
        self.warmup = warmup
        self.cooldown = cooldown_seconds
        self.detect_smoke = detect_smoke
        self._n = 0
        self._fire_hist: deque = deque(maxlen=flicker_window)
        self._prev_gray = None
        self._bg_gray = None                       # slow background for smoke motion
        self._last_fire = -1e9
        self._last_smoke = -1e9

    def update(self, frame_bgr, timestamp: float = 0.0) -> dict | None:
        import cv2
        import numpy as np
        self._n += 1
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        total = float(frame_bgr.shape[0] * frame_bgr.shape[1])
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ---- fire: flame-coloured + bright + saturated ----
        flame = (((h <= 35) | (h >= 160)) & (s >= 90) & (v >= 150))
        fire_frac = float(flame.sum()) / total
        self._fire_hist.append(fire_frac)
        fire = None
        if (self._n > self.warmup and len(self._fire_hist) >= self.flicker_window
                and timestamp - self._last_fire >= self.cooldown):
            arr = np.array(self._fire_hist, dtype=float)
            mean = float(arr.mean())
            if mean >= self.fire_area_frac:
                cv = float(arr.std()) / max(mean, 1e-6)      # flicker = area oscillation
                if cv >= self.flicker_cv:
                    self._last_fire = timestamp
                    fire = {"kind": "fire", "area_frac": round(mean, 4), "flicker": round(cv, 3)}

        # ---- smoke: low-sat grey haze that moves vs a slow background ----
        # `_bg_gray` is a float32 running-average accumulator; compare against its
        # uint8 view so absdiff types match.
        smoke = None
        if self.detect_smoke and self._bg_gray is not None:
            bg_u8 = cv2.convertScaleAbs(self._bg_gray)
            diff = cv2.absdiff(gray, bg_u8)
            haze = (s < 55) & (v > 70) & (v < 210)
            moving_haze = haze & (diff > 12)
            smoke_frac = float(moving_haze.sum()) / total
            if (self._n > self.warmup and smoke_frac >= self.smoke_area_frac
                    and timestamp - self._last_smoke >= self.cooldown):
                motion = float(diff[moving_haze].mean()) if moving_haze.any() else 0.0
                if motion >= self.smoke_motion:
                    self._last_smoke = timestamp
                    smoke = {"kind": "smoke", "area_frac": round(smoke_frac, 4),
                             "motion": round(motion, 1)}

        # update the slow background accumulator (float32)
        if self._bg_gray is None:
            self._bg_gray = gray.astype("float32")
        else:
            cv2.accumulateWeighted(gray, self._bg_gray, 0.02)
        self._prev_gray = gray

        # fire is more urgent — return it in preference to smoke
        return fire or smoke
