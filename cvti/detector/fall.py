"""Fall / person-collapsed detection — cheap CV on tracked person boxes, no model.

A person lying on the ground (medical emergency, assault aftermath, faint) reads as
a person box that goes *horizontal* (wider than tall) and stays that way. A brief
bend or crouch shouldn't fire, so we require the horizontal posture to persist for
several consecutive frames on the same track. One alert per track per episode
(latched until they stand back up), with a cooldown so a persistent body on the
floor doesn't re-fire endlessly.

This is a trigger; TrueSight confirms whether the person is actually collapsed vs.
sitting/crouching/lying down normally.
"""
from __future__ import annotations


class FallDetector:
    def __init__(
        self,
        *,
        down_aspect: float = 1.15,    # w/h at/above this = horizontal (fallen)
        up_aspect: float = 0.85,      # w/h at/below this = upright (reset latch)
        min_frames: int = 4,          # consecutive down frames before firing
        min_box_frac: float = 0.004,  # ignore tiny/far boxes (noise)
        cooldown_seconds: float = 90.0,
    ) -> None:
        self.down_aspect = down_aspect
        self.up_aspect = up_aspect
        self.min_frames = min_frames
        self.min_box_frac = min_box_frac
        self.cooldown = cooldown_seconds
        self._streak: dict = {}       # track_id -> consecutive down frames
        self._fired: dict = {}        # track_id -> last-fired timestamp

    def update(self, person_boxes, frame_area: float, timestamp: float = 0.0) -> dict | None:
        """`person_boxes`: list of (track_id, x1, y1, x2, y2). Returns a fall dict or None."""
        seen = set()
        result = None
        for tid, x1, y1, x2, y2 in person_boxes:
            seen.add(tid)
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            if (w * h) / max(frame_area, 1.0) < self.min_box_frac:
                continue
            aspect = w / h
            if aspect >= self.down_aspect:
                self._streak[tid] = self._streak.get(tid, 0) + 1
                last = self._fired.get(tid, -1e9)
                if self._streak[tid] >= self.min_frames and timestamp - last >= self.cooldown:
                    self._fired[tid] = timestamp
                    if result is None:      # one candidate per call; prefer the first
                        result = {"kind": "fall", "track_id": int(tid), "aspect": round(aspect, 2)}
            elif aspect <= self.up_aspect:
                self._streak[tid] = 0        # clearly upright → reset latch
        # forget tracks that vanished
        for tid in [t for t in self._streak if t not in seen]:
            self._streak.pop(tid, None)
        return result
