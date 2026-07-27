"""Crowd / stampede detection — cheap CV on person counts + global motion, no model.

Two related signals:
  • crowd     — an unusually high number of people in view, sustained for a few
                frames (a gathering / overcrowding).
  • stampede  — a crowd that is ALSO moving violently: the mean frame-to-frame
                motion spikes well above the scene's baseline. Sudden panic/rush.

Both use a per-scene baseline so a normally-busy camera doesn't cry crowd, and a
cooldown so one event fires once. Triggers only; TrueSight confirms.
"""
from __future__ import annotations

from collections import deque


class CrowdDetector:
    def __init__(
        self,
        *,
        crowd_count: int = 8,          # people in view to count as a crowd
        min_frames: int = 4,           # sustained frames before firing crowd
        stampede_count: int = 5,       # min people for a stampede to be meaningful
        motion_spike: float = 3.0,     # current motion >= this * baseline → spike
        motion_floor: float = 8.0,     # absolute min motion so a still scene can't spike
        warmup: int = 8,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.crowd_count = crowd_count
        self.min_frames = min_frames
        self.stampede_count = stampede_count
        self.motion_spike = motion_spike
        self.motion_floor = motion_floor
        self.warmup = warmup
        self.cooldown = cooldown_seconds
        self._n = 0
        self._crowd_streak = 0
        self._prev_gray = None
        self._motion_base: float | None = None
        self._last_crowd = -1e9
        self._last_stampede = -1e9

    def update(self, person_count: int, frame_bgr, timestamp: float = 0.0) -> dict | None:
        import cv2
        self._n += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        motion = 0.0
        if self._prev_gray is not None:
            motion = float(cv2.absdiff(gray, self._prev_gray).mean())
        self._prev_gray = gray

        # stampede: many people + a big motion spike over the learned baseline
        stampede = None
        if self._motion_base is not None and self._n > self.warmup:
            spike = motion >= max(self.motion_floor, self.motion_spike * self._motion_base)
            if (person_count >= self.stampede_count and spike
                    and timestamp - self._last_stampede >= self.cooldown):
                self._last_stampede = timestamp
                stampede = {"kind": "stampede", "people": int(person_count),
                            "motion": round(motion, 1), "baseline": round(self._motion_base, 1)}
        # adapt the motion baseline slowly (always — it tracks "normal" busyness)
        self._motion_base = motion if self._motion_base is None else \
            self._motion_base + 0.05 * (motion - self._motion_base)

        # crowd: sustained high headcount
        crowd = None
        if person_count >= self.crowd_count:
            self._crowd_streak += 1
            if (self._crowd_streak >= self.min_frames
                    and timestamp - self._last_crowd >= self.cooldown):
                self._last_crowd = timestamp
                crowd = {"kind": "crowd", "people": int(person_count)}
        else:
            self._crowd_streak = 0

        return stampede or crowd     # stampede is the more urgent
