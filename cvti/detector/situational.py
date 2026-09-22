"""Lightweight situational HSE candidate detectors.

These are intentionally small candidate generators for demo/pilot use. They do
not replace the VLM gate; they create cheap temporal signals that the existing
Customization Engine and Verification Gate can confirm with scene context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _frame_diagonal(frame_shape: tuple[int, ...]) -> float:
    h, w = frame_shape[:2]
    return max((float(w) ** 2 + float(h) ** 2) ** 0.5, 1.0)


@dataclass
class _TrackMotion:
    center: tuple[float, float]
    timestamp: float
    fast_frames: int = 0
    latched: bool = False


@dataclass
class RunningPanicDetector:
    """Detect sustained fast person movement from tracked bbox centers."""

    min_speed_ratio: float = 0.18
    min_frames: int = 3
    reset_speed_ratio: float = 0.08
    _tracks: dict[int, _TrackMotion] = field(default_factory=dict)

    def update(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        timestamp: float,
        frame_shape: tuple[int, ...],
    ) -> dict[str, Any] | None:
        center = _bbox_center(bbox)
        # Sweep departed tracks: ByteTrack ids never return, so anything not
        # updated for 30s is a person who left — one leaked _TrackMotion per
        # visitor otherwise, forever. (RAM audit 24 Aug, #3.)
        stale = [t for t, m in self._tracks.items() if timestamp - m.timestamp > 30.0]
        for t in stale:
            self._tracks.pop(t, None)
        previous = self._tracks.get(track_id)
        if previous is None:
            self._tracks[track_id] = _TrackMotion(center=center, timestamp=timestamp)
            return None

        dt = max(timestamp - previous.timestamp, 1e-3)
        dx = center[0] - previous.center[0]
        dy = center[1] - previous.center[1]
        speed_px = ((dx * dx + dy * dy) ** 0.5) / dt
        speed_ratio = speed_px / _frame_diagonal(frame_shape)

        fast_frames = previous.fast_frames + 1 if speed_ratio >= self.min_speed_ratio else 0
        latched = previous.latched
        fired = None
        if fast_frames >= self.min_frames and not latched:
            latched = True
            fired = {
                "kind": "running",
                "track_id": track_id,
                "bbox": bbox,
                "speed_ratio": round(speed_ratio, 4),
                "fast_frames": fast_frames,
            }
        if speed_ratio <= self.reset_speed_ratio:
            latched = False

        self._tracks[track_id] = _TrackMotion(
            center=center,
            timestamp=timestamp,
            fast_frames=fast_frames,
            latched=latched,
        )
        return fired


@dataclass
class CrowdFormationDetector:
    """Detect a persistent tight cluster of people."""

    min_people: int = 4
    min_frames: int = 3
    max_cluster_ratio: float = 0.24
    _cluster_frames: int = 0
    _latched: bool = False

    def update(
        self,
        people: list[dict[str, Any]],
        timestamp: float,
        frame_shape: tuple[int, ...],
    ) -> dict[str, Any] | None:
        if len(people) < self.min_people:
            self._cluster_frames = 0
            self._latched = False
            return None

        centers = [_bbox_center(tuple(p["bbox"])) for p in people]
        best: list[int] = []
        frame_diag = _frame_diagonal(frame_shape)
        max_distance = self.max_cluster_ratio * frame_diag
        for idx, center in enumerate(centers):
            members = [
                j
                for j, other in enumerate(centers)
                if ((center[0] - other[0]) ** 2 + (center[1] - other[1]) ** 2) ** 0.5
                <= max_distance
            ]
            if len(members) > len(best):
                best = members

        if len(best) < self.min_people:
            self._cluster_frames = 0
            self._latched = False
            return None

        self._cluster_frames += 1
        if self._cluster_frames < self.min_frames or self._latched:
            return None

        self._latched = True
        member_people = [people[i] for i in best]
        xs: list[int] = []
        ys: list[int] = []
        for person in member_people:
            x1, y1, x2, y2 = tuple(person["bbox"])
            xs.extend([int(x1), int(x2)])
            ys.extend([int(y1), int(y2)])
        return {
            "kind": "crowd_formation",
            "people_count": len(member_people),
            "track_ids": [p.get("track_id") for p in member_people],
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "cluster_frames": self._cluster_frames,
            "timestamp": timestamp,
        }


@dataclass
class FireSmokeCandidateDetector:
    """Flag flame-coloured or smoke-like regions that are NEW to the scene and
    keep changing — never the scene's own colours.

    The first version thresholded colour over the whole frame: any frame with
    ≥1.2% warm pixels or ≥8% grey pixels was a candidate. Measured on 22 Sep
    against normal footage that flagged EVERY frame — 39/39 of an ordinary
    indoor camera, 116/116 of a night-IR driveway (grey = "smoke"), every
    gate and PPE clip — so each engine start produced a critical fire alert
    within seconds of the settle window, and it kept happening ("the baseline
    fire just comes up from start", pilot, 22 Sep). Fire is a critical alert
    and reaches the operator BEFORE verification, so a colour histogram of the
    room was ringing the phone.

    What fire and smoke actually are, in a fixed camera: regions that were not
    there (relative to a slowly learned background of the scene's own warm and
    grey areas), that flicker or drift frame to frame, and that persist for a
    few frames. A warm wall, a wooden floor, an orange sign and an IR-lit yard
    are background; they never become candidates. Something warm that appears
    and moves (a hi-vis vest walking through) still can — that is the VLM
    gate's call, and its prompt already says signage and lighting are not fire.
    Colour smoke detection is skipped on monochrome (IR night) frames, where
    "grey" is every pixel.

    Tried and rejected on the same footage (22 Sep): masking out YOLO
    person/vehicle boxes (no gain on people, worse on gates — the box hides
    the floor and the floor comes back "new"), and a centroid-drift test for
    in-place flicker (no gain). Raising the brightness floor did the work.
    """

    min_frames: int = 3
    min_hot_area_ratio: float = 0.012
    # Whole-frame ceilings (3 Sep): whole-frame warm or grey is the camera's
    # own optics (IR bloom, exposure hunt, fog), not a candidate.
    max_hot_area_ratio: float = 0.65
    min_smoke_area_ratio: float = 0.08
    max_smoke_area_ratio: float = 0.65
    # Background model: an EMA of each pixel's hot/grey membership. 0.05 ≈ 20
    # frames (~5 s at the engine's 4 fps) to absorb a change into "normal".
    background_alpha: float = 0.05
    # Learn the scene before judging it (~3 s at 4 fps). Together with the
    # camera's 8 s settle window this covers the exposure/IR hunt at start.
    warmup_frames: int = 12
    # Flicker: fraction of the new region that changed since the last frame.
    # A flame or a smoke plume never holds still; a newly parked orange car
    # does, and drops out here.
    min_change_ratio: float = 0.15
    # Mean HSV saturation below this = monochrome frame (IR night mode).
    min_saturation_for_smoke: float = 12.0
    # After one candidate, hold the same camera quiet for this long unless the
    # new area doubles. One provisional per camera per 5 min at most — the
    # engine and the gate decide the rest.
    rearm_seconds: float = 300.0
    # Flame is BRIGHT: V floor for the hot mask. Skin shares the hue band
    # (0-35) at V 130-190 and made every walking person a candidate; flame
    # cores sit near white. Measured 22 Sep: 130 -> 180 cut people-clip
    # events 15 -> 3 per minute with every real fire still caught.
    hot_min_value: int = 180
    # Analyse at this width; colour masks do not need full resolution.
    analysis_width: int = 320

    _bg_hot: Any = field(default=None, init=False, repr=False)
    _bg_smoke: Any = field(default=None, init=False, repr=False)
    _prev_new_hot: Any = field(default=None, init=False, repr=False)
    _prev_new_smoke: Any = field(default=None, init=False, repr=False)
    _frames_seen: int = field(default=0, init=False, repr=False)
    # Candidate flags for the last 2*min_frames frames. Persistence is
    # "min_frames hits in that window", not "consecutive": a flame's
    # frame-to-frame change is not uniform, and one quiet frame must not
    # reset the count (it did, and a steady synthetic flame never fired).
    _recent: list = field(default_factory=list, init=False, repr=False)
    _candidate_frames: int = field(default=0, init=False, repr=False)
    _latched: bool = field(default=False, init=False, repr=False)
    _last_fired_ts: float = field(default=float("-inf"), init=False, repr=False)
    _last_fired_area: float = field(default=0.0, init=False, repr=False)

    def _masks(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        h, w = frame.shape[:2]
        if w > self.analysis_width:
            frame = cv2.resize(frame, (self.analysis_width, max(1, int(h * self.analysis_width / w))),
                               interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hot = cv2.inRange(hsv, np.array([0, 70, self.hot_min_value]), np.array([35, 255, 255]))
        smoke = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 60, 230]))
        return (hot > 0), (smoke > 0), float(hsv[..., 1].mean())

    @staticmethod
    def _change(now: np.ndarray, prev: Any) -> float:
        """Fraction of the current region that differs from the previous frame."""
        area = float(now.sum())
        if area <= 0:
            return 0.0
        if prev is None or prev.shape != now.shape:
            return 1.0
        return float(np.logical_xor(now, prev).sum()) / area

    def update(self, frame: np.ndarray, timestamp: float) -> dict[str, Any] | None:
        if frame.size == 0:
            return None
        hot, smoke, saturation = self._masks(frame)
        if self._bg_hot is None or self._bg_hot.shape != hot.shape:
            # New scene (first frame, or a resolution change): learn, don't judge.
            self._bg_hot = hot.astype(np.float32)
            self._bg_smoke = smoke.astype(np.float32)
            self._prev_new_hot = self._prev_new_smoke = None
            self._frames_seen = 1
            self._recent = []
            self._candidate_frames = 0
            self._latched = False
            return None

        new_hot = np.logical_and(hot, self._bg_hot < 0.5)
        new_smoke = np.logical_and(smoke, self._bg_smoke < 0.5)
        a = self.background_alpha
        self._bg_hot += a * (hot.astype(np.float32) - self._bg_hot)
        self._bg_smoke += a * (smoke.astype(np.float32) - self._bg_smoke)
        self._frames_seen += 1

        total = float(hot.size)
        hot_ratio = float(hot.sum()) / total
        new_hot_ratio = float(new_hot.sum()) / total
        new_smoke_ratio = float(new_smoke.sum()) / total
        hot_change = self._change(new_hot, self._prev_new_hot)
        smoke_change = self._change(new_smoke, self._prev_new_smoke)
        self._prev_new_hot, self._prev_new_smoke = new_hot, new_smoke

        if self._frames_seen <= self.warmup_frames:
            return None

        hot_candidate = (self.min_hot_area_ratio <= new_hot_ratio
                         and hot_ratio <= self.max_hot_area_ratio
                         and hot_change >= self.min_change_ratio)
        smoke_candidate = (saturation >= self.min_saturation_for_smoke
                           and self.min_smoke_area_ratio <= new_smoke_ratio <= self.max_smoke_area_ratio
                           and smoke_change >= self.min_change_ratio)
        self._recent.append(bool(hot_candidate or smoke_candidate))
        del self._recent[:-max(1, 2 * self.min_frames)]
        self._candidate_frames = sum(self._recent)
        if self._candidate_frames == 0:
            self._latched = False
            return None
        if not (hot_candidate or smoke_candidate) or self._candidate_frames < self.min_frames \
                or self._latched:
            return None
        self._latched = True

        area = max(new_hot_ratio, new_smoke_ratio)
        if (timestamp - self._last_fired_ts) < self.rearm_seconds and area < 2.0 * self._last_fired_area:
            return None                     # same camera, same-sized episode, too soon
        self._last_fired_ts = timestamp
        self._last_fired_area = area
        return {
            "kind": "fire_smoke",
            "hot_area_ratio": round(new_hot_ratio, 4),
            "smoke_area_ratio": round(new_smoke_ratio, 4),
            "change_ratio": round(max(hot_change, smoke_change), 3),
            "candidate_frames": self._candidate_frames,
            "timestamp": timestamp,
        }
