"""Skip the detector on frames where the scene has not changed (compute, 25 Sep).

Detection samples on a TIMER: `target_fps` frames per camera per second, every
second the engine is up, whether or not anything is happening. CCTV is mostly
nothing happening — an empty warehouse at 3am costs exactly what a busy gate
costs, ~45ms of YOLO per frame per camera. On a hub sized at roughly 4-5
cameras that fixed floor is most of the machine, and it is spent looking at
pixels identical to the ones already looked at.

This gate answers one question before the detector runs: has anything changed
since the last frame we actually analysed? It is deliberately cheap (a
downscaled greyscale mean-absolute-difference, microseconds) and deliberately
timid — it is far better to run the detector needlessly than to miss an event,
so three rails force a real detection regardless of how still the scene looks:

  * anything currently TRACKED. A person standing still to loiter produces
    almost no pixel change; dropping those frames would break the very dwell
    the zone rules measure. While the camera holds a track, the gate is off.
  * a heartbeat. Whatever happens, no camera goes longer than
    `heartbeat_seconds` without a real detection, so a gradual change (dusk,
    fog, a slow push-in) can never leave a camera blind.
  * the first frame of a camera, and any frame whose size changed (a stream
    reconnect at a different resolution) — there is nothing to compare to.

What "changed" means was chosen by measurement (25 Sep), not by taste. The
obvious metric, mean absolute difference, is dominated by codec noise: on the
KPI normals it skipped 12% of empty frames but also 18% of frames containing a
real event -- worse than useless. Counting the FRACTION OF PIXELS that moved
by more than a noise floor separates cleanly, because compression noise nudges
many pixels slightly while a person moves a contiguous block a lot:

    genuinely idle fixed camera   0.00% of pixels moved
    clips containing a person     6.5% - 35%

so a threshold anywhere in between is safe by a wide margin. Numbers are
per-site configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MotionGate:
    """Per-camera 'has anything changed?' gate in front of the detector."""

    # Fraction of pixels that must move for the frame to count as changed.
    # Idle fixed cameras measure 0.00%; anything with a person in it measures
    # >=6.5%. 1% sits an order of magnitude clear of both edges.
    min_changed_fraction: float = 0.01
    # A pixel counts as moved when it shifts this many grey levels. Codec and
    # sensor noise live well below this; real subjects are well above it.
    noise_floor: float = 25.0
    # No camera may go longer than this without a real detection.
    heartbeat_seconds: float = 2.0
    # Differencing runs on a thumbnail; full resolution buys nothing here.
    analysis_width: int = 160

    _reference: dict = field(default_factory=dict, init=False, repr=False)
    _last_detect: dict = field(default_factory=dict, init=False, repr=False)
    skipped: int = field(default=0, init=False)
    ran: int = field(default=0, init=False)

    def _thumb(self, image: Any) -> Any:
        import cv2
        h, w = image.shape[:2]
        if w > self.analysis_width:
            scale = self.analysis_width / float(w)
            image = cv2.resize(image, (self.analysis_width, max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.float32)

    def change(self, camera_id: str, image: Any) -> float | None:
        """Fraction of pixels that moved beyond the noise floor since this
        camera's reference frame, or None when there is nothing to compare."""
        thumb = self._thumb(image)
        ref = self._reference.get(camera_id)
        if ref is None or ref.shape != thumb.shape:
            return None
        return float((np.abs(thumb - ref) > self.noise_floor).mean())

    def should_detect(self, camera_id: str, image: Any, *,
                      tracked: int = 0, now: float = 0.0) -> bool:
        """True when the detector should run on this frame.

        `tracked` is how many objects the camera is currently following; any
        live track disables the gate for that camera.
        """
        last = self._last_detect.get(camera_id)
        delta = self.change(camera_id, image)
        forced = (
            delta is None                                   # new camera / new size
            or tracked > 0                                  # something is being followed
            or last is None
            or (now - last) >= self.heartbeat_seconds       # heartbeat
        )
        if not forced and delta < self.min_changed_fraction:
            self.skipped += 1
            return False
        self._reference[camera_id] = self._thumb(image)
        self._last_detect[camera_id] = now
        self.ran += 1
        return True

    def stats(self) -> dict:
        total = self.ran + self.skipped
        return {"ran": self.ran, "skipped": self.skipped,
                "skipped_fraction": round(self.skipped / total, 4) if total else 0.0}
