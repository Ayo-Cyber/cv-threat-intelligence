"""Per-camera decode thread with latest-frame-drop + FPS governor.

Each camera gets one background thread that reads its RTSP/file/webcam source
and keeps only the MOST RECENT frame. If inference falls behind, stale frames
are discarded rather than queued — so the box stays near real-time instead of
drifting further behind on every camera.

A target-FPS governor throttles the decode loop so a 30 FPS source does not
burn CPU decoding frames the pipeline will only sample at ~5 FPS.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Frame:
    camera_id: str
    image: Any          # numpy BGR array
    index: int
    timestamp: float    # seconds since decoder start


class StreamDecoder:
    def __init__(self, camera_id: str, source: int | str, *, target_fps: float = 5.0,
                 reconnect: bool = True) -> None:
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.reconnect = reconnect
        self._min_period = 1.0 / target_fps if target_fps > 0 else 0.0
        self._latest: Frame | None = None
        self._consumed = True          # True once the current latest was read
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self._t0 = 0.0
        self._fps = 0.0
        self.ended = False

    def start(self) -> "StreamDecoder":
        self._thread = threading.Thread(target=self._loop, name=f"decode-{self.camera_id}",
                                        daemon=True)
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def _open(self):
        import cv2
        src = int(self.source) if str(self.source).isdigit() else self.source
        return cv2.VideoCapture(src)

    def _loop(self) -> None:
        import cv2
        cap = self._open()
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        # Sample target_fps by SKIPPING frames, not by slowing playback: keep 1
        # frame every `stride`. This makes a file advance through video at real
        # time while only decoding the frames we need (and drops a 30fps live
        # source to the same sampled rate).
        stride = max(1, round(self._fps / self.target_fps)) \
            if (self._fps > 1e-3 and self.target_fps > 0) else 1
        orig_index = 0
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            ok = True
            for _ in range(stride - 1):        # cheaply skip intermediate frames
                ok = cap.grab()
                orig_index += 1
                if not ok:
                    break
            if ok:
                ok, image = cap.read()
                orig_index += 1
            if not ok:
                if self.reconnect and not str(self.source).isdigit() and "://" in str(self.source):
                    # Live source hiccup — back off and reopen.
                    cap.release()
                    time.sleep(1.0)
                    cap = self._open()
                    continue
                self.ended = True          # file/webcam exhausted
                break
            # VIDEO time from the original frame index so dwell/loiter thresholds
            # are correct regardless of sample rate. Wall-clock fallback if no fps.
            ts = (orig_index / self._fps) if self._fps > 1e-3 else (time.perf_counter() - self._t0)
            with self._lock:
                self._index += 1
                self._latest = Frame(self.camera_id, image, orig_index, ts)
                self._consumed = False
            # Pace each kept frame to target_fps: files play at real time; a live
            # source is naturally paced already, so this just caps intake.
            if self._min_period:
                elapsed = time.perf_counter() - loop_start
                if elapsed < self._min_period:
                    time.sleep(self._min_period - elapsed)
        cap.release()

    def read_latest(self) -> Frame | None:
        """Return the newest unread frame, or None if nothing new since last read."""
        with self._lock:
            if self._latest is None or self._consumed:
                return None
            self._consumed = True
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
