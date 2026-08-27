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
from collections import deque
from dataclasses import dataclass
from typing import Any
from cvti.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Frame:
    camera_id: str
    image: Any          # numpy BGR array
    index: int
    timestamp: float    # seconds since decoder start


# An offline camera produces no alerts — which looks exactly like a camera
# watching a quiet corridor. The tamper detector answers "did someone cover
# this?"; nothing answered "has this been unreachable since Tuesday?". In a
# security product false confidence is worse than visible failure, so the link
# state is explicit and reported rather than inferred from silence.
CONNECTED = "connected"
RECONNECTING = "reconnecting"
OFFLINE = "offline"

# How long a camera may be reconnecting before we call it offline and tell
# someone. Short enough to catch a real outage, long enough that a camera
# reboot or a network blip does not page anyone.
DEFAULT_OFFLINE_GRACE = 60.0


class StreamDecoder:
    def __init__(self, camera_id: str, source: int | str, *, target_fps: float = 5.0,
                 reconnect: bool = True, reconnect_backoff: float = 1.0,
                 loop_files: bool = True, offline_grace_seconds: float = DEFAULT_OFFLINE_GRACE,
                 on_state_change=None) -> None:
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.reconnect = reconnect
        self.reconnect_backoff = reconnect_backoff   # seconds * attempt, capped at 5s
        # File sources loop by default so demo footage streams continuously
        # (real RTSP/webcam sources are live and never hit EOF).
        self.loop_files = loop_files
        self._min_period = 1.0 / target_fps if target_fps > 0 else 0.0
        self._latest: Frame | None = None
        self._seq = 0                      # bumps per decoded frame (peek dedup)
        self.stale_dropped = 0             # buffered frames skipped to stay live
        self._consumed = True          # True once the current latest was read
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self._t0 = 0.0
        self._fps = 0.0
        self.ended = False
        self.reconnects = 0            # how many times we've re-opened a live stream
        self.offline_grace_seconds = offline_grace_seconds
        self.on_state_change = on_state_change
        self.state = CONNECTED
        self.state_since = time.time()
        self.last_frame_at = 0.0
        # Bounded: the useful signal is "it has been flapping", not every attempt
        # since install.
        self.attempt_history: deque = deque(maxlen=20)

    @property
    def time_in_state(self) -> float:
        return time.time() - self.state_since

    def _set_state(self, state: str, detail: str = "") -> None:
        if state == self.state:
            return
        previous, held = self.state, self.time_in_state
        self.state, self.state_since = state, time.time()
        level = log.warning if state != CONNECTED else log.info
        level("[link %s] %s -> %s after %.0fs%s", self.camera_id, previous, state,
              held, f" ({detail})" if detail else "")
        if self.on_state_change is not None:
            try:
                self.on_state_change(self.camera_id, previous, state, held)
            except Exception:  # noqa: BLE001
                log.warning("[link %s] state-change callback failed", self.camera_id,
                            exc_info=True)

    def link_status(self) -> dict:
        return {"camera_id": self.camera_id, "state": self.state,
                "time_in_state": round(self.time_in_state, 1),
                "reconnects": self.reconnects,
                "last_frame_at": self.last_frame_at,
                # The liveness number the UI shows. A connected camera whose
                # frames stopped is exactly the ambiguity this exists to remove.
                "last_frame_age_s": (round(time.time() - self.last_frame_at, 1)
                                     if self.last_frame_at else None),
                # How far behind live we would have been: buffered frames
                # skipped to stay at the present. A big number on a camera
                # means its stream bursts (HLS), not that anything is wrong.
                "stale_dropped": self.stale_dropped,
                "attempts": list(self.attempt_history)}

    def _is_live(self) -> bool:
        return not str(self.source).isdigit() and "://" in str(self.source)

    def start(self) -> "StreamDecoder":
        self._thread = threading.Thread(target=self._loop, name=f"decode-{self.camera_id}",
                                        daemon=True)
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def _open(self):
        import os

        import cv2
        src = int(self.source) if str(self.source).isdigit() else self.source
        if isinstance(src, str) and src.lower().startswith("rtsp"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            return cv2.VideoCapture(src, cv2.CAP_FFMPEG)
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
        attempt = 0
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
                if self.reconnect and self._is_live():
                    # Live source dropped (camera reboot / network blip). Keep
                    # reopening with an exponential, capped backoff until it
                    # comes back — linear retry hammers a camera that is down
                    # for an hour, and the log with it.
                    self.reconnects += 1
                    attempt += 1
                    backoff = min(self.reconnect_backoff * (2 ** (attempt - 1)), 30.0)
                    if self.state == CONNECTED:
                        self._set_state(RECONNECTING, "stream dropped")
                    elif (self.state == RECONNECTING
                          and self.time_in_state >= self.offline_grace_seconds):
                        self._set_state(OFFLINE,
                                        f"unreachable for {self.time_in_state:.0f}s")
                    self.attempt_history.append(
                        {"at": time.time(), "attempt": attempt, "backoff": backoff})
                    log.warning("[decode %s] stream dropped; reopening in %.0fs (attempt %d)",
                                self.camera_id, backoff, attempt)
                    cap.release()
                    self._stop.wait(backoff)   # interruptible so stop() is prompt
                    cap = self._open()
                    continue
                is_file = not self._is_live() and not str(self.source).isdigit()
                if self.loop_files and is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind: stream footage continuously
                    orig_index = 0
                    continue
                self.ended = True          # webcam exhausted / looping disabled
                break
            attempt = 0                    # a healthy read resets the backoff
            self.last_frame_at = time.time()
            if self.state != CONNECTED:
                # Recovery is announced, not silent — an operator who was told
                # the camera went down has to be told it came back.
                self._set_state(CONNECTED, "stream recovered")
            # VIDEO time from the original frame index so dwell/loiter thresholds
            # are correct regardless of sample rate. Wall-clock fallback if no fps.
            ts = (orig_index / self._fps) if self._fps > 1e-3 else (time.perf_counter() - self._t0)
            with self._lock:
                self._index += 1
                self._latest = Frame(self.camera_id, image, orig_index, ts)
                self._seq += 1
                self._consumed = False
            # Pace each kept frame to target_fps — for FILES, which must play at
            # real time. A live source must NOT be paced this way: HLS (and a
            # reconnecting RTSP camera) hands OpenCV a burst of buffered frames
            # at a segment boundary, and reading one per period consumes that
            # backlog instead of the present, falling further behind live the
            # longer it runs. Measured 26 Aug on a reference HLS CDN: bursts at
            # ~39 fps punctuated by stalls up to 5.5s — pacing turns that into
            # permanent, growing lag. Live sources therefore DRAIN to the live
            # edge instead: grab (decode-free) whatever is already buffered,
            # bounded in both count and time, and keep only the newest.
            if self._is_live():
                dropped = 0
                t_drain = time.perf_counter()
                while dropped < 120 and (time.perf_counter() - t_drain) < 0.015:
                    if not cap.grab():
                        break
                    dropped += 1
                if dropped:
                    ok_new, newest = cap.retrieve()
                    if ok_new:
                        orig_index += dropped
                        self.stale_dropped += dropped
                        ts = (orig_index / self._fps) if self._fps > 1e-3 \
                            else (time.perf_counter() - self._t0)
                        with self._lock:
                            self._latest = Frame(self.camera_id, newest, orig_index, ts)
                            self._seq += 1
                            self._consumed = False
            elif self._min_period:
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

    def peek_latest(self) -> tuple:
        """(frame, seq) WITHOUT consuming — the smooth-publish path reads here.

        read_latest() marks frames consumed for the DETECTION loop; a second
        consumer would steal frames from the models. The live wall doesn't need
        to own the frame, only to see the newest one, so it peeks. `seq` lets
        the publisher skip frames it has already shipped.
        """
        with self._lock:
            return self._latest, self._seq

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
