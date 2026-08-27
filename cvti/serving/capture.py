"""One way to open a video source, on every platform (27 Aug).

Three call sites each opened captures their own way, and all three inherited
OpenCV's defaults — which differ per platform in ways that matter:

- **Buffering.** OpenCV queues frames internally. On a live source that queue
  IS latency: you decode the past while the present piles up behind it. The
  queue is deeper on Windows than macOS, which is why the same code feels
  laggier there. `CAP_PROP_BUFFERSIZE = 1` asks for the present.
- **Webcams on Windows** default to the MSMF backend, which routinely takes
  several seconds to open a camera and negotiates poor frame rates. DirectShow
  opens immediately and is what every Windows CV app uses.
- **Network streams** need explicit FFmpeg plus low-latency flags, or they
  buffer whole segments before yielding a frame (HLS measured at 22.5s to
  first frame, with stalls to 5.5s).

Everything here degrades safely: an unsupported property or backend falls back
to OpenCV's default rather than failing to open a camera.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Low-latency FFmpeg options for live network sources. `nobuffer` and
# `low_delay` stop the demuxer holding frames back; `reorder_queue_size=0`
# stops RTSP reordering adding a queue of its own; the TCP transport is what
# most cameras and every NAT actually work with.
_FFMPEG_LIVE_OPTS = (
    "rtsp_transport;tcp"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|reorder_queue_size;0"
)


def is_live_source(source: Any) -> bool:
    """A network stream (rtsp/http/https) — as opposed to a file or a webcam."""
    s = str(source)
    return not s.isdigit() and "://" in s


def open_capture(source: Any, *, low_latency: bool = True):
    """Open `source` with the right backend and the least buffering.

    `low_latency=False` for offline work (evaluation over files) where every
    frame matters more than freshness.
    """
    import cv2

    src = int(source) if str(source).isdigit() else source

    if isinstance(src, int):
        # Webcam. DirectShow on Windows: MSMF can take seconds to open and
        # often negotiates a worse mode. Everywhere else the default is right.
        if sys.platform == "win32":
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened():
                log.warning("DirectShow could not open webcam %s; falling back", src)
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
    elif is_live_source(src):
        if low_latency:
            # setdefault so an operator's explicit env var always wins.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _FFMPEG_LIVE_OPTS)
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(src)          # a file: let OpenCV choose

    if low_latency and (isinstance(src, int) or is_live_source(src)):
        try:
            # Ask for a queue of one. Not every backend honours it; the
            # decoder's live-edge drain covers the ones that do not.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - a tuning hint must never break capture
            log.debug("BUFFERSIZE not supported by this backend", exc_info=True)
    return cap
