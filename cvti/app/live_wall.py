"""Live wall — decode several video sources concurrently and keep the latest
JPEG frame of each, for the app's multi-camera grid.

Qt-free (OpenCV + threads only) so it runs in the app process without the
detection stack, and is unit-testable headless. One daemon thread per source
reads frames in a loop (file sources loop at EOF so they play "live"),
downscales, JPEG-encodes, and stores the latest as a base64 data URI the web UI
can drop straight into an <img src>.
"""
from __future__ import annotations

import base64
import os
import threading
import time

import cv2


class LiveWall:
    def __init__(self, sources: list[dict], width: int = 400, fps: float = 8.0,
                 quality: int = 65) -> None:
        # sources: [{"id": str, "source": str|int}, ...]
        self.sources = sources
        self.width = width
        self.interval = 1.0 / max(1.0, fps)
        self.quality = quality
        self._latest: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> "LiveWall":
        self._stop.clear()
        for s in self.sources:
            t = threading.Thread(target=self._decode, args=(s["id"], s["source"]),
                                 name=f"live-{s['id']}", daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def _open(self, source):
        # webcam indices arrive as strings like "0"
        if isinstance(source, str) and source.isdigit():
            return cv2.VideoCapture(int(source))
        if isinstance(source, str) and source.lower().startswith("rtsp"):
            # RTSP over TCP is far more reliable than the UDP default (no tearing/drops)
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        return cv2.VideoCapture(source)

    def _decode(self, cam_id: str, source) -> None:
        cap = self._open(source)
        is_file = not (isinstance(source, int) or (isinstance(source, str) and source.isdigit()))
        n = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_file:                       # loop the clip so it plays continuously
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok:
                    self._set(cam_id, ok=False, error="stream ended")
                    self._stop.wait(0.3)
                    continue
            n += 1
            frame = self._downscale(frame)
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok2:
                # Store RAW jpeg bytes — the frame server streams these natively
                # over localhost so the browser never marshals base64 through
                # QWebChannel (that was the FPS ceiling).
                self._set(cam_id, jpeg=buf.tobytes(), w=int(frame.shape[1]),
                          h=int(frame.shape[0]), frame=n, ok=True)
            self._stop.wait(self.interval)
        cap.release()

    def _downscale(self, frame):
        h, w = frame.shape[:2]
        if w > self.width:
            nh = max(1, int(h * self.width / w))
            frame = cv2.resize(frame, (self.width, nh))
        return frame

    def _set(self, cam_id: str, **kw) -> None:
        with self._lock:
            rec = self._latest.setdefault(cam_id, {})
            rec.update(kw)

    def frames(self) -> dict:
        # Metadata only (no image bytes) — small + cheap to poll for status.
        with self._lock:
            return {k: {kk: vv for kk, vv in v.items() if kk != "jpeg"} for k, v in self._latest.items()}

    def jpeg(self, cam_id: str) -> bytes | None:
        with self._lock:
            rec = self._latest.get(cam_id)
            return rec.get("jpeg") if rec else None

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.5)
        self._threads = []


class FrameServer:
    """Serves the LiveWall's latest JPEG per camera over localhost, so the UI's
    <img> tags fetch frames natively (fast) instead of base64-over-QWebChannel."""

    def __init__(self, wall: "LiveWall") -> None:
        self.wall = wall
        self.port = 0
        self._httpd = None
        self._thread = None

    def start(self) -> int:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        wall = self.wall

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence request logging
                pass

            def do_GET(self):
                cam = self.path.split("?", 1)[0].rsplit("/", 1)[-1]
                jpg = wall.jpeg(cam)
                if not jpg:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                try:
                    self.wfile.write(jpg)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)   # port 0 = OS picks
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="frame-server", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
