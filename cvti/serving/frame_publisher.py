"""Publish the engine's frames so the UI doesn't decode the same video twice.

The engine and the app are separate processes. Until now each opened every camera
independently: the engine to detect, the app's live wall to display. That doubled
decode cost (the dominant per-camera cost) and capped the wall at ~4 cameras — and
the app, having no detection state, could never draw a box on anyone.

The engine already decodes every stream AND knows where every tracked person is,
so it publishes instead: the latest frame per camera, optionally with the tracked
boxes drawn on, over a localhost HTTP port. The app just fetches images.

    GET /frame/<camera_id>   -> image/jpeg  (latest, boxes drawn)
    GET /cameras             -> {"cameras": [...], "tracks": {...}}

The port is written to <output_dir>/frames.json so the app can find it without
being told. Loopback only — frames never leave the box.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

# Boxes are drawn in the engine (it owns the tracks), so the UI stays a dumb viewer.
_BOX_COLOUR = (0, 200, 255)      # BGR amber for a normal tracked person
_ALERT_COLOUR = (60, 60, 220)    # red once that track is part of an active alert


class FramePublisher:
    """Holds the latest JPEG per camera and serves it over loopback HTTP."""

    def __init__(self, *, quality: int = 70, draw_boxes: bool = True,
                 max_width: int = 640) -> None:
        self.quality = quality
        self.draw_boxes = draw_boxes
        self.max_width = max_width
        self._frames: dict[str, bytes] = {}
        self._tracks: dict[str, list] = {}
        self._alerting: dict[str, set] = {}
        self._lock = threading.Lock()
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self.port = 0
        self.published = 0

    # --- engine side ------------------------------------------------------
    def publish(self, camera_id: str, frame: Any, boxes: list | None = None) -> None:
        """Store the latest frame for a camera. `boxes`: [(track_id, x1,y1,x2,y2), ...]"""
        import cv2
        img = frame
        h, w = img.shape[:2]
        if self.max_width and w > self.max_width:      # the wall is small; don't ship 1080p
            scale = self.max_width / float(w)
            img = cv2.resize(img, (self.max_width, max(1, int(h * scale))))
        else:
            scale = 1.0
        if self.draw_boxes and boxes:
            img = img.copy()
            alerting = self._alerting.get(camera_id, set())
            for tid, x1, y1, x2, y2 in boxes:
                col = _ALERT_COLOUR if tid in alerting else _BOX_COLOUR
                p1 = (int(x1 * scale), int(y1 * scale))
                p2 = (int(x2 * scale), int(y2 * scale))
                cv2.rectangle(img, p1, p2, col, 2)
                cv2.putText(img, f"#{tid}", (p1[0] + 3, max(12, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self._lock:
            self._frames[camera_id] = buf.tobytes()
            self._tracks[camera_id] = [int(b[0]) for b in (boxes or [])]
            self.published += 1

    def mark_alerting(self, camera_id: str, track_ids) -> None:
        """Tracks to draw in alert colour (cleared by passing an empty set)."""
        with self._lock:
            self._alerting[camera_id] = set(track_ids or ())

    def frame(self, camera_id: str) -> bytes | None:
        with self._lock:
            return self._frames.get(camera_id)

    def snapshot(self) -> dict:
        with self._lock:
            return {"cameras": sorted(self._frames), "tracks": dict(self._tracks),
                    "published": self.published}

    # --- server -----------------------------------------------------------
    def start(self, output_dir: str | Path | None = None) -> "FramePublisher":
        pub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):        # keep the engine log readable
                return

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass                        # viewer navigated away mid-write

            def do_GET(self):                   # noqa: N802 - BaseHTTPRequestHandler API
                path = self.path.split("?", 1)[0]
                if path.startswith("/frame/"):
                    cam = unquote(path[len("/frame/"):])
                    data = pub.frame(cam)
                    if data is None:
                        self._send(404, b"no frame", "text/plain")
                    else:
                        self._send(200, data, "image/jpeg")
                elif path in ("/cameras", "/"):
                    self._send(200, json.dumps(pub.snapshot()).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="frame-publisher", daemon=True)
        self._thread.start()
        if output_dir:                          # so the app can find us
            p = Path(output_dir) / "frames.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"port": self.port}))
        print(f"[frames] publishing on http://127.0.0.1:{self.port} "
              f"(boxes={'on' if self.draw_boxes else 'off'})")
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
