"""Viewer-owned camera captures, independent of the inference engine."""
from __future__ import annotations

import threading
from urllib.parse import quote, urlencode

from cvti.app.live_wall import FrameServer, LiveWall


class CameraPreview:
    def __init__(self):
        self._lock = threading.RLock()
        self._sources = {}
        self._walls = {}
        self._viewers = {}
        self._closed = False
        self._server = FrameServer(self)
        self._server.start()

    def descriptor(self, camera_id, source):
        with self._lock:
            if self._closed:
                raise RuntimeError("preview is closing")
            if camera_id in self._sources and self._sources[camera_id] != source:
                raise RuntimeError("camera source changed; reopen preview")
            self._sources[camera_id] = source
            return {"kind": "mjpeg", "preview": True,
                    "url": f"http://127.0.0.1:{self._server.port}/stream/"
                    f"{quote(camera_id, safe='')}?{urlencode({'token': self._server.token})}"}

    def acquire(self, camera_id):
        with self._lock:
            if self._closed or camera_id not in self._sources:
                return False
            if camera_id not in self._walls:
                # Preserve source coordinates for the zone editor's snapshots.
                self._walls[camera_id] = LiveWall(
                    [{"id": camera_id, "source": self._sources[camera_id]}],
                    width=100000, fps=8, quality=80).start()
            self._viewers[camera_id] = self._viewers.get(camera_id, 0) + 1
            return True

    def release(self, camera_id):
        with self._lock:
            self._viewers[camera_id] = max(0, self._viewers.get(camera_id, 0) - 1)
            if not self._viewers[camera_id] and camera_id in self._walls:
                wall = self._walls[camera_id]
                wall.stop()
                if not any(t.is_alive() for t in wall._threads):
                    del self._walls[camera_id]

    def jpeg(self, camera_id):
        with self._lock:
            wall = self._walls.get(camera_id)
            return wall.jpeg(camera_id) if wall and not self._closed else None

    def snapshot(self, camera_id):
        with self._lock:
            wall = self._walls.get(camera_id)
            if wall is None:
                return None
            import base64
            jpg = wall.jpeg(camera_id)
            meta = wall.frames().get(camera_id, {})
            if not jpg:
                return {"error": "preview is connecting; retry the snapshot shortly"}
            return {"uri": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(),
                    "w": meta["w"], "h": meta["h"]}

    def close(self):
        with self._lock:
            self._closed = True
            for wall in self._walls.values():
                wall.stop()
            stopped = all(not t.is_alive() for wall in self._walls.values()
                          for t in wall._threads)
        self._server.stop()
        if not stopped:
            raise RuntimeError("Camera preview is still releasing its capture; retry monitoring shortly")
