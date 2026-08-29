"""The console's own live view must STREAM, not freeze on one frame.

Field report (29 Aug): 'once I expand the stream it shows for a while and
cut off.' The moving part was the engine's publisher before the engine died;
the frozen part after was this server — the app-side FrameServer answered the
UI's /stream/ URLs with a single static JPEG, because it had no streaming
route at all. Now it speaks real MJPEG over HTTP/1.1, so Watch is live even
before monitoring starts.
"""
from __future__ import annotations

import http.client
import time
import unittest

from cvti.app.live_wall import FrameServer, LiveWall

CLIP = "data/test_clips/normal_street_01.mp4"


class FallbackStreamTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.wall = LiveWall([{"id": "cam1", "source": CLIP}], fps=10).start()
        cls.fs = FrameServer(cls.wall)
        cls.port = cls.fs.start()
        deadline = time.time() + 10
        while cls.wall.jpeg("cam1") is None and time.time() < deadline:
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.fs.stop()
        cls.wall.stop()

    def _get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", path)
        return conn, conn.getresponse()

    def test_stream_delivers_many_frames_not_one(self):
        conn, r = self._get(f"/stream/cam1?token={self.fs.token}")
        self.assertEqual(r.status, 200)
        self.assertIn("multipart/x-mixed-replace", r.getheader("Content-Type", ""))
        buf, frames, deadline = b"", 0, time.time() + 6
        while frames < 4 and time.time() < deadline:
            chunk = r.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                i = buf.find(b"\xff\xd8\xff")
                j = buf.find(b"\xff\xd9", i + 3) if i >= 0 else -1
                if i < 0 or j < 0:
                    break
                frames += 1
                buf = buf[j + 2:]
        conn.close()
        self.assertGreaterEqual(frames, 4,
                                f"got {frames} frame(s) — the fallback froze on a still")

    def test_the_stream_route_still_requires_the_token(self):
        conn, r = self._get("/stream/cam1?token=wrong")
        self.assertEqual(r.status, 401)
        conn.close()

    def test_single_frame_route_survives(self):
        conn, r = self._get(f"/frame/cam1?token={self.fs.token}")
        self.assertEqual(r.status, 200)
        self.assertEqual(r.getheader("Content-Type"), "image/jpeg")
        body = r.read()
        self.assertTrue(body.startswith(b"\xff\xd8"))
        conn.close()

    def test_the_server_speaks_http_11(self):
        conn, r = self._get(f"/frame/cam1?token={self.fs.token}")
        self.assertEqual(r.version, 11, "Chromium will not render multipart over HTTP/1.0")
        conn.close()


if __name__ == "__main__":
    unittest.main()
