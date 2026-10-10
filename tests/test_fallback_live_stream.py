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


class NetworkStreamRecoversTest(unittest.TestCase):
    """A camera blip must not freeze the fallback forever. Proven live against
    mediamtx (29 Aug): kill the publisher, bring it back — the frame counter
    stayed identical, because a dead VideoCapture never revives on its own and
    nothing reopened it. Same disease the rules scanner had on 23 Aug. Tested
    with a dropped capture here; the live-RTSP proof ran against a real server
    (frame counter 17 -> 58 after recovery)."""

    def test_the_decode_loop_reopens_dropped_network_streams(self):
        from unittest.mock import Mock, patch
        import numpy as np

        wall = LiveWall([])
        failed, recovered = Mock(), Mock()
        failed.read.return_value = (False, None)
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        def read_recovered():
            wall._stop.set()
            return True, image

        recovered.read.side_effect = read_recovered
        clock = [100.0]

        def advance(_delay):
            clock[0] += 4
            if clock[0] > 112:
                wall._stop.set()  # bound the test even if reconnect regresses

        with patch("cvti.serving.capture.open_capture", side_effect=[failed, recovered]) as opener, \
                patch("cvti.app.live_wall.time.time", side_effect=lambda: clock[0]), \
                patch.object(wall._stop, "wait", side_effect=advance):
            wall._decode("cam", "rtsp://camera/live")

        self.assertEqual(opener.call_count, 2)
        opener.assert_called_with("rtsp://camera/live")
        failed.release.assert_called_once()
        self.assertIsNotNone(wall.jpeg("cam"), "no image published after recovery")
        report = wall.diagnostics()["cam"]
        self.assertEqual(report["open_attempts"], 2)
        self.assertGreater(report["read_failures"], 0)
        self.assertEqual(report["state"], "receiving")
