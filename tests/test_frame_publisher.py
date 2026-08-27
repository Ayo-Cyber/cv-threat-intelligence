"""The engine publishes frames so the UI never decodes the same stream twice.

Decode is the dominant per-camera cost, and the app used to pay it a second time
just to display. It also had no detection state, so it could never draw a box.
These cover the transport and the fallback — the app must still work when no
engine is running.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.serving.frame_publisher import FramePublisher

from _backend_helper import signed_in

JPEG_MAGIC = b"\xff\xd8"


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.pub = FramePublisher(max_width=320).start(self.d)
        self.frame = np.full((480, 640, 3), 60, np.uint8)

    def tearDown(self):
        self.pub.stop()

    def _get(self, path, token=None):
        """Every route authenticates — this serves live camera frames."""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.pub.port}{path}",
            headers={"X-Argus-Token": self.pub.token if token is None else token})
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.read()

    def _status(self, path, token=None):
        url = f"http://127.0.0.1:{self.pub.port}{path}"
        headers = {} if token is None else {"X-Argus-Token": token}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=3) as r:
                return r.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_serves_the_latest_frame_as_jpeg(self):
        self.pub.publish("cam1", self.frame, [(7, 100, 50, 200, 300)])
        body = self._get("/frame/cam1")
        self.assertEqual(body[:2], JPEG_MAGIC)

    def test_publishes_port_so_the_app_can_find_it(self):
        self.assertEqual(json.loads((self.d / "frames.json").read_text())["port"], self.pub.port)

    def test_cameras_endpoint_lists_cameras_and_tracks(self):
        self.pub.publish("cam1", self.frame, [(7, 10, 10, 50, 90), (9, 60, 10, 90, 90)])
        meta = json.loads(self._get("/cameras"))
        self.assertEqual(meta["cameras"], ["cam1"])
        self.assertEqual(meta["tracks"]["cam1"], [7, 9])

    def test_unknown_camera_is_404_not_a_crash(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/frame/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_boxes_change_the_image_and_never_mutate_the_source(self):
        original = self.frame.copy()
        self.pub.publish("a", self.frame, [(1, 10, 10, 100, 200)])
        with_box = self.pub.frame("a")
        self.pub.draw_boxes = False
        self.pub.publish("a", self.frame, [(1, 10, 10, 100, 200)])
        self.assertNotEqual(with_box, self.pub.frame("a"))
        self.assertTrue(np.array_equal(self.frame, original))

    def test_large_frames_are_downscaled_for_the_wall(self):
        self.pub.publish("big", np.full((1080, 1920, 3), 90, np.uint8), [])
        import cv2
        img = cv2.imdecode(np.frombuffer(self.pub.frame("big"), np.uint8), cv2.IMREAD_COLOR)
        self.assertLessEqual(img.shape[1], 320)

    def test_alerting_tracks_are_coloured_differently(self):
        self.pub.publish("c", self.frame, [(3, 20, 20, 120, 220)])
        normal = self.pub.frame("c")
        self.pub.mark_alerting("c", {3})
        self.pub.publish("c", self.frame, [(3, 20, 20, 120, 220)])
        self.assertNotEqual(normal, self.pub.frame("c"))


    # --- EP-03-T1: no unauthenticated route to a camera ---------------------
    def test_every_route_rejects_an_unauthenticated_request(self):
        self.pub.publish("cam1", self.frame, [])
        for path in ("/frame/cam1", "/cameras", "/", "/frame/nonexistent", "/anything"):
            self.assertEqual(self._status(path), 401,
                             f"{path} served without a token")

    def test_a_wrong_token_is_rejected(self):
        self.pub.publish("cam1", self.frame, [])
        self.assertEqual(self._status("/frame/cam1", "not-the-token"), 401)

    def test_the_token_also_works_as_a_query_parameter(self):
        # An <img> tag cannot set a header.
        self.pub.publish("cam1", self.frame, [])
        self.assertEqual(self._status(f"/frame/cam1?token={self.pub.token}"), 200)

    def test_the_token_is_random_per_publisher(self):
        from cvti.serving.frame_publisher import FramePublisher
        self.assertNotEqual(FramePublisher().token, FramePublisher().token)

    def test_frames_json_carries_the_token_and_is_not_world_readable(self):
        import json
        import os
        import tempfile
        from pathlib import Path

        from cvti.serving.frame_publisher import FramePublisher
        with tempfile.TemporaryDirectory() as tmp:
            pub = FramePublisher().start(tmp)
            try:
                info = json.loads((Path(tmp) / "frames.json").read_text())
                self.assertEqual(info["token"], pub.token)
                mode = os.stat(Path(tmp) / "frames.json").st_mode & 0o777
                self.assertEqual(mode, 0o600, "the frame token was world-readable")
            finally:
                pub.stop()

    def test_no_wildcard_cors_on_camera_frames(self):
        # Letting any origin read these is the browser-side open port.
        self.pub.publish("cam1", self.frame, [])
        req = urllib.request.Request(f"http://127.0.0.1:{self.pub.port}/frame/cam1",
                                     headers={"X-Argus-Token": self.pub.token})
        with urllib.request.urlopen(req, timeout=3) as r:
            self.assertIsNone(r.headers.get("Access-Control-Allow-Origin"))

class AppFallbackTests(unittest.TestCase):
    def test_app_decodes_for_itself_when_no_engine_is_publishing(self):
        from cvti.app.console_backend import ConsoleBackend
        d = Path(tempfile.mkdtemp())
        be = signed_in(site_path=str(d / "s.json"), db_path=str(d / "e.db"),
                            enable_demo=False)
        self.assertEqual(be._engine_frame_port(), (0, ""))   # nothing published
        res = be.live_start(2)
        try:
            # falls back to its own decode rather than showing nothing
            self.assertIn(res.get("source"), ("app", None))
        finally:
            be.live_stop()

    def test_stale_port_file_is_not_trusted(self):
        from cvti.app.console_backend import ConsoleBackend
        d = Path(tempfile.mkdtemp())
        (d / "frames.json").write_text(json.dumps({"port": 9}))   # nothing listening
        be = signed_in(site_path=str(d / "s.json"), db_path=str(d / "e.db"),
                            enable_demo=False)
        self.assertEqual(be._engine_frame_port(), (0, ""))


if __name__ == "__main__":
    unittest.main()


class TokenThreadingTests(unittest.TestCase):
    """The demo-day broken-tile bug: EP-03 put a token on every frame route,
    but live_start's probe checked /cameras WITHOUT it (so engine frames were
    never used) and neither return path handed the UI a token (so every <img>
    got 401). The UI cannot render what it cannot authenticate to."""

    def test_engine_probe_sends_the_token_and_returns_it(self):
        from cvti.serving.frame_publisher import FramePublisher
        import numpy as np
        d = Path(tempfile.mkdtemp())
        pub = FramePublisher().start(d)
        try:
            pub.publish("cam1", np.zeros((8, 8, 3), np.uint8), [])
            be = signed_in(site_path=str(d / "s.json"), db_path=str(d / "e.db"),
                           enable_demo=False)
            port, token = be._engine_frame_port()
            self.assertEqual(port, pub.port)
            self.assertEqual(token, pub.token, "the UI is useless without the token")
        finally:
            pub.stop()

    def test_live_start_app_path_returns_the_frameserver_token(self):
        import inspect
        from cvti.app.console_backend import ConsoleBackend
        src = inspect.getsource(ConsoleBackend.live_start)
        self.assertIn('"token"', src)
        self.assertIn("self._fs.token", src)


class MjpegStreamTests(unittest.TestCase):
    """The live wall streams instead of polling (26 Aug). The UI used to fire
    11 requests/second/camera at /frame — a fresh HTTP round-trip, JPEG decode
    and repaint each time, which is what 'the stream is sluggish' was."""

    def setUp(self):
        import numpy as np
        from cvti.serving.frame_publisher import FramePublisher
        self.d = Path(tempfile.mkdtemp())
        self.pub = FramePublisher(max_width=64).start(self.d)
        self.np = np
        self.addCleanup(self.pub.stop)

    def _url(self, path):
        return f"http://127.0.0.1:{self.pub.port}{path}?token={self.pub.token}"

    def test_the_stream_pushes_successive_frames(self):
        import threading
        import urllib.request
        self.pub.publish("cam1", self.np.zeros((48, 64, 3), self.np.uint8))

        chunks = []

        def read_some():
            r = urllib.request.urlopen(self._url("/stream/cam1"), timeout=10)
            self.assertIn("multipart/x-mixed-replace", r.headers["Content-Type"])
            deadline = time.time() + 6
            while time.time() < deadline and len(chunks) < 2:
                data = r.read(4096)
                if not data:
                    break
                if b"\xff\xd8" in data:
                    chunks.append(data)
            r.close()

        t = threading.Thread(target=read_some, daemon=True)
        t.start()
        for i in range(1, 12):          # keep publishing new frames
            time.sleep(0.15)
            img = self.np.full((48, 64, 3), i * 10, self.np.uint8)
            self.pub.publish("cam1", img)
        t.join(timeout=8)
        self.assertGreaterEqual(len(chunks), 2,
                                "the MJPEG stream sent fewer than two frames")

    def test_the_stream_is_authenticated_like_every_other_route(self):
        import urllib.error
        import urllib.request
        self.pub.publish("cam1", self.np.zeros((48, 64, 3), self.np.uint8))
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.pub.port}/stream/cam1", timeout=5)
        self.assertEqual(ctx.exception.code, 401,
                         "the stream route served video without a token")

    def test_the_stream_speaks_http_1_1(self):
        """Chromium — which QtWebEngine is — will not progressively render
        multipart/x-mixed-replace over HTTP/1.0. Every tile showed a broken
        image while the engine published perfectly (27 Aug). A socket client
        reads it either way, which is why the earlier test missed it."""
        import urllib.request
        self.pub.publish("cam1", self.np.zeros((48, 64, 3), self.np.uint8))
        r = urllib.request.urlopen(self._url("/stream/cam1"), timeout=5)
        try:
            self.assertEqual(r.version, 11,
                             "HTTP/1.0 multipart — a browser will show a broken image")
            self.assertEqual(r.headers.get("Connection", "").lower(), "close",
                             "a response with no end must not be kept alive")
        finally:
            r.close()

    def test_normal_responses_still_carry_content_length(self):
        # HTTP/1.1 without Content-Length would hang a keep-alive client.
        import urllib.request
        r = urllib.request.urlopen(self._url("/cameras"), timeout=5)
        try:
            self.assertTrue(r.headers.get("Content-Length"))
        finally:
            r.close()

    def test_a_publish_bumps_the_sequence(self):
        self.pub.publish("cam1", self.np.zeros((48, 64, 3), self.np.uint8))
        _, s1 = self.pub.frame_seq("cam1")
        self.pub.publish("cam1", self.np.ones((48, 64, 3), self.np.uint8))
        _, s2 = self.pub.frame_seq("cam1")
        self.assertGreater(s2, s1, "streams would never send a second frame")


class UiStreamWiringTests(unittest.TestCase):
    UI = (Path(__file__).resolve().parents[1] / "cvti/app/web/index.html").read_text()

    def test_tiles_use_the_stream_not_a_polled_still(self):
        self.assertIn("streamUrl(c.id)", self.UI)
        live = self.UI.split("function startLivePoll")[1].split("\nfunction ")[0]
        self.assertNotIn("img.src=frameUrl", live,
                         "the wall still re-fetches a still image on a timer")

    def test_leaving_the_wall_closes_the_streams(self):
        stop = self.UI.split("function stopLive()")[1].split("\nfunction ")[0]
        self.assertIn('img.src=""', stop,
                      "an <img> on an MJPEG stream holds the socket (and an "
                      "engine thread) open forever")
