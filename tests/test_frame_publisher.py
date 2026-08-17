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
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.serving.frame_publisher import FramePublisher

JPEG_MAGIC = b"\xff\xd8"


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.pub = FramePublisher(max_width=320).start(self.d)
        self.frame = np.full((480, 640, 3), 60, np.uint8)

    def tearDown(self):
        self.pub.stop()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.pub.port}{path}", timeout=3) as r:
            return r.read()

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


class AppFallbackTests(unittest.TestCase):
    def test_app_decodes_for_itself_when_no_engine_is_publishing(self):
        from cvti.app.console_backend import ConsoleBackend
        d = Path(tempfile.mkdtemp())
        be = ConsoleBackend(site_path=str(d / "s.json"), db_path=str(d / "e.db"),
                            enable_demo=False)
        self.assertEqual(be._engine_frame_port(), 0)      # nothing published
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
        be = ConsoleBackend(site_path=str(d / "s.json"), db_path=str(d / "e.db"),
                            enable_demo=False)
        self.assertEqual(be._engine_frame_port(), 0)


if __name__ == "__main__":
    unittest.main()
