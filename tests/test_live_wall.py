from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from _backend_helper import signed_in

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CLIPS = sorted((ROOT / "data" / "test_clips").glob("*.mp4"))


class PreviewPacingTests(unittest.TestCase):
    def test_network_reads_are_not_throttled_but_jpeg_encoding_is(self):
        import numpy as np
        from cvti.app.live_wall import LiveWall
        wall = LiveWall([], fps=8)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        capture = Mock()
        reads = []

        def read():
            reads.append(1)
            if len(reads) == 4:
                wall._stop.set()
            return True, image

        capture.read.side_effect = read
        with patch.object(wall, "_open", return_value=capture), \
                patch("cvti.app.live_wall.time.monotonic", side_effect=[0, .04, .08, .16]), \
                patch("cvti.app.live_wall.cv2.imencode", return_value=(True, np.array([1]))) as encode, \
                patch.object(wall._stop, "wait") as wait:
            wall._decode("camera", "rtsp://example/live")
        self.assertEqual(len(reads), 4)
        self.assertEqual(encode.call_count, 2)
        wait.assert_not_called()
        capture.release.assert_called()

    def test_local_files_keep_playback_pacing(self):
        import numpy as np
        from cvti.app.live_wall import LiveWall
        wall = LiveWall([], fps=8)
        capture = Mock()
        capture.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
        with patch.object(wall, "_open", return_value=capture), \
                patch.object(wall._stop, "wait", side_effect=lambda _: wall._stop.set()) as wait:
            wall._decode("camera", "example.mp4")
        wait.assert_called_once_with(.125)


@unittest.skipUnless(CLIPS, "no test clips present")
class LiveWallTests(unittest.TestCase):
    def test_decodes_multiple_sources(self):
        from cvti.app.live_wall import LiveWall
        srcs = [{"id": p.stem, "source": str(p)} for p in CLIPS[:2]]
        wall = LiveWall(srcs, width=240, fps=10).start()
        try:
            # give the decoder threads a beat to produce at least one frame each
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                f = wall.frames()
                if len(f) == len(srcs) and all(wall.jpeg(k) for k in f):
                    break
                time.sleep(0.1)
            f = wall.frames()
            self.assertEqual(set(f), {s["id"] for s in srcs})
            for k, v in f.items():
                self.assertNotIn("jpeg", v)                      # metadata only, no bytes
                self.assertGreater(v["frame"], 0)
                self.assertLessEqual(v["w"], 240)
                self.assertEqual(wall.jpeg(k)[:2], b"\xff\xd8")   # raw JPEG bytes served
        finally:
            wall.stop()

    def test_backend_live_cycle(self):
        import tempfile
        from cvti.app.console_backend import ConsoleBackend
        with tempfile.TemporaryDirectory() as d:
            be = signed_in(site_path=str(Path(d) / "s.json"), db_path=str(Path(d) / "e.db"), enable_demo=False)
            res = be.live_start(count=2)               # empty site -> demo clips fallback
            self.assertGreaterEqual(len(res["cameras"]), 1)
            self.assertGreater(res["port"], 0)         # frame server started
            time.sleep(1.5)
            frames = be.live_frames()
            self.assertTrue(any(v.get("frame", 0) > 0 for v in frames.values()))
            self.assertEqual(be.live_stop(), {"stopped": True})


if __name__ == "__main__":
    unittest.main()
