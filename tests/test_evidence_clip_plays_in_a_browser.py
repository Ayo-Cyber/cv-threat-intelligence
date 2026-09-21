"""An evidence clip must be H.264, or the app's own player cannot show it.

OpenCV tries avc1 and falls back to mp4v. The opencv-python-headless wheel
that every build installs has no H.264 encoder on Windows, so pilot clips
were MPEG-4 Part 2 -- which Chromium, and therefore the Electron renderer,
cannot decode in <video>. Developer Macs get an OS encoder and never saw it
("the replay videos weren't playing", Windows, 21 Sep). Telegram's player
coped, which is why phones showed video and the app did not.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cvti.serving import alert_sink


def _frames(n: int = 6):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
    return [np.roll(base, i * 3, axis=1) for i in range(n)]


class EvidenceClipCodec(unittest.TestCase):
    def test_a_written_clip_ends_up_h264_when_ffmpeg_is_present(self):
        if alert_sink._ffmpeg_exe() is None:
            self.skipTest("imageio-ffmpeg not installed here")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "clip.mp4"
            sink = alert_sink.AlertSink.__new__(alert_sink.AlertSink)
            sink._write_clip(path, _frames(), fps=12)
            self.assertTrue(path.exists() and path.stat().st_size > 0)
            self.assertEqual(alert_sink._clip_codec(path), "h264")

    def test_an_mp4v_clip_is_transcoded_not_left_behind(self):
        """Force the fallback path a Windows build takes, then repair it."""
        if alert_sink._ffmpeg_exe() is None:
            self.skipTest("imageio-ffmpeg not installed here")
        import cv2
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "clip.mp4"
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (128, 96))
            self.assertTrue(vw.isOpened(), "this platform cannot even write mp4v")
            for f in _frames():
                vw.write(f)
            vw.release()
            self.assertEqual(alert_sink._clip_codec(path), "mp4v")
            self.assertEqual(alert_sink.ensure_h264(path), "h264")
            self.assertEqual(alert_sink._clip_codec(path), "h264")
            self.assertFalse(list(Path(d).glob("*.tmp.mp4")), "no temp file left over")

    def test_without_ffmpeg_the_clip_survives_and_says_so(self):
        """Best effort: never delete evidence because a transcode was unavailable."""
        import cv2
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "clip.mp4"
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (128, 96))
            for f in _frames():
                vw.write(f)
            vw.release()
            before = path.stat().st_size
            with mock.patch.object(alert_sink, "_ffmpeg_exe", return_value=None):
                with self.assertLogs("cvti.serving.alert_sink", level="WARNING") as caught:
                    codec = alert_sink.ensure_h264(path)
            self.assertEqual(codec, "mp4v")
            self.assertEqual(path.stat().st_size, before, "the original must be untouched")
            self.assertIn("no ffmpeg", " ".join(caught.output))


if __name__ == "__main__":
    unittest.main()
