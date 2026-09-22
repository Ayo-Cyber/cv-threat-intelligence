"""The engine can prove, by itself, that its evidence clips will play.

Replays were black boxes on the pilot's Windows machine because the OpenCV
wheel there has no H.264 encoder and Chromium cannot decode the mp4v it
falls back to. The fix bundles ffmpeg — but from a Mac nobody could say
whether the bundled copy actually ran on Windows. `--check-clip-encoder`
writes one synthetic clip the way evidence is written and reports the codec;
the bundle smoke and the Windows test job call it on every build.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving import alert_sink


def _can_decode(path: Path) -> bool:
    import cv2
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    return bool(ok)


class ProbeWritesAPlayableClip(unittest.TestCase):
    def test_probe_reports_h264_when_ffmpeg_is_available(self):
        if alert_sink._ffmpeg_exe() is None:
            self.skipTest("imageio-ffmpeg not installed here")
        with tempfile.TemporaryDirectory() as d:
            got = alert_sink.probe_clip_encoder(d)
            self.assertTrue(Path(got["path"]).exists(), got)
            if not _can_decode(Path(got["path"])):
                self.skipTest("this OpenCV build cannot read back what it wrote")
            self.assertEqual(got["codec"], "h264", got)
            self.assertTrue(got["ffmpeg"])

    def test_probe_names_the_codec_it_could_not_fix(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(alert_sink, "_ffmpeg_exe", return_value=None), \
                mock.patch.object(alert_sink, "_clip_codec", return_value="mp4v"):
            got = alert_sink.probe_clip_encoder(d)
            self.assertEqual(got["codec"], "mp4v")
            self.assertIsNone(got["ffmpeg"])


class TheEngineFlagExitsWithTheVerdict(unittest.TestCase):
    def _run(self, result: dict) -> tuple[int, str]:
        import io
        from contextlib import redirect_stdout

        from cvti.serving import pipeline
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["argus-engine", "--check-clip-encoder"]), \
                mock.patch.object(alert_sink, "probe_clip_encoder", return_value=result), \
                redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            pipeline.main()
        return int(cm.exception.code or 0), buf.getvalue()

    def test_h264_is_exit_zero(self):
        code, out = self._run({"codec": "h264", "ffmpeg": "/x/ffmpeg", "path": "/x/c.mp4"})
        self.assertEqual(code, 0)
        self.assertIn("clip encoder: h264", out)

    def test_anything_else_fails_loudly(self):
        code, out = self._run({"codec": "mp4v", "ffmpeg": None, "path": "/x/c.mp4"})
        self.assertEqual(code, 1)
        self.assertIn("would not play", out)
        self.assertIn("not bundled", out)


if __name__ == "__main__":
    unittest.main()
