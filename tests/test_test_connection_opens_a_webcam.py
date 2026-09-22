"""The Test connection button opens a webcam the way the engine does.

"0 for webcam" reached cv2 as the string "0" — a file path to OpenCV — so
the button said "Could not open" for every webcam on every OS while Add
camera + monitoring worked. Seen live on a Windows screen share, 22 Sep,
with camera permissions all green.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving import onboarding


class _Cap:
    def __init__(self, opened=True, frames=None):
        self._opened = opened
        self._frames = list(frames if frames is not None else [np.zeros((48, 64, 3), np.uint8)])

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        f = self._frames.pop(0)
        return (f is not None), f

    def get(self, _prop):
        return 30.0

    def release(self):
        pass


class _Cv2:
    CAP_DSHOW = 700
    CAP_PROP_FPS = 5
    INTER_AREA = 3

    def __init__(self, caps):
        self.calls = []
        self._caps = list(caps)

    def VideoCapture(self, *args):
        self.calls.append(args)
        return self._caps.pop(0)

    def resize(self, frame, size, **kw):
        return frame

    def imencode(self, ext, frame):
        return True, np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8)


class TestConnectionWebcam(unittest.TestCase):
    def _run(self, source, cv2, platform="linux"):
        with mock.patch.dict(sys.modules, {"cv2": cv2}), \
                mock.patch.object(onboarding.sys, "platform", platform):
            return onboarding.test_url(source)

    def test_a_digit_source_is_a_device_index_not_a_path(self):
        cv2 = _Cv2([_Cap()])
        out = self._run("0", cv2)
        self.assertTrue(out.get("ok"), out)
        self.assertEqual(cv2.calls, [(0,)])                     # int, not "0"

    def test_windows_tries_directshow_first(self):
        cv2 = _Cv2([_Cap()])
        out = self._run("0", cv2, platform="win32")
        self.assertTrue(out.get("ok"), out)
        self.assertEqual(cv2.calls, [(0, _Cv2.CAP_DSHOW)])

    def test_windows_falls_back_to_the_default_backend(self):
        cv2 = _Cv2([_Cap(opened=False), _Cap()])
        out = self._run("1", cv2, platform="win32")
        self.assertTrue(out.get("ok"), out)
        self.assertEqual(cv2.calls, [(1, _Cv2.CAP_DSHOW), (1,)])

    def test_a_webcam_that_needs_a_moment_still_passes(self):
        cv2 = _Cv2([_Cap(frames=[None, None, None, np.zeros((48, 64, 3), np.uint8)])])
        out = self._run("0", cv2)
        self.assertTrue(out.get("ok"), out)

    def test_a_busy_webcam_is_named_with_the_fix(self):
        cv2 = _Cv2([_Cap(opened=False)])
        out = self._run("0", cv2)
        self.assertFalse(out.get("ok"))
        self.assertIn("webcam 0", out["error"])
        self.assertIn("Meet", out["error"])

    def test_urls_are_untouched(self):
        cv2 = _Cv2([_Cap()])
        out = self._run("rtsp://user:pw@10.0.0.5:554/stream1", cv2, platform="win32")
        self.assertTrue(out.get("ok"), out)
        self.assertEqual(cv2.calls, [("rtsp://user:pw@10.0.0.5:554/stream1",)])


if __name__ == "__main__":
    unittest.main()
