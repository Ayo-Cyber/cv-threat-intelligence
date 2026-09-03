"""Decode rides the GPU where one exists (3 Sep, 'the RTSP is slow').

A CPU-only box decoding a 1080p RTSP main stream burns cores its integrated
GPU could spare — D3D11 on Windows, VideoToolbox on macOS. The capture layer
asks for VIDEO_ACCELERATION_ANY, which negotiates whatever exists and falls
back to software by itself; ancient OpenCV builds without the property keep
today's behaviour untouched.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.capture import _hw_decode_params, open_capture


class HwDecodeTests(unittest.TestCase):
    def test_params_request_any_acceleration_when_supported(self):
        import cv2
        params = _hw_decode_params(cv2)
        if hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
            self.assertEqual(params,
                             [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        else:                                     # ancient build: explicit no-op
            self.assertEqual(params, [])

    def test_an_opencv_without_the_property_is_a_noop(self):
        class AncientCv2:                          # no HW attributes at all
            pass
        self.assertEqual(_hw_decode_params(AncientCv2()), [])

    def test_a_file_still_opens_and_reads_through_the_hw_path(self):
        """The upgrade must never cost a camera: open a real clip through
        open_capture and prove frames come out."""
        clip = Path("data/test_clips/theft_shop_01.mp4")
        if not clip.exists():
            self.skipTest("demo clip not present")
        cap = open_capture(str(clip))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        self.assertTrue(ok)
        self.assertGreater(frame.shape[0], 0)

    def test_live_open_passes_the_hw_params(self):
        import inspect
        from cvti.serving import capture
        src = inspect.getsource(capture.open_capture)
        self.assertIn("cv2.CAP_FFMPEG, hw", src)
        self.assertIn("retrying software", src)   # broken-driver fallback stays


if __name__ == "__main__":
    unittest.main()
