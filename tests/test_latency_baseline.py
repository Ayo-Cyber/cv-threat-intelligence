"""The latency yardstick measures the pipeline, not itself (7 Sep).

W1 proposes replacing the MJPEG wall against a 300ms SLO nobody had measured.
tools/latency_baseline.py is the yardstick — and its first run already earned
it: the 810ms it found was OpenCV's decoder frame-threading (one frame of
delay per CPU core), now pinned single-threaded for live sources in
capture.py. These tests hold the yardstick itself honest: the burned clock
must survive exactly what the pipeline does to a frame (H.264's blocking, the
publisher's 640px resize, JPEG q70), must fail closed when compression mangles
it, and the arithmetic must be right.

No ffmpeg here — the synthetic camera needs it, the codec and stats do not.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tools.latency_baseline import (  # noqa: E402
    FRAME_H, FRAME_W, _stats, decode_clock, encode_clock,
)

A_TIME_MS = 1_788_790_000_123        # a real ms-since-epoch, 48 bits in use


def _frame_with_clock(ms: int = A_TIME_MS):
    rng = np.random.default_rng(3)
    frame = rng.integers(40, 200, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    encode_clock(frame, ms)
    return frame


class ClockCodecTests(unittest.TestCase):
    def test_roundtrip_on_a_clean_frame(self):
        self.assertEqual(decode_clock(_frame_with_clock()), A_TIME_MS)

    def test_roundtrip_survives_the_publishers_treatment(self):
        """The wall client sees the frame AFTER publish(): resized to
        max_width=640 and re-encoded JPEG q70 — the exact production path."""
        import cv2
        frame = _frame_with_clock()
        small = cv2.resize(frame, (640, 360))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.assertTrue(ok)
        received = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8),
                                cv2.IMREAD_COLOR)
        self.assertEqual(decode_clock(received), A_TIME_MS)

    def test_roundtrip_survives_jpeg_at_full_size(self):
        import cv2
        ok, buf = cv2.imencode(".jpg", _frame_with_clock(),
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.assertTrue(ok)
        received = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8),
                                cv2.IMREAD_COLOR)
        self.assertEqual(decode_clock(received), A_TIME_MS)

    def test_a_mangled_clock_fails_closed(self):
        """A wrong timestamp poisons every percentile downstream — the
        checksum must turn corruption into a dropped sample, never a number."""
        frame = _frame_with_clock()
        # flip one data block outright (top-left block of the grid)
        y = x = 24 + 12                      # margin + border
        frame[y:y + 36, x:x + 36] = 255 - frame[y + 1, x + 1]
        self.assertIsNone(decode_clock(frame))

    def test_a_frame_without_a_clock_yields_nothing(self):
        rng = np.random.default_rng(9)
        noise = rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.assertIsNone(decode_clock(noise))

    def test_a_frame_too_small_to_read_yields_nothing(self):
        tiny = np.zeros((90, 160, 3), dtype=np.uint8)    # scale 0.125 < floor
        self.assertIsNone(decode_clock(tiny))

    def test_different_times_decode_differently(self):
        self.assertEqual(decode_clock(_frame_with_clock(A_TIME_MS + 1)),
                         A_TIME_MS + 1)


class StatsTests(unittest.TestCase):
    def test_percentiles_and_mean(self):
        s = _stats([float(v) for v in range(1, 101)])
        self.assertEqual(s["count"], 100)
        self.assertAlmostEqual(s["p50_ms"], 51.0, delta=1.5)
        self.assertAlmostEqual(s["p95_ms"], 95.0, delta=1.5)
        self.assertEqual(s["max_ms"], 100.0)

    def test_empty_input_reports_zero_not_a_crash(self):
        self.assertEqual(_stats([]), {"count": 0})


if __name__ == "__main__":
    unittest.main()
