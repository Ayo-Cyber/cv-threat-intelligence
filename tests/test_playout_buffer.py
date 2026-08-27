"""The live wall must play bursts at content rate — never fast-forward.

27 Aug, user report: "the stream is like too fast". HLS delivers video in
multi-second segments; draining to the newest frame made the wall freeze for a
segment and then replay it at decode speed. The playout buffer decouples
arrival speed from playback speed: bursts land in a bounded queue and the
publisher pops at content rate, catching up at a quiet 2x only when genuinely
behind, evicting oldest when the source outruns everything.
"""
from __future__ import annotations

import unittest

import numpy as np

from cvti.serving.streams import PlayoutBuffer, StreamDecoder

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class PlayoutPacingTest(unittest.TestCase):

    def test_a_burst_is_paced_not_replayed_at_delivery_speed(self):
        """20 frames arriving at once must NOT pop in one tick."""
        b = PlayoutBuffer(rate=10.0, depth_seconds=8.0)
        for _ in range(20):
            b.push(FRAME)
        now, popped = 0.0, 0
        while b.pop_due(now) is not None:      # what a single publisher tick sees
            popped += 1
        self.assertEqual(popped, 1, "a burst replayed instantly — that IS fast-forward")

    def test_content_plays_at_its_own_rate(self):
        b = PlayoutBuffer(rate=10.0, depth_seconds=8.0)
        for _ in range(5):
            b.push(FRAME)
        got = [t for t in np.arange(0.0, 1.0, 0.01) if b.pop_due(float(t))]
        self.assertEqual(len(got), 5)
        gaps = np.diff(got)
        self.assertTrue((gaps >= 0.099).all(),
                        f"frames popped faster than the content rate: {gaps}")

    def test_catchup_is_2x_and_only_when_genuinely_behind(self):
        b = PlayoutBuffer(rate=10.0, depth_seconds=2.0)          # maxlen 20
        for _ in range(19):                                       # > 75% full
            b.push(FRAME)
        self.assertIsNotNone(b.pop_due(0.0))
        self.assertIsNotNone(b.pop_due(0.05), "2x catch-up should allow a pop at half-period")
        t = 1000.0
        while len(b) > 10:                                        # drain below the threshold
            if b.pop_due(t) is None:
                t += 0.01                                         # the clock only moves forward
        t = 2000.0
        self.assertIsNotNone(b.pop_due(t))
        self.assertIsNone(b.pop_due(t + 0.05), "still catching up when no longer behind")
        self.assertIsNotNone(b.pop_due(t + 0.11))

    def test_lag_is_hard_capped_by_eviction(self):
        b = PlayoutBuffer(rate=10.0, depth_seconds=2.0)          # maxlen 20
        for _ in range(200):
            b.push(FRAME)
        self.assertEqual(len(b), 20, "the buffer grew without bound")
        self.assertEqual(b.evicted, 180)

    def test_empty_buffer_shows_nothing_rather_than_stale(self):
        b = PlayoutBuffer(rate=10.0)
        self.assertIsNone(b.pop_due(0.0))

    def test_frames_are_stored_bounded_not_raw(self):
        b = PlayoutBuffer(rate=10.0, max_width=32)
        b.push(np.zeros((1080, 1920, 3), dtype=np.uint8))
        jpeg = b.pop_due(1e9)
        self.assertIsNotNone(jpeg)
        self.assertLess(len(jpeg), 20_000, "1080p stored raw would be ~6 MB per frame")


class WhoGetsAPlayoutTest(unittest.TestCase):
    """URL sources burst; files pace themselves; a webcam is real-time."""

    def test_url_sources_get_one(self):
        for src in ("rtsp://cam/1", "https://x/playlist.m3u8"):
            self.assertIsNotNone(StreamDecoder("c", src).playout, src)

    def test_files_and_webcams_do_not(self):
        for src in ("0", "clip.mp4"):
            self.assertIsNone(StreamDecoder("c", src).playout, src)


if __name__ == "__main__":
    unittest.main()
