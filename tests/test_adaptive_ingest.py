"""The decoder measures what a stream costs HERE, adapts, and says so (3 Sep).

'The stream is slow' on the pilot's box was the machine, not the camera: a
CPU-only laptop cannot decode a 1080p25 main stream at rate while the VLM
pegs the cores — and nothing measured it, adapted to it, or told anyone.
The contract:

- the deficit signal is the loop missing ITS OWN deadline (the larger of our
  sampling period and the source's arrival time for the stride) — waiting at
  the live edge for frames to ARRIVE is the camera's pacing, never counted
  as this machine falling behind;
- a machine in deficit gets capped at the rate it actually achieves, never
  below the floor, and recovers when headroom returns;
- the truth travels: link_status carries the numbers, the health doc grows a
  reason naming the sustainable rate and the one fix (the substream), and
  the Cameras screen shows 'machine-limited, sampling Nfps'.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.streams import StreamDecoder


def _decoder(target=4.0, source_fps=25.0) -> StreamDecoder:
    d = StreamDecoder.__new__(StreamDecoder)
    d.camera_id = "cam1"
    d.target_fps = target
    d.display_fps = 0.0
    d._fps = source_fps
    d._min_period = 1.0 / target
    d.stream_width, d.stream_height = 1920, 1080
    d._busy_ms = None
    d.sustainable_fps = 30.0
    d._ingest_limited = False
    return d


class AdaptiveIngestTests(unittest.TestCase):
    def test_a_healthy_machine_is_never_throttled_by_arrival_waits(self):
        """At the live edge, consuming stride frames takes stride/source_fps of
        WAITING on a perfectly healthy machine — that must read as on-pace."""
        d = _decoder(target=4.0, source_fps=25.0)   # stride ~6 -> deadline 240ms
        for _ in range(60):
            d._observe_ingest(0.245, stride=6)      # right at arrival pace
        self.assertFalse(d._ingest_limited)
        self.assertGreaterEqual(d._effective_fps(), 4.0)

    def test_a_starved_machine_is_capped_at_what_it_achieves(self):
        d = _decoder(target=4.0, source_fps=25.0)
        for _ in range(100):
            d._observe_ingest(0.6, stride=6)        # 600ms per kept frame
        self.assertTrue(d._ingest_limited)
        self.assertLess(d.sustainable_fps, 4.0)
        self.assertAlmostEqual(d._effective_fps(), d.sustainable_fps, places=3)
        self.assertGreaterEqual(d._effective_fps(), d.ADAPT_FLOOR_FPS)

    def test_the_display_boost_cannot_outrun_the_machine(self):
        d = _decoder(target=4.0, source_fps=25.0)
        for _ in range(100):
            d._observe_ingest(0.6, stride=6)
        d.display_fps = 15.0                        # a viewer opens the tile
        self.assertLess(d._effective_fps(), 4.0)    # capped, not 15

    def test_headroom_recovers_the_full_rate(self):
        d = _decoder(target=4.0, source_fps=25.0)
        for _ in range(100):
            d._observe_ingest(0.6, stride=6)
        self.assertTrue(d._ingest_limited)
        for _ in range(200):
            d._observe_ingest(0.05, stride=6)       # pressure gone
        self.assertFalse(d._ingest_limited)
        self.assertGreaterEqual(d._effective_fps(), 4.0)

    def test_no_source_fps_still_measures_against_our_own_pace(self):
        """The pilot's Tapo reports no fps metadata; the deadline falls back
        to the sampling period we chose for ourselves."""
        d = _decoder(target=4.0, source_fps=0.0)    # min_period 250ms
        for _ in range(100):
            d._observe_ingest(0.5, stride=1)        # can't hold 4fps
        self.assertTrue(d._ingest_limited)

    def test_link_status_carries_the_numbers(self):
        d = _decoder()
        for _ in range(50):
            d._observe_ingest(0.6, stride=6)
        status = d.ingest_status()
        self.assertEqual(status["width"], 1920)
        self.assertTrue(status["limited"])
        self.assertLess(status["sustainable_fps"], 4.0)
        self.assertEqual(status["sampling_fps"], status["sustainable_fps"])


class TruthTravelsTests(unittest.TestCase):
    def test_the_health_doc_names_the_rate_and_the_fix(self):
        from cvti.serving.health_doc import derive_status
        status, reasons = derive_status(
            cameras=[{"camera_id": "cam1", "state": "connected",
                      "ingest": {"limited": True, "sustainable_fps": 2.7,
                                 "width": 1920, "height": 1080,
                                 "source_fps": 25.0, "sampling_fps": 2.7}}],
            gate={}, disk={}, memory={}, components={"degraded": []})
        self.assertEqual(status, "degraded")
        self.assertTrue(any("2.7fps" in r and "substream" in r for r in reasons),
                        reasons)

    def test_an_unlimited_camera_adds_no_reason(self):
        from cvti.serving.health_doc import derive_status
        status, reasons = derive_status(
            cameras=[{"camera_id": "cam1", "state": "connected",
                      "ingest": {"limited": False, "sustainable_fps": 30.0}}],
            gate={}, disk={}, memory={}, components={"degraded": []})
        self.assertEqual(status, "ok")
        self.assertEqual(reasons, [])

    def test_the_cameras_screen_shows_the_limit(self):
        html = (Path(__file__).resolve().parents[1]
                / "cvti" / "app" / "web" / "index.html").read_text()
        self.assertIn("machine-limited", html)
        self.assertIn("link.ingest", html)


if __name__ == "__main__":
    unittest.main()
