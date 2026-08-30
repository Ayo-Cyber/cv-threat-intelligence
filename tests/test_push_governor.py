"""'Kinda slow and moves fast-forwarded sometimes' (pilot's Windows RTSP,
30 Aug): a camera with no FPS metadata defeats stride sampling, every frame
(~25/s) enters a playout popping 12/s, the buffer lives near full, and the 2x
catch-up loops forever — lag, sprint, lag. The governor thins pushes only
under SUSTAINED fullness; HLS's legitimate spike-drain rhythm never trips it.
"""
from __future__ import annotations

import unittest

from cvti.serving.streams import PushGovernor


class PushGovernorTest(unittest.TestCase):

    def test_calm_buffers_admit_everything(self):
        g = PushGovernor()
        self.assertTrue(all(g.admit(0.2, t * 0.05) for t in range(200)))
        self.assertEqual(g.skip, 0)

    def test_sustained_pressure_thins_pushes(self):
        """The Windows-Tapo pattern: fullness pinned high for many seconds."""
        g = PushGovernor()
        admitted = sum(g.admit(0.9, t * 0.04) for t in range(1000))  # 40s at 25fps
        self.assertGreater(g.skip, 0, "the governor never engaged")
        self.assertLess(admitted, 800, "pushes were not meaningfully thinned")

    def test_hls_spike_drain_never_trips_it(self):
        """Fullness spikes at each segment, drains between — intermittent."""
        g = PushGovernor()
        now = 0.0
        for _ in range(20):                      # 20 segment cycles
            for _ in range(60):                  # 5s burst lands, buffer high
                g.admit(0.8, now); now += 0.02
            for _ in range(40):                  # drains while waiting
                g.admit(0.2, now); now += 0.1
        self.assertEqual(g.skip, 0, "a normal HLS rhythm was throttled")

    def test_pressure_release_recovers_full_rate(self):
        g = PushGovernor()
        for t in range(600):
            g.admit(0.9, t * 0.04)
        self.assertGreater(g.skip, 0)
        now = 600 * 0.04
        for t in range(600):
            g.admit(0.1, now + t * 0.04)
        self.assertEqual(g.skip, 0, "the governor never let go")
