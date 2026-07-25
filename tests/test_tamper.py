from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from cvti.detector.tamper import TamperDetector


def _normal_frame(seed):
    rng = np.random.default_rng(seed)
    # a bright, textured scene (high brightness + high Laplacian variance)
    return rng.integers(60, 200, size=(120, 160, 3), dtype=np.uint8)


class TamperTests(unittest.TestCase):
    def _warm(self, det, n=20):
        for i in range(n):
            det.update(_normal_frame(i))

    def test_blackout_fires(self):
        det = TamperDetector(min_frames=4)
        self._warm(det)
        black = np.zeros((120, 160, 3), dtype=np.uint8)
        fired = None
        for _ in range(10):
            r = det.update(black)
            if r:
                fired = r
                break
        self.assertIsNotNone(fired)
        self.assertEqual(fired["kind"], "blackout")

    def test_obscured_fires(self):
        det = TamperDetector(min_frames=4)
        self._warm(det)
        # bright but flat/blurred (uniform grey -> ~zero Laplacian variance)
        flat = np.full((120, 160, 3), 130, dtype=np.uint8)
        fired = None
        for _ in range(10):
            r = det.update(flat)
            if r:
                fired = r
                break
        self.assertIsNotNone(fired)
        self.assertEqual(fired["kind"], "obscured")

    def test_normal_scene_never_fires(self):
        det = TamperDetector(min_frames=4)
        for i in range(60):
            self.assertIsNone(det.update(_normal_frame(i)))

    def test_fires_once_per_episode(self):
        det = TamperDetector(min_frames=4)
        self._warm(det)
        black = np.zeros((120, 160, 3), dtype=np.uint8)
        fires = [det.update(black) for _ in range(20)]
        self.assertEqual(sum(1 for f in fires if f is not None), 1)  # latched


if __name__ == "__main__":
    unittest.main()
