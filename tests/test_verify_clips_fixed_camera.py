"""The clip verifier must reject footage that is not from a fixed camera.

tools/fetch_eval_clips.py searches YouTube, and YouTube returns news coverage
ABOUT incidents at least as often as footage OF them. A people-count cannot
tell the difference -- a news anchor is a person -- so on 20-21 Sep the
verifier passed a studio broadcast as violence, a police body-cam as
loitering, and a picture-in-picture montage as running. Every one of those
"results" was worthless and looked fine.

Synthetic clips here, so the test runs anywhere without the real footage.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_clips  # noqa: E402


def _scene(seed: int, w: int = 320, h: int = 180) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    # some structure so phase correlation has something to lock on to
    base[h // 3: h // 2, :, :] = rng.integers(0, 60, size=3, dtype=np.uint8)
    base[:, w // 4: w // 3, :] = rng.integers(180, 255, size=3, dtype=np.uint8)
    return base


def _frames(kind: str, n: int = 40):
    scene = _scene(1)
    for i in range(n):
        if kind == "fixed":
            frame = scene.copy()
            frame[10:20, 10:20] = (i * 6) % 255          # a tiny changing overlay
        elif kind == "cuts":
            frame = _scene(1 + i // 8).copy()              # a new shot every 8 frames
        elif kind == "handheld":
            # Consecutive frames jump 10-16 px at full size, i.e. 5-8 px at the
            # verifier's 160x90 measurement scale -- the real body-cam measured
            # 7.7 px there. Alternating sign so every frame-to-frame delta is
            # large, not just the excursion.
            step = (10 + (i % 7)) * (1 if i % 2 else -1)
            frame = np.roll(np.roll(scene, step, axis=1), step // 2, axis=0)
        elif kind == "title_card":
            frame = (np.full_like(scene, 230) if i < 8 else scene).copy()
        else:
            raise ValueError(kind)
        yield frame


class FixedCameraGate(unittest.TestCase):
    def test_a_fixed_camera_measures_still(self):
        cuts, motion = verify_clips.camera_stability(_frames("fixed"))
        self.assertEqual(cuts, 0)
        self.assertLess(motion, verify_clips.MAX_MOTION_MEDIAN_PX)

    def test_edited_footage_shows_cuts(self):
        cuts, _ = verify_clips.camera_stability(_frames("cuts"))
        self.assertGreaterEqual(cuts, verify_clips.REJECT_CUTS,
                                "a shot change every 8 frames is edited footage")

    def test_a_handheld_camera_shows_motion(self):
        _, motion = verify_clips.camera_stability(_frames("handheld"))
        self.assertGreater(motion, verify_clips.MAX_MOTION_MEDIAN_PX,
                           "several px of wobble per frame is not a fixed camera")

    def test_a_title_card_is_exactly_one_cut(self):
        """Flagged for review, not rejected: one spike can be a barrier arm."""
        cuts, _ = verify_clips.camera_stability(_frames("title_card"))
        self.assertEqual(cuts, 1)
        self.assertLess(cuts, verify_clips.REJECT_CUTS)

    def test_the_gate_runs_before_any_class_check(self):
        """A news studio named violence_*.mp4 must be REJECTED, not REVIEWED."""
        import cv2
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "violence_synthetic.mp4"
            out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
            for frame in _frames("cuts", n=60):
                out.write(np.ascontiguousarray(frame))
            out.release()
            row = verify_clips.verify(path)
        self.assertEqual(row["verdict"], "REJECT", row)
        self.assertIn("hard cuts", row["why"])


if __name__ == "__main__":
    unittest.main()
