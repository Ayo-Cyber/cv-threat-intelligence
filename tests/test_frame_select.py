from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.verification.frame_select import frames_for_rule, select_evidence_frames


def _blurry(v=128):
    return np.full((32, 32, 3), v, dtype=np.uint8)          # flat -> ~0 sharpness


def _sharp(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)  # noise -> high sharpness


class FramesForRuleTests(unittest.TestCase):
    def test_per_rule_counts(self):
        self.assertEqual(frames_for_rule("baseline_weapon"), 1)
        self.assertEqual(frames_for_rule("weapon_sighting"), 1)
        self.assertEqual(frames_for_rule("violence"), 4)
        self.assertEqual(frames_for_rule("shoplifting_concealment"), 3)
        self.assertEqual(frames_for_rule("armed_robbery"), 5)
        self.assertEqual(frames_for_rule("loitering_at_shelf"), 1)
        self.assertEqual(frames_for_rule("something_unmapped"), 3)   # default


class SelectFramesTests(unittest.TestCase):
    def test_empty_buffer(self):
        frames, meta = select_evidence_frames([], "violence")
        self.assertEqual(frames, [])
        self.assertEqual(meta["strategy"], "none")

    def test_single_frame_rule_picks_sharpest(self):
        buf = [_blurry(), _blurry(), _sharp(1), _blurry()]
        frames, meta = select_evidence_frames(buf, "baseline_weapon")
        self.assertEqual(len(frames), 1)
        self.assertEqual(meta["strategy"], "sharpest")
        self.assertEqual(meta["selected_indices"], [2])          # the sharp one

    def test_multi_frame_rule_spans_and_caps(self):
        buf = [_sharp(i) for i in range(8)]
        frames, meta = select_evidence_frames(buf, "violence")   # wants 4
        self.assertEqual(len(frames), 4)
        self.assertEqual(meta["strategy"], "motion_peak_span")
        self.assertEqual(meta["selected_indices"], sorted(meta["selected_indices"]))

    def test_multi_frame_rule_small_buffer(self):
        buf = [_sharp(0), _sharp(1)]
        frames, meta = select_evidence_frames(buf, "violence")   # wants 4, only 2 exist
        self.assertLessEqual(len(frames), 2)


if __name__ == "__main__":
    unittest.main()
