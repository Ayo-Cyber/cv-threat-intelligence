from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_action_model import build_centered_window, build_segment_windows, sample_evenly, sample_evenly_with_indices


class VideoActionModelTests(unittest.TestCase):
    def test_sample_evenly_returns_requested_number_in_order(self) -> None:
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(10)]

        sampled = sample_evenly(frames, count=4)

        self.assertEqual([int(frame[0, 0, 0]) for frame in sampled], [0, 3, 6, 9])

    def test_sample_evenly_with_indices_returns_source_indices(self) -> None:
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(10)]

        sampled = sample_evenly_with_indices(frames, count=4)

        self.assertEqual([item.index for item in sampled], [0, 3, 6, 9])
        self.assertEqual([int(item.frame[0, 0, 0]) for item in sampled], [0, 3, 6, 9])

    def test_sample_evenly_duplicates_short_inputs_to_match_model_count(self) -> None:
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)]

        sampled = sample_evenly(frames, count=5)

        self.assertEqual(len(sampled), 5)
        self.assertEqual([int(frame[0, 0, 0]) for frame in sampled], [0, 0, 1, 1, 2])

    def test_sample_evenly_rejects_empty_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            sample_evenly([], count=16)

    def test_build_segment_windows_samples_beginning_middle_and_ending(self) -> None:
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(90)]

        windows = build_segment_windows(frames, count=4)

        self.assertEqual([window.name for window in windows], ["beginning", "middle", "ending"])
        self.assertEqual([item.index for item in windows[0].sampled], [0, 9, 19, 29])
        self.assertEqual([item.index for item in windows[1].sampled], [30, 39, 49, 59])
        self.assertEqual([item.index for item in windows[2].sampled], [60, 69, 79, 89])

    def test_build_centered_window_samples_around_event_frame(self) -> None:
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(100)]

        window = build_centered_window(frames, center_index=50, radius_frames=10, count=5)

        self.assertEqual(window.name, "event")
        self.assertEqual(window.start_index, 40)
        self.assertEqual(window.end_index, 60)
        self.assertEqual([item.index for item in window.sampled], [40, 45, 50, 55, 60])


if __name__ == "__main__":
    unittest.main()
