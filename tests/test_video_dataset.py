from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.training.video_dataset import DEFAULT_CLASS_MAP, scan_clips


def _make_dataset(tmp: Path):
    for split, counts in {"training": {"normal": 4, "theft": 6}, "test": {"normal": 2, "theft": 3}}.items():
        for cls, n in counts.items():
            d = tmp / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"clip_{i}.mp4").write_bytes(b"")
            (d / "notes.txt").write_text("ignore me")   # non-video, must be skipped


class ScanClipsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_dataset(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_scans_videos_and_labels(self):
        items = scan_clips(self.root, "training")
        self.assertEqual(len(items), 10)                      # 4 normal + 6 theft, .txt skipped
        labels = [lbl for _, lbl in items]
        self.assertEqual(labels.count(DEFAULT_CLASS_MAP["normal"]), 4)
        self.assertEqual(labels.count(DEFAULT_CLASS_MAP["theft"]), 6)

    def test_per_class_limit_is_balanced(self):
        items = scan_clips(self.root, "training", per_class_limit=2)
        self.assertEqual(len(items), 4)                       # 2 per class
        labels = [lbl for _, lbl in items]
        self.assertEqual(labels.count(0), 2)
        self.assertEqual(labels.count(1), 2)

    def test_missing_class_dir_is_skipped(self):
        items = scan_clips(self.root, "test", class_map={"normal": 0, "theft": 1, "weapon": 2})
        self.assertEqual({lbl for _, lbl in items}, {0, 1})   # no 'weapon' dir -> skipped, no crash
        self.assertEqual(len(items), 5)


if __name__ == "__main__":
    unittest.main()
