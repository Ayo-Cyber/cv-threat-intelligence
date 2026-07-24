from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving import scene_map

ROOT = Path(__file__).resolve().parents[1]
CLIPS = sorted((ROOT / "data" / "test_clips").glob("*.mp4"))


class SceneParseTests(unittest.TestCase):
    def test_clean_json(self):
        d = scene_map._parse('{"environment_type": "retail", "scene_description": "shop counter"}')
        self.assertEqual(d["environment_type"], "retail")
        self.assertEqual(d["scene_description"], "shop counter")

    def test_json_with_prose(self):
        raw = 'Sure! Here is the scene:\n{"environment_type":"street","scene_description":"a road"} hope that helps'
        d = scene_map._parse(raw)
        self.assertEqual(d["environment_type"], "street")
        self.assertEqual(d["scene_description"], "a road")

    def test_garbage_falls_back(self):
        d = scene_map._parse("no json here")
        self.assertEqual(d["environment_type"], "unknown")
        self.assertIn("no json", d["scene_description"])

    def test_none(self):
        d = scene_map._parse(None)
        self.assertEqual(d["environment_type"], "unknown")

    @unittest.skipUnless(CLIPS, "no test clips")
    def test_sample_frame_returns_jpeg(self):
        jb = scene_map._sample_frame(str(CLIPS[0]))
        self.assertIsInstance(jb, (bytes, bytearray))
        self.assertGreater(len(jb), 100)
        self.assertEqual(jb[:2], b"\xff\xd8")  # JPEG SOI marker


if __name__ == "__main__":
    unittest.main()
