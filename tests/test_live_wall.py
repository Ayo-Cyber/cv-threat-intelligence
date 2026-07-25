from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CLIPS = sorted((ROOT / "data" / "test_clips").glob("*.mp4"))


@unittest.skipUnless(CLIPS, "no test clips present")
class LiveWallTests(unittest.TestCase):
    def test_decodes_multiple_sources(self):
        from cvti.app.live_wall import LiveWall
        srcs = [{"id": p.stem, "source": str(p)} for p in CLIPS[:2]]
        wall = LiveWall(srcs, width=240, fps=10).start()
        try:
            # give the decoder threads a beat to produce at least one frame each
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                f = wall.frames()
                if len(f) == len(srcs) and all(v.get("uri") for v in f.values()):
                    break
                time.sleep(0.1)
            f = wall.frames()
            self.assertEqual(set(f), {s["id"] for s in srcs})
            for v in f.values():
                self.assertTrue(v["uri"].startswith("data:image/jpeg;base64,"))
                self.assertGreater(v["frame"], 0)
                self.assertLessEqual(v["w"], 240)
        finally:
            wall.stop()

    def test_backend_live_cycle(self):
        import tempfile
        from cvti.app.console_backend import ConsoleBackend
        with tempfile.TemporaryDirectory() as d:
            be = ConsoleBackend(site_path=str(Path(d) / "s.json"), db_path=str(Path(d) / "e.db"), enable_demo=False)
            cams = be.live_start(count=2)              # empty site -> demo clips fallback
            self.assertTrue(len(cams) >= 1)
            time.sleep(1.5)
            frames = be.live_frames()
            self.assertTrue(any(v.get("uri") for v in frames.values()))
            self.assertEqual(be.live_stop(), {"stopped": True})


if __name__ == "__main__":
    unittest.main()
