"""View-only cameras are glass, not detectors (4 Sep, pilot ask:
"make the stream just streaming the video, no detection on it").

A camera marked view_only streams to the wall and runs NOTHING else: no
camera state, no models, no English scans, no scene mapping — and a site of
only view-only cameras still starts. The pilot's starved Windows box gets a
smooth wall on the cameras he only watches, and spends its cycles on the
cameras that detect.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _backend_helper import signed_in


class EngineSideTests(unittest.TestCase):
    def _site(self, tmp: str) -> dict:
        cfg = Path(tmp) / "rules.json"
        cfg.write_text('{"rules": []}')
        return {"cameras": [
            {"id": "watcher", "source": "a.mp4", "config": str(cfg)},
            {"id": "glass", "source": "b.mp4", "config": str(cfg),
             "view_only": True},
        ]}

    def test_no_camera_state_is_built_for_glass(self):
        from cvti.serving.camera import build_camera_states
        with tempfile.TemporaryDirectory() as tmp:
            cams = build_camera_states(self._site(tmp))
        self.assertIn("watcher", cams)
        self.assertNotIn("glass", cams)

    def test_glass_is_not_an_active_detection_camera(self):
        from cvti.serving.pipeline import active_cameras_after_preflight
        with tempfile.TemporaryDirectory() as tmp:
            active = active_cameras_after_preflight(
                self._site(tmp)["cameras"], preflight=None)
        self.assertEqual([c["id"] for c in active], ["watcher"])

    def test_the_pipeline_never_detects_on_glass_frames(self):
        import inspect
        from cvti.serving.pipeline import MultiStreamPipeline
        src = inspect.getsource(MultiStreamPipeline.run)
        self.assertIn("f.camera_id not in self.view_only", src)

    def test_glass_decoders_idle_low_and_say_what_they_are(self):
        from cvti.serving.streams import StreamDecoder
        d = StreamDecoder("glass", "b.mp4", target_fps=1.0, view_only=True)
        self.assertTrue(d.ingest_status()["view_only"])
        d2 = StreamDecoder("watcher", "a.mp4", target_fps=4.0)
        self.assertNotIn("view_only", d2.ingest_status())


class BackendTests(unittest.TestCase):
    def test_the_toggle_writes_and_clears_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = str(Path(tmp) / "site.json")
            be = signed_in(site_path=site, db_path=str(Path(tmp) / "e.db"),
                           enable_demo=False)
            be.add_camera({"id": "gate", "source": "rtsp://h/1"})
            out = be.set_view_only("gate", True)
            self.assertTrue(out["ok"] and out["view_only"])
            cam = json.loads(Path(site).read_text())["cameras"][0]
            self.assertTrue(cam["view_only"])
            be.set_view_only("gate", False)
            cam = json.loads(Path(site).read_text())["cameras"][0]
            self.assertNotIn("view_only", cam)

    def test_an_unknown_camera_is_a_named_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            be = signed_in(site_path=str(Path(tmp) / "site.json"),
                           db_path=str(Path(tmp) / "e.db"), enable_demo=False)
            self.assertIn("error", be.set_view_only("nope", True))


class UiTests(unittest.TestCase):
    def test_the_rules_card_carries_the_toggle(self):
        html = (ROOT / "cvti" / "app" / "web" / "index.html").read_text()
        start = html.index("function renderRules()")
        card = html[start:start + 6000]
        self.assertIn("toggleViewOnly", card)
        self.assertIn("view-only — streaming, no detection", card)
        fn = html[html.index("function toggleViewOnly"):][:600]
        self.assertIn('call("setViewOnly"', fn)


if __name__ == "__main__":
    unittest.main()
