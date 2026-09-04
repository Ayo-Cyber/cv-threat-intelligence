"""Every lesson from the pilot's 4 Sep screenshot, pinned.

One screenshot, five defects: transformers never in the Windows build env;
seaborn imported at weapon-load by a module PyInstaller never analyzes; his
RTSP password printed inside a mapping error on the System screen; the
mapper refused a second RTSP session by a camera the engine was already
streaming; and a config-text reviewed badge that #91 removed being
re-stamped by two callers on every engine start.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

from cvti.utils import redact_credentials


class CredentialRedactionTests(unittest.TestCase):
    def test_rtsp_credentials_never_reach_a_screen(self):
        leaked = "Unable to open source: rtsp://okwesimartins:12345678@192.168.8.103:554/stream1"
        clean = redact_credentials(leaked)
        self.assertNotIn("12345678", clean)
        self.assertNotIn("okwesimartins", clean)
        self.assertIn("rtsp://***@192.168.8.103:554/stream1", clean)

    def test_text_without_credentials_is_untouched(self):
        for text in ("plain error", "http://host/path", "rtsp://10.0.0.5:554/s1"):
            self.assertEqual(redact_credentials(text), text)

    def test_the_health_doc_redacts_defensively(self):
        from cvti.serving.health_doc import derive_status
        _, reasons = derive_status(
            cameras=[], gate={}, disk={}, memory={}, components={"degraded": []},
            scene_mapping=[{"camera_id": "cam1", "status": "failed",
                            "error": "Unable to open rtsp://user:secret@h/s"}])
        self.assertTrue(reasons)
        self.assertNotIn("secret", reasons[0])

    def test_the_mapper_raise_site_redacts_at_the_source(self):
        import inspect
        from cvti.scene import agent_mapper
        src = inspect.getsource(agent_mapper.capture_sample_frames)
        self.assertIn("redact_credentials", src)


class MapperUsesEngineFramesTests(unittest.TestCase):
    def test_map_result_with_frames_opens_no_capture(self):
        from cvti.scene import agent_mapper
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with mock.patch.object(agent_mapper, "capture_sample_frames",
                               side_effect=AssertionError("opened a session")):
            mapper = agent_mapper.AgentMapper(provider="mock")
            result = mapper.map_result("rtsp://user:pw@cam/1", "cam1",
                                       frames=[frame])
        self.assertEqual(result.context["camera_id"], "cam1")

    def test_map_result_without_frames_still_captures(self):
        from cvti.scene import agent_mapper
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        sample = agent_mapper.SampledFrame(image=frame, timestamp_seconds=0.0, score=1.0)
        with mock.patch.object(agent_mapper, "capture_sample_frames",
                               return_value=[sample]) as captured:
            agent_mapper.AgentMapper(provider="mock").map_result("clip.mp4", "cam1")
        captured.assert_called_once()

    def test_prepare_hands_the_engines_frame_to_the_mapper(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from _scene_hierarchy_fixtures import write_site  # noqa: F401 (env check)
        from cvti.serving.scene_map import FullAgentMapperService

        class Recording:
            def __init__(self):
                self.kwargs = None

            def map_result(self, *a, **kw):
                self.kwargs = kw
                from cvti.scene.agent_mapper import AgentMapper
                return AgentMapper(provider="mock").map_result(
                    a[0], a[1], frames=kw.get("frames"))

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "c.mp4"
            clip.write_bytes(b"x")
            mapper = Recording()
            service = FullAgentMapperService(Path(tmp) / "out", mapper)
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            service.frame_source = lambda cam_id: frame
            service.prepare([{"id": "cam1", "source": str(clip)}], "auto")
        self.assertIsNotNone(mapper.kwargs)
        self.assertEqual(len(mapper.kwargs["frames"]), 1)

    def test_the_pipeline_binds_the_frame_source(self):
        import inspect
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.run_site)
        self.assertIn("mapping_service.frame_source = _mapper_frame", src)


class ReviewedBadgeStaysDeadTests(unittest.TestCase):
    """#91 demoted config text to unreviewed; two callers re-stamped the badge
    on every engine start. Never again."""

    def _service_and_store(self, tmp):
        from cvti.scene.agent_mapper import AgentMapper
        from cvti.scene.context_store import SceneContextStore
        from cvti.serving.scene_map import FullAgentMapperService
        clip = Path(tmp) / "c.mp4"
        clip.write_bytes(b"x")
        camera = {"id": "cam1", "source": str(clip),
                  "scene_context_mode": "manual",
                  "scene_description": "A templated description nobody reviewed."}
        service = FullAgentMapperService(Path(tmp) / "out", AgentMapper(provider="mock"))
        store = SceneContextStore(Path(tmp) / "out/context", "cam1")
        return service, store, camera

    def test_inspect_never_restamps_site_config_reviewed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            service, store, camera = self._service_and_store(tmp)
            service.inspect([camera], "auto")
            self.assertEqual(store.load_status().status, "ready_unreviewed")
            self.assertNotEqual(store.load_status().reviewed_by, "site_config")

    def test_prepare_never_restamps_site_config_reviewed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            service, store, camera = self._service_and_store(tmp)
            result = service.prepare([camera], "auto")
            self.assertEqual(store.load_status().status, "ready_unreviewed")
            self.assertTrue(result.statuses["cam1"].get("usable"))

    def test_the_console_fallback_reports_unreviewed(self):
        src = (ROOT / "cvti" / "app" / "console_backend.py").read_text()
        self.assertNotIn('"reviewed_by": "site_config"', src)


class LeanOllamaProfileTests(unittest.TestCase):
    def _spawn_env(self, total_gb: float) -> dict:
        from cvti.verification import ollama
        captured = {}

        def fake_popen(cmd, stdout=None, stderr=None, env=None):
            captured["env"] = env
            return mock.Mock()

        vm = mock.Mock(total=int(total_gb * 1024 ** 3))
        with mock.patch.object(ollama, "ollama_binary", return_value="/fake/ollama"), \
             mock.patch.object(ollama.os, "stat", side_effect=OSError), \
             mock.patch.object(ollama.subprocess, "Popen", fake_popen), \
             mock.patch.dict(ollama.os.environ, {}, clear=False):
            for key in ("OLLAMA_NUM_PARALLEL", "OLLAMA_CONTEXT_LENGTH"):
                ollama.os.environ.pop(key, None)
            with mock.patch("psutil.virtual_memory", return_value=vm):
                self.assertTrue(ollama.start_server())
        return captured["env"]

    def test_a_16gb_box_gets_the_lean_profile(self):
        env = self._spawn_env(16.0)
        self.assertEqual(env["OLLAMA_NUM_PARALLEL"], "1")
        self.assertEqual(env["OLLAMA_CONTEXT_LENGTH"], "4096")

    def test_a_roomy_box_keeps_the_standard_profile(self):
        env = self._spawn_env(32.0)
        self.assertEqual(env["OLLAMA_NUM_PARALLEL"], "2")
        self.assertEqual(env["OLLAMA_CONTEXT_LENGTH"], "8192")

    def test_gate_slots_agree_with_the_lean_server(self):
        from cvti.verification import ollama
        vm = mock.Mock(total=int(8 * 1024 ** 3))
        with mock.patch.dict(ollama.os.environ, {}, clear=False):
            ollama.os.environ.pop("OLLAMA_NUM_PARALLEL", None)
            with mock.patch("psutil.virtual_memory", return_value=vm):
                self.assertEqual(ollama.configured_parallel_slots(), 1)


class BundleLessonsTests(unittest.TestCase):
    def test_the_build_installs_the_video_requirements(self):
        wf = (ROOT / ".github" / "workflows" / "build-app.yml").read_text()
        self.assertIn("requirements-video.txt", wf)

    def test_vendored_yolov5_survives_without_seaborn(self):
        src = (ROOT / "external" / "yolov5" / "utils" / "plots.py").read_text()
        self.assertIn("except ImportError", src.split("import seaborn")[1][:200])
        self.assertIn("sn is None", src)


if __name__ == "__main__":
    unittest.main()
