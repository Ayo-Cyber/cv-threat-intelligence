"""A slow first start must read 'Starting…', never 'engine not running'.

Field screenshot (1 Sep, Windows v1.6.0): green Monitoring dot over 'not
monitoring — engine not running' over 'TrueSight loading models…'. The engine
was fine — it was cold-loading models and running the scene preflight, which
happen BEFORE the first heartbeat existed. Pinned here: the engine's first
words land before anything slow, the warm-up names its phase, the console
passes it through, and scene mappings that failed against a still-downloading
model retry on their own.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from cvti.app.console_backend import ConsoleBackend
from cvti.serving import pipeline


def _backend(tmp):
    site = Path(tmp) / "site"
    site.mkdir(exist_ok=True)
    return ConsoleBackend(site_path=str(site / "site.json"),
                          db_path=str(site / "events.db"), enable_demo=False)


class StartingHeartbeatTest(unittest.TestCase):

    def test_first_words_land_before_the_preflight(self):
        src = inspect.getsource(pipeline.run_site)
        self.assertLess(src.index("write_starting_health("),
                        src.index("mapping_service.inspect("),
                        "the starting heartbeat must be written before the "
                        "slow scene preflight, or a fresh install reads as a "
                        "dead engine for minutes")

    def test_starting_doc_names_every_camera_and_the_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = {"cameras": [{"id": "Front Gate", "source": "x.mp4"},
                                {"id": "cam2", "source": "y.mp4"}]}
            pipeline.write_starting_health(tmp, site, gate_provider="local",
                                           gate_model="gemma3:4b")
            doc = json.loads((Path(tmp) / "gate_health.json").read_text())
            self.assertEqual([c["state"] for c in doc["cameras"]],
                             ["starting", "starting"])
            self.assertTrue(doc["engine"]["phase"].startswith("starting"))
            self.assertLess(time.time() - doc["generated_at"], 5)
            self.assertNotEqual(doc["status"], "critical")

    def test_console_reports_starting_from_a_fresh_phase_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            hb = {"generated_at": time.time(),
                  "engine": {"phase": "starting — mapping camera scenes"}}
            (Path(tmp) / "site" / "gate_health.json").write_text(json.dumps(hb))
            cb._monitor = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                s = cb.monitoring_status()
            finally:
                cb._monitor.kill()
            self.assertTrue(s["running"])
            self.assertTrue(s.get("starting"))
            self.assertIn("mapping camera scenes", s["phase"])
            self.assertFalse(s.get("stalled"))

    def test_a_stale_starting_phase_is_not_reported_as_starting(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = _backend(tmp)
            hb = {"generated_at": time.time() - 600,
                  "engine": {"phase": "starting — mapping camera scenes"}}
            (Path(tmp) / "site" / "gate_health.json").write_text(json.dumps(hb))
            cb._monitor = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                s = cb.monitoring_status()
            finally:
                cb._monitor.kill()
            self.assertFalse(s.get("starting"))

    def test_the_footer_renders_starting_as_its_own_state(self):
        html = Path("cvti/app/web/index.html").read_text()
        self.assertIn('st==="starting"', html)
        self.assertIn("engine starting", html)
        self.assertIn('"Starting…"', html.replace("'", '"'))


class _RetryService:
    """Heals 'Front Gate', leaves 'cam2' failed."""
    def __init__(self):
        self.retried: list = []

    def prepare(self, cameras, policy):
        from cvti.serving.scene_map import SceneMappingPreflight
        self.retried = [str(c["id"]) for c in cameras]
        out = SceneMappingPreflight()
        for camera in cameras:
            cid = str(camera["id"])
            if cid == "Front Gate":
                out.contexts[cid] = {"camera_id": cid,
                                     "scene_description": "A gate.",
                                     "environment_type": "estate_gate"}
                out.statuses[cid] = {"status": "ready_unreviewed", "error": ""}
            else:
                out.statuses[cid] = {"status": "failed", "error": "still down"}
        return out


class _CamState:
    def __init__(self):
        self.scene_context = None


class SceneRetryTest(unittest.TestCase):

    def test_retry_heals_states_and_health_rows_in_place(self):
        cams_cfg = [{"id": "Front Gate", "source": "a"}, {"id": "cam2", "source": "b"},
                    {"id": "cam3", "source": "c"}]
        mapping_health = [
            {"camera_id": "Front Gate", "status": "failed", "error": "connection refused"},
            {"camera_id": "cam2", "status": "failed", "error": "connection refused"},
            {"camera_id": "cam3", "status": "ready_reviewed", "error": ""},
        ]
        states = {"Front Gate": _CamState(), "cam2": _CamState(), "cam3": _CamState()}
        service = _RetryService()

        still = pipeline.retry_failed_scene_mappings(
            service, cams_cfg, "auto", mapping_health, states)

        self.assertEqual(sorted(service.retried), ["Front Gate", "cam2"],
                         "only FAILED cameras are retried")
        self.assertEqual(states["Front Gate"].scene_context["environment_type"],
                         "estate_gate")
        self.assertIsNone(states["cam3"].scene_context)
        rows = {r["camera_id"]: r["status"] for r in mapping_health}
        self.assertEqual(rows["Front Gate"], "ready_unreviewed")
        self.assertEqual(rows["cam2"], "failed")
        self.assertEqual(still, {"cam2"})

    def test_nothing_failed_means_no_mapper_calls(self):
        service = _RetryService()
        still = pipeline.retry_failed_scene_mappings(
            service, [{"id": "c1", "source": "a"}],
            "auto", [{"camera_id": "c1", "status": "ready_reviewed"}],
            {"c1": _CamState()})
        self.assertEqual(service.retried, [])
        self.assertEqual(still, set())

    def test_a_blocked_camera_not_running_is_not_retried(self):
        service = _RetryService()
        pipeline.retry_failed_scene_mappings(
            service, [{"id": "Front Gate", "source": "a"}],
            "auto", [{"camera_id": "Front Gate", "status": "failed"}],
            {})   # not in states: it never started
        self.assertEqual(service.retried, [])


class PolicyBlockIsAWaitNotADeathTest(unittest.TestCase):
    """'No camera can start until scene context is ready' used to be the
    engine's last words — then the watchdog respawned it forever (pilot,
    1 Sep: two days of 'the engine never starts'). A policy block is a
    waiting state: stay alive, say why, retry the preflight."""

    def test_the_blocked_branch_retries_instead_of_returning(self):
        src = inspect.getsource(pipeline.run_site)
        blocked = src.index("no camera can start")
        window = src[src.rindex("while not cams_cfg", 0, blocked):blocked + 400]
        self.assertIn("while not cams_cfg", window)
        self.assertNotIn("return", window,
                         "a policy block must wait and retry, never exit")
        self.assertIn("retrying the preflight", window)

    def test_the_wait_names_itself_in_the_heartbeat_phase(self):
        src = inspect.getsource(pipeline.run_site)
        self.assertIn("waiting — this site's policy requires reviewed", src)

    def test_starting_health_carries_scene_rows_when_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"camera_id": "cam1", "status": "failed", "error": "timed out"}]
            pipeline.write_starting_health(
                tmp, {"cameras": [{"id": "cam1", "source": "x"}]},
                gate_provider="local", gate_model="m",
                phase="waiting — policy", scene_mapping=rows)
            doc = json.loads((Path(tmp) / "gate_health.json").read_text())
            self.assertEqual(doc["scene_mapping"], rows)
            self.assertTrue(doc["engine"]["phase"].startswith("waiting"))
