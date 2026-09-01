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


class CoordinatorSelfHealTest(unittest.TestCase):
    """The background coordinator now owns 'failed scene maps heal
    themselves' (it replaced the #69 retry loop). Pinned: a restart resets
    the attempt budget, a deleted camera finishes its job instead of killing
    the worker, and a changed source prunes the stale job."""

    def _coordinator(self, tmp, cameras):
        from cvti.scene.coordinator import SceneMappingCoordinator
        site = Path(tmp) / "site.json"
        site.write_text(json.dumps({"cameras": cameras}))

        class NeverRuns:
            def prepare(self, cams, policy):  # pragma: no cover - not reached
                raise AssertionError("mapper must not run in this test")
        return SceneMappingCoordinator(NeverRuns(), site, Path(tmp) / "out"), site

    def test_restart_resets_the_attempt_budget_of_failed_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            coord, site = self._coordinator(
                tmp, [{"id": "cam1", "source": "a.mp4"}])
            coord.enqueue(["cam1"])
            coord.jobs[0].state = "failed"
            coord.jobs[0].attempts = 4
            coord._save()

            coord.resume()

            self.assertEqual(coord.jobs[0].state, "pending")
            self.assertEqual(coord.jobs[0].attempts, 0,
                             "a persisted failure must not outlive the restart")

    def test_a_deleted_camera_fails_its_job_instead_of_killing_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            coord, site = self._coordinator(
                tmp, [{"id": "cam1", "source": "a.mp4"}])
            coord.enqueue(["cam1"])
            site.write_text(json.dumps({"cameras": []}))   # camera removed

            ran = coord.run_next()      # used to raise StopIteration

            self.assertTrue(ran)
            self.assertEqual(coord.jobs[0].state, "failed")
            self.assertIn("no longer exists", coord.jobs[0].error)

    def test_enqueue_prunes_jobs_for_removed_or_changed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            coord, site = self._coordinator(
                tmp, [{"id": "cam1", "source": "a.mp4"},
                      {"id": "cam2", "source": "b.mp4"}])
            coord.enqueue(["cam1", "cam2"])
            site.write_text(json.dumps({"cameras": [
                {"id": "cam1", "source": "MOVED.mp4"}]}))  # cam2 gone, cam1 moved

            coord.enqueue(["cam1"])

            self.assertEqual(
                [(j.camera_id, j.state) for j in coord.jobs],
                [("cam1", "pending")],
                "stale-fingerprint and deleted-camera jobs must be pruned")

    def test_area_note_reaches_the_mapper_as_a_scene_hint(self):
        from cvti.scene.coordinator import SceneMappingCoordinator
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site.json"
            site.write_text(json.dumps({
                "areas": [{"id": "yard", "name": "Yard",
                           "note": "cars park along the left fence"}],
                "cameras": [{"id": "cam1", "source": "a.mp4",
                             "area_id": "yard"}],
            }))
            seen = {}

            class Recording:
                def prepare(self, cams, policy):
                    seen.update(cams[0])
                    from cvti.serving.scene_map import SceneMappingPreflight
                    out = SceneMappingPreflight()
                    out.statuses["cam1"] = {"status": "failed", "error": "x"}
                    return out

            coord = SceneMappingCoordinator(Recording(), site, Path(tmp) / "out")
            coord.enqueue(["cam1"])
            coord.run_next()

            self.assertEqual(seen.get("scene_hint"),
                             "cars park along the left fence",
                             "the area note must arrive under the key "
                             "_operator_hints actually reads")
