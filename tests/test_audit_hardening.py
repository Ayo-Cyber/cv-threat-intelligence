"""Pins for the 23 Aug head-to-toe audit — the remaining ten findings.

Each test names the silent failure it forbids. Behavioral where the seam
allows it; source-contract pins where the logic lives inside run_site's
closures (those are exercised live by the engine itself).
"""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _backend_helper import signed_in

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = (ROOT / "cvti/serving/pipeline.py").read_text()


class ReadOnceVsLiveTest(unittest.TestCase):
    """#5/#6/#13: the engine must apply notify/retention/heartbeat edits live."""

    def test_the_engine_watches_the_site_file(self):
        self.assertIn("_watch_site_meta", PIPELINE)
        self.assertIn("site-meta-watch", PIPELINE)
        # all three live-editable settings are applied inside the watcher
        watcher = PIPELINE.split("def _watch_site_meta")[1].split("threading.Thread")[0]
        self.assertIn("sink.notifier = build_notifier", watcher)
        self.assertIn("retention.policy = new_policy", watcher)
        self.assertIn("heartbeat.stop()", watcher)

    def test_retention_policy_swap_takes_effect_without_restart(self):
        # RetentionManager reads self.policy per cycle — a live swap must
        # change what status() reports and what the cutoff math uses.
        from cvti.serving.retention import RetentionManager, RetentionPolicy
        with tempfile.TemporaryDirectory() as tmp:
            m = RetentionManager(tmp, RetentionPolicy(days=30))
            self.assertEqual(m.policy.days, 30)
            m.policy = RetentionPolicy(days=7)
            self.assertEqual(m.status()["policy"]["days"], 7)


class RoutingDefaultTest(unittest.TestCase):
    """#2: an alert no routing rule claims goes to the SITE notifier."""

    def _sink(self, tmp, routing):
        from cvti.serving.alert_sink import AlertSink
        rp = Path(tmp) / "routing.json"
        rp.write_text(json.dumps(routing))
        sink = AlertSink(tmp, save_evidence=False, routing_path=str(rp))
        return sink

    def test_unmatched_alert_uses_the_site_notifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = self._sink(tmp, {"default": "console", "rules": [
                {"name": "night", "when": {"priority": ["critical"]},
                 "notify": "webhook:http://x/"}]})
            site_channel, routed_channel = [], []
            sink.notifier = type("N", (), {"notify": lambda s, e: site_channel.append(e)})()
            sink._notifier_cache["webhook:http://x/"] = type(
                "W", (), {"notify": lambda s, e: routed_channel.append(e)})()
            sink._dispatch({"priority": "medium", "rule": "theft", "camera_id": "c"})
            self.assertEqual(len(site_channel), 1,
                             "unmatched alert bypassed the operator's channel")
            self.assertEqual(routed_channel, [])

    def test_matched_alert_uses_its_rule_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = self._sink(tmp, {"default": "console", "rules": [
                {"name": "crit", "when": {"priority": ["critical"]},
                 "notify": "webhook:http://x/"}]})
            site_channel, routed_channel = [], []
            sink.notifier = type("N", (), {"notify": lambda s, e: site_channel.append(e)})()
            sink._notifier_cache["webhook:http://x/"] = type(
                "W", (), {"notify": lambda s, e: routed_channel.append(e)})()
            sink._dispatch({"priority": "critical", "rule": "weapons", "camera_id": "c"})
            self.assertEqual(len(routed_channel), 1)
            self.assertEqual(site_channel, [])


class FrozenFeedSwitchTest(unittest.TestCase):
    """#3: the installed bundle has an engine — a feed switch restarts it."""

    def test_restart_no_longer_excludes_frozen(self):
        import inspect
        from cvti.app.console_backend import ConsoleBackend
        src = inspect.getsource(ConsoleBackend._do_switch)
        self.assertNotIn('not getattr(sys, "frozen"', src,
                         "feed switch still refuses to restart the bundled engine")
        self.assertIn("start_monitoring", src)


class ScannerReconnectTest(unittest.TestCase):
    """#8: the custom-rules scanner reopens a dropped stream."""

    def test_the_loop_releases_and_reopens_dead_captures(self):
        src = (ROOT / "cvti/serving/custom_rules.py").read_text()
        loop = src.split("def _loop")[1]
        self.assertIn("cap.release()", loop)
        self.assertIn('caps[c["id"]] = self._open(c["source"])', loop,
                      "no reopen after a stream drop")


class ModelFailureVisibilityTest(unittest.TestCase):
    """#10/#11: configured-but-not-running coverage reaches /health."""

    def test_load_failures_become_health_reasons(self):
        self.assertIn("model_failures.append", PIPELINE)
        self.assertIn('doc["reasons"] = list(doc.get("reasons") or []) + model_failures',
                      PIPELINE)
        self.assertIn('doc["status"] = "degraded"', PIPELINE)

    def test_mobile_walks_ports_and_reports_total_failure(self):
        self.assertIn("range(mobile_port, mobile_port + 5)", PIPELINE)
        self.assertIn("notifications carry no respond link", PIPELINE)


class DbErrorHonestyTest(unittest.TestCase):
    """#12: a broken events DB must not render as a quiet site."""

    def test_list_events_reports_the_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "site.json").write_text('{"cameras": []}')
            (root / "events.db").mkdir()      # a directory: guaranteed OperationalError
            be = signed_in("owner", site_path=str(root / "site.json"),
                           db_path=str(root / "events.db"), enable_demo=False)
            out = be.list_events(10)
            self.assertIsInstance(out, dict)
            self.assertIn("events database unavailable", out["error"])


class SceneContextPersistenceTest(unittest.TestCase):
    """#7: live agent-mapping writes the file the UI and rule scanner read."""

    def test_mapping_writes_scene_context_json(self):
        import os
        import time
        from cvti.serving import scene_map
        real = scene_map.infer_scene
        scene_map.infer_scene = lambda *a, **k: {
            "environment_type": "retail", "scene_description": "A test shop."}
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                state = type("S", (), {"scene_context": None})()
                t = scene_map.map_cameras_async(
                    [{"id": "cam1", "source": "x.mp4"}], {"cam1": state},
                    model="m")
                t.join(timeout=5)
                f = Path(tmp) / "runs/context/cam1/scene_context.json"
                self.assertTrue(f.exists(), "mapping still never persists the scene")
                self.assertEqual(json.loads(f.read_text())["environment_type"], "retail")
                self.assertEqual(state.scene_context["scene_description"], "A test shop.")
            finally:
                os.chdir(cwd)
                scene_map.infer_scene = real


if __name__ == "__main__":
    unittest.main()
