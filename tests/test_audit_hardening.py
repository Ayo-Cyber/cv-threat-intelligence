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
            be = signed_in("owner", site_path=str(root / "site.json"),
                           db_path=str(root / "events.db"), enable_demo=False)
            # Break the store AFTER startup: the integrity check has already
            # run (and would have quarantined a pre-broken file — that path is
            # covered in test_backup.py). This is the mid-session failure.
            db = Path(be.db_path)
            if db.exists():
                db.unlink()
            db.mkdir()                    # a directory: guaranteed OperationalError
            out = be.list_events(10)
            self.assertIsInstance(out, dict)
            self.assertIn("events database unavailable", out["error"])

    def test_a_pre_broken_store_is_quarantined_at_startup_instead(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "site.json").write_text('{"cameras": []}')
            (root / "events.db").write_bytes(b"never a database" * 64)
            be = signed_in("owner", site_path=str(root / "site.json"),
                           db_path=str(root / "events.db"), enable_demo=False)
            self.assertEqual(be.db_check.get("state"), "quarantined")
            self.assertTrue(list(root.glob("events.corrupt-*.db")),
                            "corrupt store not preserved for recovery")


class SceneContextPersistenceTest(unittest.TestCase):
    """#7: mapping writes the site-scoped file the UI and engine share."""

    def test_mapping_writes_scene_context_json(self):
        import numpy as np
        from cvti.scene.agent_mapper import MappingResult
        from cvti.serving.scene_map import FullAgentMapperService

        class Mapper:
            def map_result(self, source, camera_id, sample_count=3,
                           source_frame_path="", operator_hints=None):
                return MappingResult({
                    "camera_id": camera_id,
                    "source_type": "video_file",
                    "environment_type": "retail_shop",
                    "scene_description": "A test shop.",
                    "expected_actors": ["customers"],
                    "zones": [],
                    "confidence": 0.8,
                    "generated_at": "2026-08-30T10:00:00Z",
                    "source_frame_path": source_frame_path,
                    "notes": "",
                }, np.zeros((20, 20, 3), dtype=np.uint8), "{}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "x.mp4"
            source.write_bytes(b"clip")
            output = root / "site-output"
            result = FullAgentMapperService(output, Mapper()).prepare(
                [{"id": "cam1", "source": str(source)}], "auto")
            context_file = output / "context/cam1/scene_context.json"
            self.assertTrue(context_file.exists())
            self.assertEqual(
                json.loads(context_file.read_text())["environment_type"],
                "retail_shop",
            )
            self.assertEqual(
                result.contexts["cam1"]["scene_description"], "A test shop."
            )


if __name__ == "__main__":
    unittest.main()
