"""Alert routing + escalation.

The risk here is silent misdelivery: an alert quietly going nowhere, or a
critical one being escalated to someone who already handled it. Both are covered.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.serving.routing import EscalationTracker, RoutingPolicy, RoutingRule


def _ev(**kw):
    base = {"camera_id": "cam1", "rule": "shoplifting", "priority": "high",
            "zone": None, "reason": "r"}
    base.update(kw)
    return base


class RuleMatchTests(unittest.TestCase):
    def test_priority_match(self):
        r = RoutingRule("crit", {"priority": ["critical"]}, "telegram:t:c")
        self.assertTrue(r.matches(_ev(priority="critical")))
        self.assertFalse(r.matches(_ev(priority="high")))

    def test_priority_is_case_insensitive(self):
        r = RoutingRule("crit", {"priority": ["CRITICAL"]}, "x")
        self.assertTrue(r.matches(_ev(priority="critical")))

    def test_camera_and_rule_and_zone(self):
        self.assertTrue(RoutingRule("c", {"camera": ["cam1"]}, "x").matches(_ev()))
        self.assertFalse(RoutingRule("c", {"camera": ["other"]}, "x").matches(_ev()))
        self.assertTrue(RoutingRule("r", {"rule": ["shoplifting"]}, "x").matches(_ev()))
        self.assertTrue(RoutingRule("z", {"zone": ["till"]}, "x").matches(_ev(zone="till")))

    def test_all_conditions_must_hold(self):
        r = RoutingRule("both", {"priority": ["critical"], "camera": ["cam1"]}, "x")
        self.assertTrue(r.matches(_ev(priority="critical")))
        self.assertFalse(r.matches(_ev(priority="critical", camera_id="cam9")))

    def test_time_window_including_overnight_wrap(self):
        r = RoutingRule("night", {"between": ["18:00", "06:00"]}, "x")
        self.assertTrue(r.matches(_ev(), now=datetime(2026, 1, 1, 23, 0)))
        self.assertTrue(r.matches(_ev(), now=datetime(2026, 1, 1, 2, 0)))
        self.assertFalse(r.matches(_ev(), now=datetime(2026, 1, 1, 12, 0)))

    def test_empty_when_is_catch_all(self):
        self.assertTrue(RoutingRule("any", {}, "x").matches(_ev()))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = RoutingPolicy(default="console", rules=[
            RoutingRule("crit", {"priority": ["critical"]}, "telegram:t:c", 5, "whatsapp"),
            RoutingRule("cam", {"camera": ["cam1"]}, "webhook:https://x"),
        ])

    def test_first_match_wins(self):
        spec, name = self.policy.channels_for(_ev(priority="critical"))
        self.assertEqual((spec, name), ("telegram:t:c", "crit"))

    def test_falls_through_to_next_rule(self):
        spec, name = self.policy.channels_for(_ev(priority="low"))
        self.assertEqual((spec, name), ("webhook:https://x", "cam"))

    def test_unmatched_uses_default_never_dropped(self):
        spec, name = self.policy.channels_for(_ev(camera_id="other", priority="low"))
        self.assertEqual((spec, name), ("console", "default"))

    def test_load_from_file(self):
        d = Path(tempfile.mkdtemp())
        (d / "r.json").write_text(json.dumps({"default": "console", "rules": [
            {"name": "n", "when": {"priority": ["critical"]}, "notify": "whatsapp",
             "escalate_after_minutes": 3, "escalate_to": "console"}]}))
        p = RoutingPolicy.load(d / "r.json")
        self.assertEqual(p.rules[0].notify, "whatsapp")
        self.assertEqual(p.rules[0].escalate_after_minutes, 3)

    def test_missing_or_broken_file_degrades_to_default(self):
        d = Path(tempfile.mkdtemp())
        self.assertEqual(RoutingPolicy.load(d / "nope.json").default, "console")
        (d / "bad.json").write_text("{not json")
        self.assertEqual(RoutingPolicy.load(d / "bad.json").rules, [])


class EscalationTests(unittest.TestCase):
    def test_fires_only_after_the_deadline(self):
        t = EscalationTracker(is_acknowledged=lambda _: False)
        rule = RoutingRule("crit", {}, "console", escalate_after_minutes=5, escalate_to="whatsapp")
        t.register(1, _ev(), rule, now=1000.0)
        self.assertEqual(t.due(now=1000.0 + 4 * 60), [])      # not yet
        due = t.due(now=1000.0 + 6 * 60)
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["to"], "whatsapp")

    def test_acknowledged_alerts_are_not_escalated(self):
        t = EscalationTracker(is_acknowledged=lambda _: True)
        rule = RoutingRule("crit", {}, "console", escalate_after_minutes=1, escalate_to="whatsapp")
        t.register(1, _ev(), rule, now=0.0)
        self.assertEqual(t.due(now=999.0), [])

    def test_escalates_at_most_once(self):
        t = EscalationTracker(is_acknowledged=lambda _: False)
        rule = RoutingRule("c", {}, "console", escalate_after_minutes=1, escalate_to="x")
        t.register(1, _ev(), rule, now=0.0)
        self.assertEqual(len(t.due(now=999.0)), 1)
        self.assertEqual(t.due(now=9999.0), [])
        self.assertEqual(t.pending_count, 0)

    def test_rule_without_escalation_registers_nothing(self):
        t = EscalationTracker()
        t.register(1, _ev(), RoutingRule("plain", {}, "console"), now=0.0)
        self.assertEqual(t.pending_count, 0)


class SinkRoutingTests(unittest.TestCase):
    def test_sink_routes_by_priority_and_registers_escalation(self):
        from cvti.serving.alert_queue import QueuedAlert
        from cvti.serving.alert_sink import AlertSink

        d = Path(tempfile.mkdtemp())
        (d / "routing.json").write_text(json.dumps({"default": "console", "rules": [
            {"name": "crit", "when": {"priority": ["critical"]}, "notify": "console",
             "escalate_after_minutes": 5, "escalate_to": "console"}]}))
        sink = AlertSink(str(d), save_evidence=False, routing_path=str(d / "routing.json"))
        sent = []
        sink._notifier_cache["console"] = type("N", (), {"notify": lambda _s, e: sent.append(e)})()

        class R:
            confirmed, confidence, reason = True, 0.9, "real"
        alert = QueuedAlert(camera_id="cam1", rule_name="weapons", priority="critical",
                            title="T", timestamp=1.0, payload={"frames": []})
        sink.handle(alert, R())
        self.assertEqual(len(sent), 1)
        self.assertEqual(sink.routed, 1)
        self.assertEqual(sink.escalations.pending_count, 1)   # awaiting acknowledgement


if __name__ == "__main__":
    unittest.main()
