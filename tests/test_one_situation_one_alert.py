"""One ongoing situation is ONE incident, not an alert every cooldown (3 Sep).

The operator's question: 'if we detect someone wearing a hoodie, why do we
constantly keep sending that alert?' The contract now:

- the FIRST sighting alerts; while the condition persists the incident is
  updated, not re-announced;
- it re-alerts only as a REMINDER, at widening intervals (90s -> 4.5min ->
  13.5min, capped 15min), and only while NOBODY has acknowledged the alert —
  repetition's one honest job is getting a human, and an acked incident has
  one;
- two consecutive clear scans close the incident; the next sighting is new
  information and alerts fresh;
- the heartbeat carries the open incidents so the Rules panel can say
  'ongoing: hoodie, 12 min' instead of looking asleep.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.custom_rules import (CLEAR_AFTER_MISSES, REMINDER_CAP_SECONDS,
                                       REMINDER_WIDENING, CustomRuleScanner)

CAM = {"id": "front", "source": "x",
       "custom_threats": [{"name": "hoodie", "description": "someone in a hoodie"}]}
HIT = {"name": "hoodie", "reason": "hood up over the head"}


class _Sink:
    def __init__(self, state: str | None = "new"):
        self.alerts: list = []
        self.state = state

    def handle(self, alert, result):
        self.alerts.append((alert.title, result.reason))

    def triage_state(self, camera_id, rule_name):
        return self.state


def _scanner(sink=None, cooldown=90.0) -> CustomRuleScanner:
    return CustomRuleScanner([CAM], sink=sink or _Sink(), model="test",
                             cooldown=cooldown)


class IncidentLifecycleTests(unittest.TestCase):
    def test_a_persisting_condition_alerts_once_not_every_scan(self):
        sink = _Sink()
        s = _scanner(sink)
        for i in range(6):                       # a minute of scans, still there
            s._route_hits(CAM, None, [HIT], now=100.0 + i * 12)
        self.assertEqual(len(sink.alerts), 1)
        self.assertEqual(sink.alerts[0][0], "CUSTOM: hoodie")

    def test_unacknowledged_incidents_remind_at_widening_intervals(self):
        sink = _Sink(state="new")                # nobody has touched the alert
        s = _scanner(sink, cooldown=90.0)
        times = [100.0 + i * 12 for i in range(200)]   # 40 minutes of scans
        for now in times:
            s._route_hits(CAM, None, [HIT], now=now)
        titles = [t for t, _ in sink.alerts]
        self.assertEqual(titles[0], "CUSTOM: hoodie")
        self.assertGreaterEqual(titles.count("STILL: hoodie"), 2)
        # Widening: far fewer reminders than the 26 the old 90s cooldown sent
        # in the same 40 minutes.
        self.assertLessEqual(len(sink.alerts), 6)
        self.assertIn("unacknowledged", sink.alerts[1][1])

    def test_an_acknowledged_incident_goes_quiet(self):
        sink = _Sink(state="acknowledged")       # a human owns it
        s = _scanner(sink, cooldown=90.0)
        for i in range(200):                     # 40 more minutes, still there
            s._route_hits(CAM, None, [HIT], now=100.0 + i * 12)
        self.assertEqual(len(sink.alerts), 1)    # the discovery, nothing after

    def test_clearance_closes_and_reappearance_alerts_fresh(self):
        sink = _Sink()
        s = _scanner(sink)
        s._route_hits(CAM, None, [HIT], now=100.0)
        for i in range(CLEAR_AFTER_MISSES):      # gone for two scans -> closed
            s._route_hits(CAM, None, [], now=112.0 + i * 12)
        self.assertEqual(s._incidents, {})
        s._route_hits(CAM, None, [HIT], now=200.0)   # back an hour later
        self.assertEqual([t for t, _ in sink.alerts],
                         ["CUSTOM: hoodie", "CUSTOM: hoodie"])

    def test_one_missed_scan_does_not_close_an_incident(self):
        """A single flaky model answer must not split one situation in two."""
        sink = _Sink(state="acknowledged")
        s = _scanner(sink)
        s._route_hits(CAM, None, [HIT], now=100.0)
        s._route_hits(CAM, None, [], now=112.0)      # one miss
        s._route_hits(CAM, None, [HIT], now=124.0)   # still the same incident
        self.assertEqual(len(sink.alerts), 1)
        self.assertEqual(s._incidents[("front", "hoodie")]["misses"], 0)

    def test_reminders_widen_geometrically_and_cap(self):
        sink = _Sink(state="new")
        s = _scanner(sink, cooldown=90.0)
        s._route_hits(CAM, None, [HIT], now=0.0)
        incident = s._incidents[("front", "hoodie")]
        # Force each reminder and observe the growing gap.
        gaps = []
        now = 0.0
        for _ in range(4):
            now = incident["next_reminder_at"]
            s._route_hits(CAM, None, [HIT], now=now)
            gaps.append(incident["next_reminder_at"] - now)
        self.assertEqual(gaps[0], 90.0 * REMINDER_WIDENING)
        self.assertEqual(gaps[1], 90.0 * REMINDER_WIDENING ** 2)
        self.assertEqual(gaps[-1], REMINDER_CAP_SECONDS)

    def test_a_sink_without_triage_state_still_reminds(self):
        """Best-effort: a bare sink (tests, older deployments) means we can't
        see acks — remind as if unacknowledged rather than going silent."""
        class BareSink:
            def __init__(self):
                self.alerts = []

            def handle(self, alert, result):
                self.alerts.append(alert.title)

        sink = BareSink()
        s = CustomRuleScanner([CAM], sink=sink, model="test", cooldown=90.0)
        s._route_hits(CAM, None, [HIT], now=0.0)
        s._route_hits(CAM, None, [HIT], now=100.0)   # past the first reminder
        self.assertEqual(sink.alerts, ["CUSTOM: hoodie", "STILL: hoodie"])

    def test_the_heartbeat_carries_open_incidents(self):
        s = _scanner(_Sink())
        s._route_hits(CAM, None, [HIT], now=100.0)
        s._record(CAM, [HIT])
        entry = s._status["front"]
        self.assertEqual(entry["ongoing"][0]["rule"], "hoodie")
        self.assertIn("for_s", entry["ongoing"][0])


class SinkTriageStateTests(unittest.TestCase):
    def test_the_sink_reports_the_newest_events_state(self):
        import sqlite3
        import tempfile
        from cvti.serving.alert_sink import AlertSink
        with tempfile.TemporaryDirectory() as tmp:
            sink = AlertSink(tmp, notifier=None)
            self.assertIsNone(sink.triage_state("front", "custom:hoodie"))
            with sink._lock:
                sink._db.execute(
                    "INSERT INTO events (ts, camera_id, rule, priority, review) "
                    "VALUES (1, 'front', 'custom:hoodie', 'high', NULL)")
                sink._db.commit()
            self.assertEqual(sink.triage_state("front", "custom:hoodie"), "new")
            # An acknowledge writes BOTH columns (triage.acknowledge does).
            with sink._lock:
                sink._db.execute("UPDATE events SET state='acknowledged', review='ack'")
                sink._db.commit()
            self.assertEqual(sink.triage_state("front", "custom:hoodie"),
                             "acknowledged")
            with sink._lock:
                sink._db.execute("UPDATE events SET state = 'resolved'")
                sink._db.commit()
            self.assertEqual(sink.triage_state("front", "custom:hoodie"), "resolved")
            # A legacy row from before the state machine: review projects.
            with sink._lock:
                sink._db.execute(
                    "INSERT INTO events (ts, camera_id, rule, priority, state, review) "
                    "VALUES (2, 'front', 'custom:hoodie', 'high', NULL, 'ack')")
                sink._db.commit()
            self.assertEqual(sink.triage_state("front", "custom:hoodie"),
                             "acknowledged")


if __name__ == "__main__":
    unittest.main()
