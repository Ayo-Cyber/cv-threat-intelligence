"""/health and the daily proof of life (EP-04 T1 + T4).

A customer's box dies at 3am Saturday — how do you find out? Today, they tell
you. These are the pieces that change the answer: one document that says
whether the site is OK and why, and a daily test that proves an alert would
actually reach a phone, rather than assuming it.
"""
import json
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

from cvti import health as health_registry
from cvti.contracts import VerificationResult
from cvti.serving.assurance import Assurance
from cvti.serving.health_doc import CRITICAL, DEGRADED, OK, build_health_doc, derive_status


def _cam(state="connected", cam_id="cam1", t=5.0):
    return {"camera_id": cam_id, "state": state, "time_in_state": t}


GOOD = dict(
    cameras=[_cam()], gate={"reachable": True}, disk={"level": "ok", "used_pct": 40},
    memory={"level": "ok", "available_gb": 8.0}, components={"degraded": []})


class StatusRulesTest(unittest.TestCase):
    def test_a_healthy_site_is_ok_with_no_reasons(self):
        status, reasons = derive_status(**GOOD)
        self.assertEqual(status, OK)
        self.assertEqual(reasons, [])

    def test_an_offline_camera_is_critical_and_named(self):
        status, reasons = derive_status(**{**GOOD, "cameras": [_cam("offline", "backroom", 360)]})
        self.assertEqual(status, CRITICAL)
        self.assertIn("backroom", reasons[0])

    def test_a_reconnecting_camera_is_degraded_not_critical(self):
        status, _ = derive_status(**{**GOOD, "cameras": [_cam("reconnecting")]})
        self.assertEqual(status, DEGRADED)

    def test_an_unreachable_gate_is_critical_because_alerts_arrive_unverified(self):
        status, reasons = derive_status(**{**GOOD, "gate": {"reachable": False}})
        self.assertEqual(status, CRITICAL)
        self.assertIn("unverified", reasons[0])

    def test_a_gate_with_no_traffic_yet_is_unknown_not_a_failure(self):
        status, _ = derive_status(**{**GOOD, "gate": {"reachable": None}})
        self.assertEqual(status, OK)

    def test_disk_and_memory_levels_map_to_status(self):
        self.assertEqual(derive_status(**{**GOOD, "disk": {"level": "warning", "used_pct": 88}})[0], DEGRADED)
        self.assertEqual(derive_status(**{**GOOD, "disk": {"level": "critical", "used_pct": 97}})[0], CRITICAL)
        self.assertEqual(derive_status(**{**GOOD, "memory": {"level": "warn", "available_gb": 1.5}})[0], DEGRADED)

    def test_a_degraded_component_is_named(self):
        status, reasons = derive_status(**{**GOOD, "components": {"degraded": ["detector.cam2"]}})
        self.assertEqual(status, DEGRADED)
        self.assertIn("detector.cam2", reasons[0])

    def test_the_worst_signal_wins(self):
        status, reasons = derive_status(**{
            **GOOD, "cameras": [_cam("offline")],
            "components": {"degraded": ["gate"]}})
        self.assertEqual(status, CRITICAL)
        self.assertEqual(len(reasons), 2)

    def test_the_document_carries_all_six_signal_classes(self):
        doc = build_health_doc(started_at=time.time() - 120, engine={"cameras": 1}, **GOOD)
        for key in ("cameras", "gate", "disk", "memory", "components", "uptime_s"):
            self.assertIn(key, doc)
        self.assertAlmostEqual(doc["uptime_s"], 120, delta=5)

    def test_nothing_in_the_document_resembles_frame_or_event_content(self):
        # The privacy constraint is structural: this doc becomes the heartbeat.
        doc = build_health_doc(started_at=time.time(), **GOOD)
        blob = json.dumps(doc)
        # Markers of frames or event rows. ("reasons" — the doc's own health
        # explanations — is ours and fine; evidence fields are not.)
        for forbidden in ("jpg", "jpeg", "frame_0", "evidence_dir", "raw_response",
                          "person_id", "track_id", "object_label", "base64"):
            self.assertNotIn(forbidden, blob, f"health doc carries {forbidden!r}")


class HealthEndpointTest(unittest.TestCase):
    """The /health route on the publisher's authenticated server."""

    def setUp(self):
        from cvti.serving.frame_publisher import FramePublisher
        self.doc = {"status": "ok", "reasons": [], "uptime_s": 1.0}
        self.pub = FramePublisher(health_provider=lambda: self.doc).start()

    def tearDown(self):
        self.pub.stop()

    def _get(self, path, token=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.pub.port}{path}",
            headers={"X-Argus-Token": self.pub.token if token is None else token})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, b""

    def test_health_returns_the_document(self):
        code, body = self._get("/health")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_health_is_authenticated_like_every_other_route(self):
        code, _ = self._get("/health", token="wrong")
        self.assertEqual(code, 401)

    def test_health_reflects_live_state_not_a_snapshot(self):
        self.doc["status"] = "critical"
        self.doc["reasons"] = ["camera backroom offline 360s"]
        code, body = self._get("/health")
        self.assertEqual(json.loads(body)["status"], "critical")

    def test_a_failing_provider_is_a_500_not_a_dead_server(self):
        def boom():
            raise RuntimeError("collector exploded")
        self.pub.health_provider = boom
        code, _ = self._get("/health")
        self.assertEqual(code, 500)
        code, _ = self._get("/cameras")     # the server survived
        self.assertEqual(code, 200)

    def test_no_provider_is_an_honest_404(self):
        self.pub.health_provider = None
        code, _ = self._get("/health")
        self.assertEqual(code, 404)


class _Notifier:
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    def notify(self, event):
        if self.fail:
            raise ConnectionError("telegram is down")
        self.sent.append(event)


class _Gate:
    def __init__(self, errored=False, raise_exc=False):
        self.errored = errored
        self.raise_exc = raise_exc

    def verify(self, frames, candidate, scene=None, examples=None):
        if self.raise_exc:
            raise TimeoutError("model timed out")
        return VerificationResult(False, 0.9, "self-test frame, nothing of note",
                                  "low", "t", error="parse: junk" if self.errored else "")


def _assurance(**kw):
    defaults = dict(
        latest_frame=lambda: np.zeros((8, 8, 3), np.uint8),
        gate_factory=lambda: _Gate(),
        notifier=_Notifier(),
        status_provider=lambda: {"status": "ok", "reasons": [],
                                 "cameras": [_cam()], "disk": {"used_pct": 40}})
    defaults.update(kw)
    return Assurance(**defaults)


class SelfTestTest(unittest.TestCase):
    def setUp(self):
        health_registry.reset()

    def tearDown(self):
        health_registry.reset()

    def test_a_working_chain_passes_and_notifies(self):
        a = _assurance()
        result = a.run_self_test()
        self.assertTrue(result["ok"])
        self.assertEqual(result["steps"]["frame"], "ok")
        self.assertIn("verdict", result["steps"]["gate"])
        self.assertEqual(len(a.notifier.sent), 1)
        self.assertIn("Self-test passed", a.notifier.sent[0]["reason"])

    def test_a_rejection_from_the_gate_still_passes(self):
        # This is a plumbing test, not an accuracy test: ANY verdict proves the
        # chain. Only "no verdict" is a failure.
        result = _assurance(gate_factory=lambda: _Gate()).run_self_test()
        self.assertTrue(result["ok"])

    def test_no_camera_frame_is_a_failure_and_raises_an_alert(self):
        a = _assurance(latest_frame=lambda: None)
        result = a.run_self_test()
        self.assertFalse(result["ok"])
        self.assertEqual(len(a.notifier.sent), 1)
        self.assertEqual(a.notifier.sent[0]["priority"], "high")

    def test_a_gate_that_cannot_decide_is_a_failure(self):
        a = _assurance(gate_factory=lambda: _Gate(errored=True))
        result = a.run_self_test()
        self.assertFalse(result["ok"])
        self.assertIn("no verdict", result["steps"]["gate"])

    def test_a_gate_exception_is_a_failure_not_a_crash(self):
        a = _assurance(gate_factory=lambda: _Gate(raise_exc=True))
        result = a.run_self_test()
        self.assertFalse(result["ok"])

    def test_a_dead_notifier_fails_and_cannot_carry_its_own_bad_news(self):
        a = _assurance(notifier=_Notifier(fail=True))
        result = a.run_self_test()          # must not raise
        self.assertFalse(result["ok"])
        self.assertIn("error", result["steps"]["notify"])

    def test_failures_land_in_the_component_registry(self):
        _assurance(latest_frame=lambda: None).run_self_test()
        snap = health_registry.snapshot()
        self.assertEqual(snap["components"][0]["name"], "self_test")
        self.assertEqual(snap["components"][0]["errors"], 1)


class DailyScheduleTest(unittest.TestCase):
    def _at(self, hour):
        # A fake clock pinned to today at `hour` local.
        base = time.localtime()
        return time.mktime((base.tm_year, base.tm_mon, base.tm_mday,
                            hour, 0, 0, 0, 0, -1))

    def test_nothing_runs_before_the_send_hour(self):
        a = _assurance(clock=lambda: self._at(7))
        a.tick()
        self.assertEqual(a.last_result, {})
        self.assertEqual(len(a.notifier.sent), 0)

    def test_both_jobs_run_once_after_the_send_hour_and_only_once(self):
        a = _assurance(clock=lambda: self._at(10))
        a.tick()
        a.tick()
        a.tick()
        self.assertTrue(a.last_result["ok"])
        # one self-test pass message + one daily normal — not three of each
        self.assertEqual(len(a.notifier.sent), 2)

    def test_the_daily_message_reports_ok_in_plain_terms(self):
        a = _assurance(clock=lambda: self._at(10))
        a.tick()
        daily = [e for e in a.notifier.sent if e["rule"] == "daily_status"][0]
        self.assertIn("All systems normal", daily["reason"])
        self.assertEqual(daily["priority"], "low")

    def test_a_degraded_site_sends_a_warning_instead(self):
        a = _assurance(clock=lambda: self._at(10),
                       status_provider=lambda: {"status": "degraded",
                                                "reasons": ["camera backroom reconnecting"],
                                                "cameras": []})
        a.send_daily_normal()
        daily = [e for e in a.notifier.sent if e["rule"] == "daily_status"][0]
        self.assertIn("degraded", daily["reason"])
        self.assertIn("backroom", daily["reason"])
        self.assertEqual(daily["priority"], "high")

    def test_opting_out_silences_the_daily_message_but_not_the_self_test(self):
        a = _assurance(clock=lambda: self._at(10), daily_normal=False)
        a.tick()
        rules = [e["rule"] for e in a.notifier.sent]
        self.assertNotIn("daily_status", rules)
        self.assertIn("self_test", rules)        # the pass notification


if __name__ == "__main__":
    unittest.main()
