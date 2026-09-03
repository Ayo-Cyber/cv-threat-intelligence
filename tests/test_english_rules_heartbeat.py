"""'The describe in english hasn't fired' — three reports in two days, and the
product could not distinguish its three silences: the model answering none
every cycle, every call failing, and nothing scanning at all. The scanner now
writes a heartbeat every cycle — scans, matches, failures, the last outcome in
words — and the Rules panel shows it beside the rules it describes.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cvti.serving.custom_rules import CustomRuleScanner

CAM = {"id": "cam1", "source": "0",
       "custom_rules": [{"question": "Is anyone standing beside the car?", "dwell": 4.0}]}
FRAME = np.zeros((32, 32, 3), dtype=np.uint8)


def _scanner(tmp):
    s = CustomRuleScanner([CAM], sink=None, model="test")
    s.status_path = Path(tmp) / "english_rules_status.json"
    return s


class HeartbeatTest(unittest.TestCase):

    def _status(self, s):
        return json.loads(s.status_path.read_text())["cameras"]["cam1"]

    def test_a_none_answer_is_recorded_in_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _scanner(tmp)
            s._record(CAM, [])
            st = self._status(s)
            self.assertEqual(st["last_outcome"], "model answered none")
            self.assertEqual((st["scans"], st["hits"], st["errors"]), (1, 0, 0))

    def test_a_match_names_the_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _scanner(tmp)
            s._record(CAM, [{"name": "is anyone standing beside the", "reason": "x"}])
            st = self._status(s)
            self.assertIn("matched: is anyone standing beside the", st["last_outcome"])
            self.assertEqual(st["hits"], 1)
            self.assertIn("last_hit_at", st)

    def test_a_failed_call_is_loud_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _scanner(tmp)
            s._record(CAM, None, error="connection refused")
            st = self._status(s)
            self.assertEqual(st["last_outcome"], "call failed")
            self.assertEqual(st["errors"], 1)
            self.assertIn("connection refused", st["last_error"])

    def test_failures_land_in_the_health_components_too(self):
        from cvti.health import snapshot
        with tempfile.TemporaryDirectory() as tmp:
            s = _scanner(tmp)
            s._record(CAM, None, error="boom")
            names = [c["name"] for c in snapshot()["components"]]
            self.assertIn("english_rules.cam1", names)

    def test_the_scan_loop_records_every_cycle(self):
        """Every scan path must call _record — the whole point is that a cycle
        can never pass silently again. The two loop paths were unified into
        _scan_camera (fast-path refactor, 3 Sep): the loop and the fast path
        must both scan THROUGH it, and it must record success and failure."""
        import inspect
        scan = inspect.getsource(CustomRuleScanner._scan_camera)
        self.assertGreaterEqual(scan.count("self._record("), 2,
                                "a scan outcome exists that records nothing")
        loop = inspect.getsource(CustomRuleScanner._loop)
        wait = inspect.getsource(CustomRuleScanner._wait_for_next_pass)
        self.assertIn("self._scan_camera(", loop)
        self.assertIn("self._scan_camera(", wait)
        self.assertNotIn("self._check(", loop + wait,
                         "a scan path bypasses _scan_camera and its recording")

    def test_the_ui_shows_the_pulse(self):
        html = Path("cvti/app/web/index.html").read_text()
        self.assertIn("fillRulePulse", html)
        self.assertIn('call("englishRulesStatus"', html)


class ScannerBacksOffUnderLoadTest(unittest.TestCase):
    """Every scanner call timed out during the 29 Aug demo: the scanner and
    the gate contend for two Ollama slots, and a starving scanner that
    retries at full cadence adds pressure to the resource it is starving on.
    Failures now double the effective interval (cap 10x); one success resets."""

    def test_failures_stretch_the_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _scanner(tmp)
            self.assertEqual(s._backoff, 1.0)
            s._record(CAM, None, error="timed out")
            s._record(CAM, None, error="timed out")
            self.assertEqual(s._backoff, 4.0)
            self.assertEqual(self_status(s)["backoff_s"], round(s.interval * 4))

    def test_one_success_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _scanner(tmp)
            for _ in range(5):
                s._record(CAM, None, error="timed out")
            s._record(CAM, [])
            self.assertEqual(s._backoff, 1.0)
            self.assertNotIn("backoff_s", self_status(s))

    def test_the_backoff_is_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _scanner(tmp)
            for _ in range(20):
                s._record(CAM, None, error="timed out")
            self.assertEqual(s._backoff, 10.0)

    def test_the_wait_uses_the_backoff(self):
        import inspect
        src = inspect.getsource(CustomRuleScanner._loop)
        self.assertIn("self.interval * self._backoff", src)


def self_status(s):
    return json.loads(s.status_path.read_text())["cameras"]["cam1"]
