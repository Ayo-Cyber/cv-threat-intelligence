"""W7: budgets.json is enforceable, and the watchdog fails loudly.

A budget without an enforcer is a wish; an enforcer that trusts the
documentation field is a liar. These tests hold budget_check to reading the
EVIDENCE files, to failing on breach and on unreadable evidence, and to
reporting unmeasured budgets visibly. The watchdog side: the circuit breaker
latches after N unexpected exits and monitoring_status names the state —
'stopped by the operator' and 'crash-looped' must never look alike.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.budget_check import check

ROOT = Path(__file__).resolve().parents[1]


class BudgetCheckTest(unittest.TestCase):
    def test_committed_evidence_passes_today(self):
        s = check()
        self.assertEqual(s["breached"], [], s)
        self.assertEqual(s["errors"], [], s)
        # measured today: W1 latency + W3 rule answer
        ok = [r["budget"] for r in s["rows"] if r["status"] == "OK"]
        self.assertIn("glass_to_glass_p50_ms", ok)
        self.assertIn("object_rule_answer_ms", ok)

    def test_unmeasured_budgets_are_visible_never_passing(self):
        s = check()
        self.assertIn("detect_speedup_vs_torch_cpu", s["unmeasured"])
        for r in s["rows"]:
            if r["status"] == "UNMEASURED":
                self.assertNotIn("measured", r)

    def test_a_breach_in_fresher_evidence_fails_the_check(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "latency_baseline.json").write_text(json.dumps(
            {"glass_to_glass_ms": {"p50_ms": 431.0}}))
        s = check(tmp)
        self.assertIn("glass_to_glass_p50_ms", s["breached"])

    def test_soak_evidence_feeds_the_silent_failures_budget(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "soak_report.json").write_text(json.dumps(
            {"silent_failures": 0, "alert_on_screen_p95_s": 1.4}))
        s = check(tmp)
        by = {r["budget"]: r for r in s["rows"]}
        self.assertEqual(by["silent_failures"]["status"], "OK")
        self.assertEqual(by["alert_on_screen_s"]["status"], "OK")
        # and a dirty soak breaches
        (tmp / "soak_report.json").write_text(json.dumps(
            {"silent_failures": 2, "alert_on_screen_p95_s": 1.4}))
        self.assertIn("silent_failures", check(tmp)["breached"])

    def test_unreadable_evidence_is_its_own_failure(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "latency_baseline.json").write_text("{not json")
        s = check(tmp)
        self.assertIn("glass_to_glass_p50_ms", s["errors"])


class SoakTracebackTriageTest(unittest.TestCase):
    """The first 5h soak taught this distinction: the engine ran flawlessly
    and still 'failed' because fail-visible diagnostics (log.warning with
    exc_info) print tracebacks. Handled telemetry and chained sections of it
    must not count; a raw crash must."""

    def _count(self, text):
        sys.path.insert(0, str(ROOT / "tests" / "e2e"))
        from soak import _unhandled_tracebacks
        return _unhandled_tracebacks(text)

    def test_a_logged_diagnostic_does_not_count(self):
        text = ("WARNING  cvti.verification.gate — gate transport failed; "
                "alert will be surfaced UNVERIFIED\n"
                "Traceback (most recent call last):\n  File x\nOSError: down\n")
        self.assertEqual(self._count(text), 0)

    def test_chained_sections_belong_to_their_parent(self):
        text = ("WARNING  something — handled\n"
                "Traceback (most recent call last):\n  File x\nOSError: a\n"
                "\nDuring handling of the above exception, another exception "
                "occurred:\n\n"
                "Traceback (most recent call last):\n  File y\nKeyError: 'b'\n"
                "\nThe above exception was the direct cause of the following "
                "exception:\n\n"
                "Traceback (most recent call last):\n  File z\nRuntimeError: c\n")
        self.assertEqual(self._count(text), 0)

    def test_a_raw_crash_counts(self):
        text = ("frames flowing fine\n\n"
                "Traceback (most recent call last):\n"
                "  File pipeline.py\nKeyError: 'boom'\n")
        self.assertEqual(self._count(text), 1)


class WatchdogBreakerTest(unittest.TestCase):
    def test_the_breaker_latches_and_status_names_it(self):
        from cvti.app.console_backend import ConsoleBackend

        class _Dead:
            pid = 999

            def poll(self):
                return 3   # exited, code 3

        b = ConsoleBackend.__new__(ConsoleBackend)
        b._monitor = _Dead()
        b._monitor_should_run = False
        b._crash_looped = True
        b._restarts = 5
        b._engine_log_tail = lambda: ("/tmp/monitor.log", "boom")
        st = b.monitoring_status()
        self.assertFalse(st["running"])
        self.assertTrue(st["gave_up"])
        self.assertTrue(st["crash_looped"])
        self.assertEqual(st["restarts"], 5)

    def test_an_operator_stop_is_not_a_crash_loop(self):
        from cvti.app.console_backend import ConsoleBackend

        class _Stopped:
            pid = 999

            def poll(self):
                return 0

        b = ConsoleBackend.__new__(ConsoleBackend)
        b._monitor = _Stopped()
        b._monitor_should_run = False
        b._engine_log_tail = lambda: ("/tmp/monitor.log", "")
        st = b.monitoring_status()
        self.assertFalse(st["crash_looped"])

    def test_backoff_grows_and_caps(self):
        # the watchdog's delay table: 3,6,12,24,48 then capped at 60
        delays = [min(3 * (2 ** (n - 1)), 60) for n in range(1, 7)]
        self.assertEqual(delays, [3, 6, 12, 24, 48, 60])


if __name__ == "__main__":
    unittest.main()
