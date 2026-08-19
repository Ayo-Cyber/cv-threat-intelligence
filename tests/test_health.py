"""Per-component error counters (EP-01-T3).

Catching an exception so one bad detector cannot kill the camera loop is the
right instinct. But a swallowed exception makes "this detector correctly found
nothing" and "this detector has thrown on every frame for a week" produce
identical silence. These counters are what tells them apart.
"""
import logging
import unittest

from cvti import health
from cvti.health import DEGRADED_MIN_SAMPLE, component, snapshot


class _Recorder(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class ComponentHealthTest(unittest.TestCase):
    def setUp(self):
        health.reset()
        self.handler = _Recorder()
        self.log = logging.getLogger("cvti.test.health")
        self.log.addHandler(self.handler)
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False

    def tearDown(self):
        self.log.removeHandler(self.handler)
        health.reset()

    def test_the_same_name_returns_the_same_record(self):
        component("detector.cam1").ok()
        self.assertEqual(component("detector.cam1").processed, 1)

    def test_an_error_increments_the_counter_and_logs_a_traceback(self):
        h = component("detector.cam1")
        try:
            raise ValueError("detector blew up")
        except ValueError as exc:
            h.failed(exc, self.log, "processing a frame")
        self.assertEqual(h.errors, 1)
        self.assertIn("ValueError", h.last_error)
        self.assertTrue(h.last_error_at)
        rec = self.handler.records[-1]
        self.assertIsNotNone(rec.exc_info, "no traceback was captured")
        self.assertIn("detector.cam1", rec.getMessage())
        self.assertIn("processing a frame", rec.getMessage())

    def test_a_healthy_component_is_not_degraded(self):
        h = component("detector.cam1")
        for _ in range(100):
            h.ok()
        self.assertFalse(h.degraded)
        self.assertEqual(h.error_rate, 0.0)

    def test_one_error_on_a_tiny_sample_is_not_a_100_percent_failure_rate(self):
        h = component("detector.cam1")
        h.failed(ValueError("x"), self.log)
        self.assertTrue(h.errors)
        self.assertFalse(h.degraded, "one error in one attempt is one error, not a trend")

    def test_a_persistently_failing_component_is_degraded(self):
        h = component("detector.cam1")
        for _ in range(DEGRADED_MIN_SAMPLE * 2):
            h.ok()
        for _ in range(DEGRADED_MIN_SAMPLE):
            h.failed(ValueError("x"), self.log)
        self.assertGreater(h.error_rate, 0.10)
        self.assertTrue(h.degraded)
        self.assertIn("detector.cam1", snapshot()["degraded"])

    def test_a_failure_on_every_frame_cannot_fill_the_disk(self):
        # The counter must carry the true scale while the log stays bounded.
        h = component("detector.cam1")
        for _ in range(1000):
            h.failed(ValueError("same failure every frame"), self.log)
        self.assertEqual(h.errors, 1000)
        self.assertLess(len(self.handler.records), 30,
                        "a per-frame failure logged unbounded")
        self.assertGreater(h.suppressed_logs, 900)

    def test_a_new_kind_of_failure_is_always_logged_at_least_once(self):
        # Rate limiting must not hide a *different* failure appearing later.
        h = component("detector.cam1")
        for _ in range(500):
            h.failed(ValueError("the usual"), self.log)
        before = len(self.handler.records)
        h.failed(MemoryError("something new"), self.log)
        self.assertEqual(len(self.handler.records), before + 1)

    def test_snapshot_puts_the_worst_component_first(self):
        good = component("detector.cam1")
        for _ in range(100):
            good.ok()
        bad = component("detector.cam2")
        for _ in range(50):
            bad.ok()
        for _ in range(50):
            bad.failed(ValueError("x"), self.log)
        names = [c["name"] for c in snapshot()["components"]]
        self.assertEqual(names[0], "detector.cam2")

    def test_snapshot_is_empty_before_anything_runs(self):
        snap = snapshot()
        self.assertEqual(snap["components"], [])
        self.assertEqual(snap["degraded"], [])
        self.assertEqual(snap["total_errors"], 0)


class HandlersAreNotSilentTest(unittest.TestCase):
    """No `except Exception` may swallow without leaving a record."""

    def test_no_broad_handler_is_completely_silent(self):
        import pathlib
        import re

        silent = []
        for path in sorted(pathlib.Path("cvti").rglob("*.py")):
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                if not re.match(r"\s*except Exception\b", line):
                    continue
                # An exemption must be stated on the handler and say why, so
                # adding one is a decision someone reviews rather than a habit.
                window = "\n".join(lines[i:i + 7])
                if "SILENT-OK" in window:
                    continue
                body = "\n".join(lines[i + 1:i + 7])
                if not re.search(r"log\.(debug|info|warning|error|exception|critical)"
                                 r"|raise|_health\.failed|health\.failed", body):
                    silent.append(f"{path}:{i + 1}")
        self.assertEqual(silent, [], f"handlers that swallow silently: {silent}")


if __name__ == "__main__":
    unittest.main()
