"""A freshly typed English rule gets its first scan NOW (audit 1 Sep, scanner).

The demo moment: an operator types "Detect anyone wearing a red cap" and
watches the screen. The scanner used to sleep out the remainder of its
inter-pass wait — up to interval x backoff, two minutes under load — before
the sentence's first scan. The contract now:

- during the inter-pass sleep the scanner polls the site file's mtime (one
  stat per second); a change that ADDS a rule scans that camera immediately;
- the sleep then resumes to its original deadline — one typed sentence never
  doubles the whole site's scan cadence;
- edits and deletions don't cut the sleep: only a new sentence has a person
  watching for its first answer.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.custom_rules import CustomRuleScanner


def _write_site(path: Path, questions: list[str]) -> None:
    path.write_text(json.dumps({"cameras": [{
        "id": "front", "source": "demo.mp4",
        "custom_rules": [{"question": q, "dwell": 4.0} for q in questions],
    }]}))


class FastPathTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.site = Path(self._tmp.name) / "site.json"
        _write_site(self.site, ["watch for hats"])
        self.scanned: list[str] = []
        self.scanner = CustomRuleScanner(
            [], sink=None, model="gemma3:4b",
            site_config_path=str(self.site),
            frame_source=lambda cid: np.zeros((32, 32, 3), dtype=np.uint8))
        self.scanner._refresh_cameras()
        self.scanner._site_changed()          # prime the mtime baseline
        # The scan itself is not under test — count who got scanned.
        self.scanner._scan_camera = (          # type: ignore[method-assign]
            lambda c, caps, dead: self.scanned.append(c["id"]))

    def tearDown(self):
        self.scanner._stop.set()
        self._tmp.cleanup()

    def _wait_in_thread(self, seconds: float) -> threading.Thread:
        t = threading.Thread(
            target=lambda: self.scanner._wait_for_next_pass(seconds, {}, {}))
        t.start()
        return t

    def test_a_new_sentence_is_scanned_mid_sleep(self):
        started = time.monotonic()
        waiter = self._wait_in_thread(4.0)
        time.sleep(0.3)
        _write_site(self.site, ["watch for hats", "Detect anyone wearing a red cap"])
        deadline = time.monotonic() + 3.0
        while not self.scanned and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.scanned, ["front"],
                         "the new sentence's camera was not scanned mid-sleep")
        self.assertLess(time.monotonic() - started, 3.0,
                        "the scan happened only after the full sleep")
        waiter.join(timeout=6)

    def test_the_sleep_resumes_to_its_original_deadline(self):
        """One typed sentence must not double the site's scan cadence: after
        the fast-path scan the wait keeps sleeping until its deadline."""
        started = time.monotonic()
        waiter = self._wait_in_thread(3.0)
        time.sleep(0.3)
        _write_site(self.site, ["watch for hats", "and one more thing"])
        waiter.join(timeout=6)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 2.7)   # slept it out
        self.assertEqual(self.scanned, ["front"])

    def test_a_deleted_rule_does_not_cut_the_sleep(self):
        waiter = self._wait_in_thread(1.6)
        time.sleep(0.3)
        _write_site(self.site, [])              # rule removed, file changed
        waiter.join(timeout=4)
        self.assertEqual(self.scanned, [])

    def test_an_unchanged_file_never_wakes_anything(self):
        waiter = self._wait_in_thread(1.2)
        waiter.join(timeout=3)
        self.assertEqual(self.scanned, [])

    def test_stop_cuts_the_sleep_immediately(self):
        started = time.monotonic()
        waiter = self._wait_in_thread(30.0)
        time.sleep(0.2)
        self.scanner._stop.set()
        waiter.join(timeout=3)
        self.assertLess(time.monotonic() - started, 5.0)


class LoopWiringPin(unittest.TestCase):
    def test_the_loop_sleeps_through_the_watching_wait(self):
        import inspect
        src = inspect.getsource(CustomRuleScanner._loop)
        self.assertIn("_wait_for_next_pass", src)
        self.assertNotIn("self._stop.wait(self.interval * self._backoff)", src)


if __name__ == "__main__":
    unittest.main()
