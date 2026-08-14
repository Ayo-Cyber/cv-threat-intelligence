"""Feed switching must never block the UI thread.

Regression guard: switch_feed used to resolve 4 live stream URLs sequentially
(~2.5s each, 25s+ timeouts) and restart the engine INLINE, on the Qt UI thread —
which froze the whole window. It now returns immediately and works in the
background, with feed_switch_status() reporting progress.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.app.console_backend import ConsoleBackend


class FeedSwitchTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.site = self.d / "site.json"
        self.site.write_text(json.dumps(
            {"name": "T", "notify": "console", "configured": True, "cameras": []}))
        self.be = ConsoleBackend(site_path=str(self.site), db_path=str(self.d / "e.db"),
                                 enable_demo=False)

    def _wait(self, timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.be.feed_switch_status()
            if not st.get("busy"):
                return st
            time.sleep(0.2)
        return self.be.feed_switch_status()

    def test_switch_returns_immediately(self):
        """The call must not block — this is what froze the app."""
        t0 = time.time()
        r = self.be.switch_feed("demo")
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.5, "switch_feed blocked the caller (UI would freeze)")
        self.assertTrue(r.get("ok"))
        self._wait()

    def test_demo_switch_completes_and_repoints_site(self):
        self.be.switch_feed("demo")
        st = self._wait()
        self.assertTrue(st.get("done"))
        self.assertIsNone(st.get("error"))
        self.assertEqual(st.get("active"), "demo")
        self.assertTrue(self.be.site_path.endswith("deluxe_demo.json"))

    def test_unknown_source_errors_without_starting_work(self):
        r = self.be.switch_feed("nope")
        self.assertFalse(r.get("ok"))
        self.assertIn("unknown", r.get("error", "").lower())

    def test_concurrent_switch_is_rejected_not_queued(self):
        self.be._switch_state = {"busy": True, "status": "resolving…", "done": False}
        r = self.be.switch_feed("demo")
        self.assertFalse(r.get("ok"))
        self.assertTrue(r.get("busy"))

    def test_status_is_safe_before_any_switch(self):
        fresh = ConsoleBackend(site_path=str(self.site), db_path=str(self.d / "e2.db"),
                               enable_demo=False)
        st = fresh.feed_switch_status()
        self.assertFalse(st.get("busy"))

    def test_failure_clears_busy_so_the_ui_never_wedges(self):
        """Even a crash mid-switch must release the busy flag."""
        self.be._resolve_live_config = lambda src: (_ for _ in ()).throw(RuntimeError("boom"))
        reg = {"key": "live", "label": "L", "kind": "live", "config": str(self.d / "live.json")}
        self.be._switch_state = {"busy": True, "status": "x", "error": None, "done": False}
        self.be._do_switch(reg, "live")
        st = self.be.feed_switch_status()
        self.assertFalse(st.get("busy"))
        self.assertTrue(st.get("done"))
        self.assertIn("boom", st.get("error", ""))


if __name__ == "__main__":
    unittest.main()
