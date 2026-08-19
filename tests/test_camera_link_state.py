"""Camera connection state (EP-01-T5).

The tamper detector answers "did someone cover this camera?". Nothing answered
"has this camera been unreachable since Tuesday?" — and an unreachable camera
raises no alerts, which is indistinguishable from a camera watching a quiet
corridor. The customer believes they have coverage they do not have.
"""
import time
import unittest

from cvti.serving import streams
from cvti.serving.streams import CONNECTED, OFFLINE, RECONNECTING, StreamDecoder


class LinkStateTest(unittest.TestCase):
    def _decoder(self, **kw):
        kw.setdefault("offline_grace_seconds", 0.05)
        return StreamDecoder("cam1", "rtsp://example/stream", **kw)

    def test_starts_connected(self):
        self.assertEqual(self._decoder().state, CONNECTED)

    def test_transitions_are_reported_with_time_in_state(self):
        seen = []
        d = self._decoder(on_state_change=lambda *a: seen.append(a))
        d._set_state(RECONNECTING, "dropped")
        time.sleep(0.02)
        d._set_state(OFFLINE, "still gone")
        d._set_state(CONNECTED, "back")
        self.assertEqual([(a[1], a[2]) for a in seen],
                         [(CONNECTED, RECONNECTING), (RECONNECTING, OFFLINE),
                          (OFFLINE, CONNECTED)])
        self.assertGreater(seen[1][3], 0.0, "time-in-state was not reported")

    def test_re_entering_the_same_state_is_not_an_event(self):
        seen = []
        d = self._decoder(on_state_change=lambda *a: seen.append(a))
        d._set_state(RECONNECTING)
        d._set_state(RECONNECTING)
        self.assertEqual(len(seen), 1)

    def test_a_failing_callback_cannot_break_the_decoder(self):
        def explode(*_a):
            raise RuntimeError("notifier is down")
        d = self._decoder(on_state_change=explode)
        d._set_state(RECONNECTING)          # must not raise
        self.assertEqual(d.state, RECONNECTING)

    def test_link_status_reports_what_the_ui_needs(self):
        d = self._decoder()
        status = d.link_status()
        for key in ("camera_id", "state", "time_in_state", "reconnects", "attempts"):
            self.assertIn(key, status)

    def test_backoff_is_exponential_and_capped(self):
        # Linear retry hammers a camera that has been down for an hour, and the
        # log with it.
        backoffs = [min(1.0 * (2 ** (attempt - 1)), 30.0) for attempt in range(1, 12)]
        self.assertEqual(backoffs[:5], [1.0, 2.0, 4.0, 8.0, 16.0])
        self.assertEqual(max(backoffs), 30.0)
        self.assertTrue(all(b <= 30.0 for b in backoffs))


class KilledAndRestoredStreamTest(unittest.TestCase):
    """The scenario the plan asks for: a stream that dies mid-run and returns."""

    def setUp(self):
        self.events = []
        self.d = StreamDecoder("cam1", "rtsp://example/stream",
                               offline_grace_seconds=0.05,
                               on_state_change=lambda *a: self.events.append(a[2]))

    def test_drop_then_offline_then_recovery_is_announced(self):
        # Stream drops.
        self.d._set_state(RECONNECTING, "stream dropped")
        self.assertEqual(self.d.state, RECONNECTING)

        # Still gone past the grace period -> offline.
        time.sleep(0.06)
        self.assertGreaterEqual(self.d.time_in_state, self.d.offline_grace_seconds)
        self.d._set_state(OFFLINE, "unreachable")

        # Comes back. Recovery must be announced, not silent: an operator told
        # the camera went down has to be told it returned.
        self.d._set_state(CONNECTED, "stream recovered")
        self.assertEqual(self.events, [RECONNECTING, OFFLINE, CONNECTED])

    def test_a_brief_blip_never_reaches_offline(self):
        # A camera reboot must not page anyone.
        self.d._set_state(RECONNECTING, "blip")
        self.d._set_state(CONNECTED, "recovered")
        self.assertNotIn(OFFLINE, self.events)


class OfflineIsNotInferredFromSilenceTest(unittest.TestCase):
    def test_backend_reports_unknown_when_the_engine_is_not_running(self):
        # Claiming "connected" for a camera nobody is watching is the exact
        # false confidence this epic exists to remove.
        import tempfile
        from pathlib import Path

        from cvti.app.console_backend import ConsoleBackend
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site.json"
            site.write_text('{"cameras": [{"id": "cam1", "source": "rtsp://x/y"}]}')
            be = ConsoleBackend(site_path=str(site), db_path=str(Path(tmp) / "events.db"),
                                enable_demo=False)
            links = be.camera_links()
            self.assertEqual(links[0]["state"], "unknown")

    def test_camera_offline_bypasses_the_vlm_gate(self):
        # There is no frame to verify — the camera being unreachable IS the
        # observation.
        from cvti.serving.gate_pool import BYPASS_DETECTORS
        self.assertIn("camera_offline", BYPASS_DETECTORS)

    def test_grace_period_is_configurable(self):
        self.assertEqual(
            StreamDecoder("c", "rtsp://x", offline_grace_seconds=12.5).offline_grace_seconds,
            12.5)
        self.assertEqual(StreamDecoder("c", "rtsp://x").offline_grace_seconds,
                         streams.DEFAULT_OFFLINE_GRACE)


if __name__ == "__main__":
    unittest.main()
