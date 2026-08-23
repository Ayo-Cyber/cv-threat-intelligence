from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.alert_queue import AlertQueue, QueuedAlert
from cvti.serving.batcher import collect_batch
from cvti.serving.streams import Frame


def _alert(cam="cam0", rule="loitering", prio="medium", ts=0.0, track=1, zone="shelf_right"):
    return QueuedAlert(camera_id=cam, rule_name=rule, priority=prio, title="T",
                       timestamp=ts, track_id=track, zone=zone)


class AlertQueueTests(unittest.TestCase):
    def test_duplicate_within_cooldown_is_suppressed(self):
        q = AlertQueue(cooldown_seconds=8.0)
        self.assertTrue(q.add(_alert(ts=0.0)))
        self.assertFalse(q.add(_alert(ts=3.0)))       # same cam/rule/track/zone, < cooldown
        self.assertEqual(q.dropped_duplicates, 1)
        self.assertEqual(q.pending_count, 1)

    def test_refires_after_cooldown(self):
        q = AlertQueue(cooldown_seconds=8.0)
        q.add(_alert(ts=0.0))
        self.assertTrue(q.add(_alert(ts=9.0)))         # cooldown elapsed
        self.assertEqual(q.pending_count, 2)

    def test_different_track_not_deduped(self):
        q = AlertQueue(cooldown_seconds=8.0)
        q.add(_alert(ts=0.0, track=1))
        self.assertTrue(q.add(_alert(ts=1.0, track=2)))  # concurrent second person
        self.assertEqual(q.pending_count, 2)

    def test_drain_is_priority_ordered_and_throttled(self):
        q = AlertQueue(cooldown_seconds=0.0)           # no dedup for this test
        q.add(_alert(prio="medium", ts=1.0, track=1))
        q.add(_alert(prio="critical", ts=2.0, track=2))
        q.add(_alert(prio="high", ts=3.0, track=3))
        first = q.drain(max_per_drain=2)
        self.assertEqual([a.priority for a in first], ["critical", "high"])
        self.assertEqual(q.pending_count, 1)           # throttled: one left
        self.assertEqual(q.drain()[0].priority, "medium")

    def test_concurrent_cameras_both_queued(self):
        q = AlertQueue()
        self.assertTrue(q.add(_alert(cam="cam0", ts=0.0)))
        self.assertTrue(q.add(_alert(cam="cam7", ts=0.0)))  # different camera, same rule
        self.assertEqual(q.pending_count, 2)


class _FakeDecoder:
    def __init__(self, frames):
        self._frames = list(frames)

    def read_latest(self):
        return self._frames.pop(0) if self._frames else None


class BatcherTests(unittest.TestCase):
    def test_collect_only_fresh_frames_one_per_camera(self):
        f0 = Frame("cam0", object(), 1, 0.0)
        f1 = Frame("cam1", object(), 1, 0.0)
        decoders = {
            "cam0": _FakeDecoder([f0, None]),   # fresh, then nothing new
            "cam1": _FakeDecoder([f1]),
            "cam2": _FakeDecoder([]),           # idle camera contributes nothing
        }
        batch = collect_batch(decoders)
        self.assertEqual({f.camera_id for f in batch}, {"cam0", "cam1"})
        self.assertEqual(collect_batch(decoders), [])   # nothing new next tick

    def test_max_batch_caps_size(self):
        decoders = {f"cam{i}": _FakeDecoder([Frame(f"cam{i}", object(), 1, 0.0)])
                    for i in range(10)}
        self.assertEqual(len(collect_batch(decoders, max_batch=4)), 4)


class GatePoolTests(unittest.TestCase):
    def test_pool_drains_queue_and_confirms_via_mock_gate(self):
        import time
        import numpy as np
        from cvti.contracts import CandidateAlert
        from cvti.serving.gate_pool import GatePool
        from cvti.verification.gate import VerificationGate

        q = AlertQueue(cooldown_seconds=0.0)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        candidate = CandidateAlert(rule_name="loitering_at_shelf", priority="medium",
                                   detector="presence", title="PERSON IN ZONE",
                                   person_id=6, object_label=None, timestamp=1.0)
        qa = QueuedAlert(camera_id="cam0", rule_name="loitering_at_shelf", priority="medium",
                         title="PERSON IN ZONE", timestamp=1.0, track_id=6, zone="shelf_right",
                         payload={"candidate": candidate, "frames": [frame],
                                  "scene": {"environment_type": "retail_shop"}})
        q.add(qa)

        verdicts = []
        pool = GatePool(q, gate_factory=lambda: VerificationGate(provider="mock"),
                        on_verdict=lambda a, r: verdicts.append((a, r))).start()
        try:
            deadline = time.time() + 5.0
            while pool.verified < 1 and time.time() < deadline:
                time.sleep(0.05)
        finally:
            pool.stop()

        self.assertEqual(pool.verified, 1)
        self.assertEqual(pool.confirmed, 1)          # mock auto-confirms
        self.assertEqual(q.pending_count, 0)
        self.assertEqual(verdicts[0][0].camera_id, "cam0")


class ReconnectTests(unittest.TestCase):
    """A live stream that drops must be reopened and recover — without a real camera."""

    def test_decoder_recovers_after_drop(self):
        import time
        import numpy as np
        from cvti.serving.streams import StreamDecoder

        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        class _DropCap:      # simulates a dropped stream (read always fails)
            def get(self, p): return 0.0
            def grab(self): return True
            def read(self): return (False, None)
            def release(self): pass

        class _LiveCap:      # healthy stream after reconnect
            def get(self, p): return 0.0
            def grab(self): return True
            def read(self): return (True, frame)
            def release(self): pass

        state = {"n": 0}

        def fake_open():
            state["n"] += 1
            return _DropCap() if state["n"] == 1 else _LiveCap()

        d = StreamDecoder("cam", "rtsp://fake/stream", target_fps=5, reconnect_backoff=0.01)
        d._open = fake_open       # avoid real cv2.VideoCapture
        d.start()
        got = None
        for _ in range(60):
            got = d.read_latest()
            if got is not None:
                break
            time.sleep(0.05)
        d.stop()
        self.assertGreaterEqual(d.reconnects, 1)   # it detected the drop + reopened
        self.assertIsNotNone(got)                  # and recovered a live frame


class Phase1ConcurrentAlertsTests(unittest.TestCase):
    """Two simultaneous RawEvents must yield two CandidateAlerts that are BOTH
    queued and processable — not just candidate_alerts[0]."""

    def test_two_concurrent_events_both_reach_the_gate(self):
        from cvti.contracts import RawEvent
        from cvti.rules.customization import CustomizationEngine

        engine = CustomizationEngine("configs/all_threats_v1.json")
        events = [
            RawEvent(detector="weapons", active=True, title="GUN", level="critical",
                     object_label="gun", timestamp=1.0),
            RawEvent(detector="violence", active=True, title="FIGHT", level="critical",
                     person_id=3, timestamp=1.0),
        ]
        candidates = engine.evaluate(events)
        self.assertGreaterEqual(len(candidates), 2)  # both rules fire, not just one

        q = AlertQueue(cooldown_seconds=8.0)
        for c in candidates:
            q.add(QueuedAlert(camera_id="main", rule_name=c.rule_name, priority=c.priority,
                              title=c.title, timestamp=1.0, track_id=c.person_id,
                              object_label=c.object_label, payload=c))
        drained = q.drain(max_per_drain=10)
        self.assertEqual(len(drained), len(candidates))  # every concurrent alert processed

    def test_object_label_distinguishes_weapon_alerts(self):
        q = AlertQueue(cooldown_seconds=8.0)
        base = dict(camera_id="main", rule_name="weapon_sighting", priority="critical",
                    title="WEAPON", timestamp=0.0, track_id=None)
        self.assertTrue(q.add(QueuedAlert(**base, object_label="gun")))
        self.assertTrue(q.add(QueuedAlert(**base, object_label="knife")))   # different object
        self.assertFalse(q.add(QueuedAlert(**base, object_label="gun")))    # dup within cooldown
        self.assertEqual(q.pending_count, 2)


class CameraMappingTests(unittest.TestCase):
    def test_candidate_maps_to_queued_with_evidence(self):
        from cvti.contracts import CandidateAlert
        from cvti.serving.camera import _to_queued

        cand = CandidateAlert(rule_name="after_hours_in_aisle", priority="high",
                              detector="presence", title="PERSON IN ZONE SHELF_RIGHT",
                              person_id=8, object_label=None, timestamp=2.5)
        qa = _to_queued("aisle_cam_2", cand, 2.5, "shelf_right", frames=["F"],
                        scene={"environment_type": "retail_shop"})
        self.assertEqual(qa.camera_id, "aisle_cam_2")
        self.assertEqual(qa.track_id, 8)
        self.assertEqual(qa.zone, "shelf_right")
        self.assertEqual(qa.payload["candidate"], cand)
        self.assertEqual(qa.payload["frames"], ["F"])


if __name__ == "__main__":
    unittest.main()


class AuditHardeningTests(unittest.TestCase):
    """Pins for the 23 Aug head-to-toe audit's fixed findings."""

    def test_watchdog_treats_long_lived_exit_as_scheduled(self):
        # The spawn caps --seconds (~28h); an exit after an hour+ of uptime is
        # a scheduled restart and must not burn crash budget. (finding #1)
        import inspect
        from cvti.app.console_backend import ConsoleBackend
        src = inspect.getsource(ConsoleBackend._start_watchdog)
        self.assertIn("self._restarts = 0", src)
        self.assertIn("3600", src)

    def test_gate_base_url_reaches_the_ollama_call(self):
        # --gate-base-url was silently ignored by the default provider (finding #4)
        import inspect
        from cvti.verification import gate as g
        self.assertIn("base_url", inspect.signature(g._call_ollama).parameters)
        src = inspect.getsource(g.VerificationGate._call_provider)
        self.assertIn("base_url=self.base_url", src)

    def test_a_preset_change_keeps_zone_and_english_rules(self):
        # Applying a template used to clobber cam["config"], dropping the
        # loitering + custom-English rules silently. (finding #9)
        import json, os, tempfile
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from _backend_helper import signed_in
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        (tmp / "site.json").write_text(json.dumps({"cameras": [{"id": "c1", "source": "0"}]}))
        cwd = os.getcwd()
        os.chdir(tmp)     # zones/rules regen writes under configs/ relative
        try:
            be = signed_in("owner", site_path=str(tmp / "site.json"),
                           db_path=str(tmp / "events.db"), enable_demo=False)
            be.set_custom_rule("c1", "Is anyone wearing a black hoodie?")
            be.apply_template("office")
            cam = json.loads((tmp / "site.json").read_text())["cameras"][0]
            from cvti.serving.custom_rules import _rules_for
            descs = [t["description"] for t in _rules_for(cam)]
            self.assertTrue(any("hoodie" in d for d in descs),
                            "template application dropped the customer's English rule")
        finally:
            os.chdir(cwd)
