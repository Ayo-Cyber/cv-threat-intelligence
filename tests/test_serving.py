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
