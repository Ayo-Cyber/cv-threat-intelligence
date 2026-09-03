"""RAM-growth audit pins (24 Aug): every structure that grew per-person or
per-alert forever now sheds. Each test simulates the growth driver directly —
thousands of track ids — and asserts the structure stays bounded."""
import time
import unittest
from collections import deque


class PoseHistoryTest(unittest.TestCase):
    def test_departed_tracks_leave_the_history(self):
        from cvti.detector.core import enrich_pose_people_with_history

        class P:  # minimal PosePersonState stand-in
            def __init__(self, tid):
                self.track_id = tid
                self.left_wrist = self.right_wrist = (0.0, 0.0)
                self.timestamp = time.time()
                self.max_wrist_speed = 0.0
                self.max_wrist_accel = 0.0
        history: dict = {}
        for tid in range(2000):                       # 2000 visitors, one at a time
            enrich_pose_people_with_history([P(tid)], history)
        self.assertLessEqual(len(history), 1,
                             "one dict entry per person who EVER walked past")


class RunningTracksTest(unittest.TestCase):
    def test_stale_tracks_are_swept(self):
        from cvti.detector.situational import RunningPanicDetector
        det = RunningPanicDetector()
        t0 = 1000.0
        for tid in range(2000):
            det.update(tid, (0, 0, 10, 20), t0 + tid * 31.0, (480, 640, 3))
        self.assertLess(len(det._tracks), 5, "departed visitors never swept")


class FallFiredTest(unittest.TestCase):
    def test_fired_latches_die_with_their_tracks(self):
        from cvti.detector.fall import FallDetector
        det = FallDetector(min_frames=1)
        t = 0.0
        for tid in range(500):
            for _ in range(2):                      # enough to fire the latch
                t += 1.0
                det.update([(tid, 0, 0, 100, 20)], 640 * 480, t)
            det.update([], 640 * 480, t + 0.1)       # track vanishes
        self.assertEqual(len(det._fired), 0, "latches for departed tracks leaked")
        self.assertEqual(len(det._streak), 0)


class AlertQueueBoundsTest(unittest.TestCase):
    def _alert(self, tid, ts, clip=False):
        from cvti.serving.alert_queue import QueuedAlert
        payload = {"frames": [], "enqueued_at": None}
        if clip:
            payload["clip_frames"] = [b"x" * 1000] * 10
        return QueuedAlert(camera_id="c", rule_name="r", priority="high",
                           title="t", timestamp=ts, track_id=tid, payload=payload)

    def test_dedup_signatures_are_pruned_past_cooldown(self):
        from cvti.serving.alert_queue import AlertQueue
        q = AlertQueue(cooldown_seconds=60)
        for tid in range(5000):
            q.add(self._alert(tid, ts=float(tid * 61)))
            q.drain(4)                                # keep pending small
        self.assertLess(len(q._last_seen), 4 * q.max_pending + 100,
                        "one dedup signature per alert, forever")

    def test_backlog_past_half_cap_sheds_clip_frames_oldest_first(self):
        from cvti.serving.alert_queue import AlertQueue
        q = AlertQueue(cooldown_seconds=0)
        n = q.max_pending // 2 + 20
        for tid in range(n):
            q.add(self._alert(tid, ts=float(tid), clip=True))
        stripped = [a for a in q._pending if not a.payload.get("clip_frames")]
        kept = [a for a in q._pending if a.payload.get("clip_frames")]
        self.assertTrue(stripped, "no clips shed past half cap")
        self.assertTrue(kept, "shed everything — the front of the drain order "
                              "should keep clips")
        # drain order is (-priority, timestamp): the next-to-verify alert must
        # keep its clip — its evidence is about to be used.
        next_up = min(q._pending)
        self.assertTrue(next_up.payload.get("clip_frames"),
                        "the next alert to verify lost its clip")


class CaseBookTest(unittest.TestCase):
    def test_closed_cases_are_deleted_after_one_report(self):
        from cvti.serving.watches import CaseBook
        book = CaseBook(stale_after=1.0)
        now = 1000.0
        for tid in range(3000):
            book.observe("cam", "red hoodie", tid, now=now + tid * 0.001)
        book.expire(now=now + 100)                    # closes all
        book.expire(now=now + 101)                    # deletes all closed
        self.assertEqual(len(book._cases), 0,
                         "one dead Case per person who ever matched a watch")


class ScannerCooldownTest(unittest.TestCase):
    def test_cooldown_keys_for_deleted_rules_are_pruned(self):
        import json, tempfile
        from pathlib import Path
        from cvti.serving.custom_rules import CustomRuleScanner
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site.json"
            site.write_text(json.dumps({"cameras": [{"id": "c", "source": "x.mp4",
                "custom_rules": [{"question": "Old rule?"}]}]}))
            sc = CustomRuleScanner([], sink=None, model="m", site_config_path=str(site))
            sc._refresh_cameras()
            sc._incidents[("c", "old rule")] = {"opened_at": 1.0, "last_seen": 1.0,
                                                "misses": 0, "reminders": 0,
                                                "next_reminder_at": 2.0}
            site.write_text(json.dumps({"cameras": [{"id": "c", "source": "x.mp4",
                "custom_rules": [{"question": "New rule?"}]}]}))
            sc._refresh_cameras()
            self.assertNotIn(("c", "old rule"), sc._incidents)


if __name__ == "__main__":
    unittest.main()
