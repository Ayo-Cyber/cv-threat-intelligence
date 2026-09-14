from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.contracts import CandidateAlert
from cvti.serving.alert_queue import AlertQueue, QueuedAlert
from cvti.serving.alert_sink import AlertSink, build_notifier


@dataclass
class _Result:
    confirmed: bool
    confidence: float
    reason: str
    error: str = ""
    prompt_version: str = "prompt123"

    @property
    def errored(self):
        return bool(self.error)


class _RecordingNotifier:
    def __init__(self):
        self.events = []

    def notify(self, event):
        self.events.append(event)


def _alert(cam="cam0", rule="shoplifting", track=3, priority="high", ts=4.25):
    candidate = CandidateAlert(
        rule_name=rule, priority=priority, detector="concealment",
        title="POSSIBLE CONCEALMENT (bag)", person_id=track,
        object_label=None, timestamp=ts,
        reasons=["hand reached a personal bag"],
        metadata={
            "destination": "bag", "score": 0.84,
            "components": {"f_waist": 0.1, "f_bag": 0.9,
                           "f_retract": 0.8, "f_dwell": 0.75},
            "reasons": ["hand reached a personal bag"],
            "limited": False,
            "associated_bag": (180.0, 170.0, 240.0, 235.0),
        },
    )
    return QueuedAlert(camera_id=cam, rule_name=rule, priority=priority, title="T",
                       timestamp=ts, track_id=track, zone="shelf", object_label=None,
                       payload={"candidate": candidate, "frames": [], "scene": None,
                                "enqueued_at": time.time() - 0.25})


def _motion_alert(cam="S5-P01", ts=2.0, priority="high"):
    candidate = CandidateAlert(
        rule_name="chi_multiple_people_moving",
        priority=priority,
        detector="multiple_people_moving",
        title="MULTIPLE PEOPLE MOVING",
        person_id=None,
        object_label=None,
        timestamp=ts,
        metadata={"track_ids": [1, 2], "people_count": 2},
    )
    return QueuedAlert(
        camera_id=cam,
        rule_name=candidate.rule_name,
        priority=priority,
        title=candidate.title,
        timestamp=ts,
        payload={
            "candidate": candidate,
            "frames": [],
            "scene": None,
            "enqueued_at": time.time() - 0.1,
        },
    )


def _object_alert(
    cam="cam1",
    ts=12.5,
    priority="high",
    object_id="chi-carton",
    object_label="Chi carton",
):
    candidate = CandidateAlert(
        rule_name="chi_product_removed_from_storage",
        priority=priority,
        detector="object_watch",
        title="CHI CARTON OBJECT REMOVED",
        person_id=None,
        object_label=object_label,
        timestamp=ts,
        reasons=["object disappeared after being stable"],
        metadata={
            "object_id": object_id,
            "object_category": "product",
            "state": "object_removed",
            "zone": "storage",
            "track_id": 7,
            "bbox": (1, 2, 30, 40),
            "similarity": 0.84,
            "dwell_seconds": 2.0,
            "reasons": ["object disappeared after being stable"],
        },
    )
    return QueuedAlert(
        camera_id=cam,
        rule_name=candidate.rule_name,
        priority=priority,
        title=candidate.title,
        timestamp=ts,
        track_id=None,
        zone="storage",
        object_label=object_label,
        payload={
            "candidate": candidate,
            "frames": [],
            "scene": None,
            "enqueued_at": time.time() - 0.1,
        },
    )


class AlertSinkTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.notifier = _RecordingNotifier()
        self.sink = AlertSink(self._tmp.name, notifier=self.notifier, save_evidence=False)

    def tearDown(self):
        self.sink.close()

    def _audit_rows(self):
        con = sqlite3.connect(self.sink.db_path)
        con.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in con.execute(
                "SELECT * FROM concealment_audit ORDER BY id"
            )]
        finally:
            con.close()
        self._tmp.cleanup()

    def test_confirmed_persists_and_notifies(self):
        self.sink.handle(_alert(), _Result(confirmed=True, confidence=0.9, reason="clear theft"))
        self.assertEqual(self.sink.persisted, 1)
        self.assertEqual(len(self.notifier.events), 1)
        self.assertEqual(self.notifier.events[0]["camera_id"], "cam0")
        # DB row written
        rows = sqlite3.connect(self.sink.db_path).execute(
            "SELECT camera_id, rule, confidence FROM events").fetchall()
        self.assertEqual(rows, [("cam0", "shoplifting", 0.9)])
        # evidence dir + event.json written
        ev_dirs = list((Path(self._tmp.name) / "events").iterdir())
        self.assertEqual(len(ev_dirs), 1)
        self.assertTrue((ev_dirs[0] / "event.json").exists())

    def test_rejected_does_not_persist_or_notify(self):
        self.sink.handle(_alert(), _Result(confirmed=False, confidence=0.1, reason="normal"))
        self.assertEqual(self.sink.persisted, 0)
        self.assertEqual(self.notifier.events, [])
        rows = sqlite3.connect(self.sink.db_path).execute("SELECT id FROM events").fetchall()
        self.assertEqual(rows, [])
        audit = self._audit_rows()[0]
        self.assertEqual(audit["verdict"], "rejected")
        self.assertEqual(audit["candidate_timestamp"], 4.25)
        self.assertEqual(audit["track_id"], 3)
        self.assertEqual(audit["destination"], "bag")
        self.assertEqual(audit["peak_score"], 0.84)
        self.assertIn('"f_bag":0.9', audit["components_json"])
        self.assertEqual(audit["limited"], 0)
        self.assertEqual(audit["associated_bag_json"], "[180.0,170.0,240.0,235.0]")
        self.assertGreaterEqual(audit["gate_latency_s"], 0.2)

    def test_unverified_concealment_is_audited_without_a_user_alert(self):
        self.sink.handle(
            _alert(),
            _Result(confirmed=False, confidence=0.0, reason="provider unavailable",
                    error="connection refused"),
        )
        self.assertEqual(self.sink.persisted, 0)
        self.assertEqual(self.notifier.events, [])
        audit = self._audit_rows()[0]
        self.assertEqual(audit["verdict"], "unverified")
        self.assertEqual(audit["gate_error"], "connection refused")

    def test_confirmed_concealment_is_audited_and_still_persisted(self):
        self.sink.handle(
            _alert(), _Result(confirmed=True, confidence=0.91, reason="concealment visible")
        )
        audit = self._audit_rows()[0]
        self.assertEqual(audit["verdict"], "confirmed")
        self.assertEqual(self.sink.persisted, 1)
        self.assertEqual(len(self.notifier.events), 1)

    def test_missing_result_is_audited_as_unverified(self):
        self.sink.handle(_alert(), None)   # gate error path
        self.assertEqual(self.sink.persisted, 0)
        audit = self._audit_rows()[0]
        self.assertEqual(audit["verdict"], "unverified")
        self.assertEqual(audit["gate_error"], "missing verification result")

    def test_generation_audit_records_admitted_deduplicated_and_capacity_dropped(self):
        queue = AlertQueue(
            cooldown_seconds=60.0,
            max_pending=1,
            on_generated=self.sink.audit_candidate_generated,
            on_admission=self.sink.audit_candidate_admission,
        )
        first = _alert(track=1, priority="low", ts=10.0)
        duplicate = _alert(track=1, priority="low", ts=10.1)
        urgent = _alert(track=2, priority="high", ts=10.2)

        self.assertTrue(queue.add(first))
        self.assertFalse(queue.add(duplicate))
        self.assertTrue(queue.add(urgent))

        rows = self._audit_rows()
        self.assertEqual(len(rows), 3, "every generated candidate must be auditable")
        self.assertEqual(
            [(row["track_id"], row["admission_status"]) for row in rows],
            [(1, "capacity_dropped"), (1, "deduplicated"), (2, "admitted")],
        )

    def test_concurrent_candidate_audit_writes_are_complete(self):
        queue = AlertQueue(
            cooldown_seconds=0.0,
            max_pending=64,
            on_generated=self.sink.audit_candidate_generated,
            on_admission=self.sink.audit_candidate_admission,
        )
        threads = [
            threading.Thread(target=queue.add, args=(_alert(track=i, ts=float(i)),))
            for i in range(24)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        rows = self._audit_rows()
        self.assertEqual(len(rows), 24)
        self.assertEqual({row["track_id"] for row in rows}, set(range(24)))
        self.assertEqual({row["admission_status"] for row in rows}, {"admitted"})

    def test_unserializable_metadata_does_not_drop_the_candidate_audit_row(self):
        alert = _alert()
        cyclic = {}
        cyclic["self"] = cyclic
        alert.payload["candidate"].metadata["components"] = cyclic
        queue = AlertQueue(
            on_generated=self.sink.audit_candidate_generated,
            on_admission=self.sink.audit_candidate_admission,
        )

        self.assertTrue(queue.add(alert))

        row = self._audit_rows()[0]
        self.assertEqual(row["admission_status"], "admitted")
        self.assertIsNotNone(row["audit_error"])

    def test_runtime_motion_audit_is_scorer_compatible_end_to_end(self):
        from tools.score_chi_motion import load_audit_rows

        queue = AlertQueue(
            cooldown_seconds=60.0,
            max_pending=4,
            on_generated=self.sink.audit_candidate_generated,
            on_admission=self.sink.audit_candidate_admission,
        )
        admitted = _motion_alert(ts=2.0)
        duplicate = _motion_alert(ts=2.1)

        self.assertTrue(queue.add(admitted))
        self.assertFalse(queue.add(duplicate))
        self.sink.handle(
            queue.drain()[0],
            _Result(confirmed=True, confidence=0.9, reason="simultaneous movement"),
        )

        rows = load_audit_rows(self.sink.db_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["case_id"], "S5-P01")
        self.assertEqual(rows[0]["timestamp_s"], 2.0)
        self.assertEqual(rows[0]["admission_status"], "admitted")
        self.assertEqual(rows[0]["gate_status"], "confirmed")
        self.assertTrue(rows[0]["persisted_event_id"])
        self.assertEqual(rows[1]["admission_status"], "deduplicated")
        self.assertEqual(rows[1]["gate_status"], "not_gated")
        self.assertEqual(rows[1]["persisted_event_id"], "")

    def test_motion_audit_records_capacity_gate_error_and_persistence_outcomes(self):
        queue = AlertQueue(
            cooldown_seconds=0.0,
            max_pending=1,
            on_generated=self.sink.audit_candidate_generated,
            on_admission=self.sink.audit_candidate_admission,
        )
        displaced = _motion_alert(cam="S5-P01", ts=1.0, priority="low")
        errored = _motion_alert(cam="S5-N01", ts=2.0, priority="high")
        self.assertTrue(queue.add(displaced))
        self.assertTrue(queue.add(errored))
        self.sink.handle(
            queue.drain()[0],
            _Result(
                confirmed=False,
                confidence=0.0,
                reason="provider unavailable",
                error="connection refused",
            ),
        )

        con = sqlite3.connect(self.sink.db_path)
        con.row_factory = sqlite3.Row
        rows = [dict(row) for row in con.execute(
            "SELECT * FROM motion_candidate_audit ORDER BY generated_at"
        )]
        con.close()
        self.assertEqual(rows[0]["admission_status"], "capacity_dropped")
        self.assertEqual(rows[0]["gate_status"], "not_gated")
        self.assertEqual(rows[1]["admission_status"], "admitted")
        self.assertEqual(rows[1]["gate_status"], "unverified")
        self.assertEqual(rows[1]["gate_error"], "connection refused")
        self.assertEqual(rows[1]["persistence_status"], "not_applicable")

    def test_object_watch_audit_lifecycle_records_all_queue_and_gate_outcomes(self):
        queue = AlertQueue(
            cooldown_seconds=60.0,
            max_pending=1,
            on_generated=self.sink.audit_candidate_generated,
            on_admission=self.sink.audit_candidate_admission,
        )
        displaced = _object_alert(ts=10.0, priority="medium")
        duplicate = _object_alert(ts=10.1, priority="medium")
        admitted = _object_alert(
            ts=10.2, priority="high",
            object_id="chi-crate", object_label="Chi crate",
        )

        self.assertTrue(queue.add(displaced))
        self.assertFalse(queue.add(duplicate))
        self.assertTrue(queue.add(admitted))
        self.sink.handle(
            queue.drain()[0],
            _Result(confirmed=True, confidence=0.92, reason="product is gone"),
        )

        con = sqlite3.connect(self.sink.db_path)
        con.row_factory = sqlite3.Row
        rows = [dict(row) for row in con.execute(
            "SELECT * FROM object_watch_audit ORDER BY generated_at"
        )]
        con.close()

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["admission_status"] for row in rows],
            ["capacity_dropped", "deduplicated", "admitted"],
        )
        self.assertEqual(rows[2]["gate_status"], "confirmed")
        self.assertTrue(rows[2]["persisted_event_id"])
        self.assertEqual(rows[2]["object_id"], "chi-crate")
        self.assertEqual(rows[2]["object_label"], "Chi crate")
        self.assertEqual(rows[2]["state"], "object_removed")
        self.assertEqual(rows[2]["zone"], "storage")
        self.assertEqual(rows[2]["similarity"], 0.84)
        self.assertIn('"bbox":[1,2,30,40]', rows[2]["payload_json"])

    def test_object_watch_rejected_and_unverified_verdicts_are_audited(self):
        rejected = _object_alert(ts=20.0)
        rejected_id = self.sink.audit_candidate_generated(rejected)
        rejected.payload["object_watch_audit_id"] = rejected_id
        self.sink.audit_candidate_admission(rejected, "admitted")
        self.sink.handle(
            rejected,
            _Result(confirmed=False, confidence=0.2, reason="different product"),
        )

        unverified = _object_alert(ts=21.0)
        unverified_id = self.sink.audit_candidate_generated(unverified)
        unverified.payload["object_watch_audit_id"] = unverified_id
        self.sink.audit_candidate_admission(unverified, "admitted")
        self.sink.handle(
            unverified,
            _Result(confirmed=False, confidence=0.0, reason="provider unavailable",
                    error="connection refused"),
        )

        con = sqlite3.connect(self.sink.db_path)
        con.row_factory = sqlite3.Row
        rows = [dict(row) for row in con.execute(
            "SELECT gate_status, admission_status, persisted_event_id FROM object_watch_audit "
            "ORDER BY generated_at"
        )]
        con.close()

        self.assertEqual(
            [(row["gate_status"], row["admission_status"], row["persisted_event_id"])
             for row in rows],
            [("rejected", "admitted", ""), ("unverified", "admitted", "")],
        )


class VideoClipTests(unittest.TestCase):
    def test_write_video_clip_is_real_realtime_video(self):
        import tempfile
        import cv2
        import numpy as np
        from pathlib import Path
        from cvti.serving.alert_sink import AlertSink
        jpegs = []
        for i in range(12):                       # 12 continuous frames at src ~4fps
            fr = np.full((120, 160, 3), 20, np.uint8)
            cv2.putText(fr, str(i), (60, 70), 0, 2, (0, 200, 255), 3)
            ok, b = cv2.imencode(".jpg", fr)
            jpegs.append(b.tobytes())
        sink = AlertSink.__new__(AlertSink)       # only need the method
        out = Path(tempfile.mkdtemp()) / "clip.mp4"
        sink._write_video_clip(out, jpegs, src_fps=4.0, container_fps=24)
        self.assertTrue(out.exists())
        cap = cv2.VideoCapture(str(out))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertEqual(n, 12 * 6)               # each source frame held 24/4 = 6x
        # -> 72 frames / 24fps = 3.0s == 12 frames / 4fps real-time. A real video.


class NotifierFactoryTests(unittest.TestCase):
    def test_build_notifier_variants(self):
        from cvti.serving.alert_sink import ConsoleNotifier, TelegramNotifier, WebhookNotifier
        self.assertIsInstance(build_notifier("console"), ConsoleNotifier)
        self.assertIsInstance(build_notifier(""), ConsoleNotifier)
        self.assertIsInstance(build_notifier("webhook:https://example.com/hook"), WebhookNotifier)
        tg = build_notifier("telegram:12345:67890")
        self.assertIsInstance(tg, TelegramNotifier)
        self.assertIn("12345", tg.base)
        self.assertEqual(tg.chat_id, "67890")


if __name__ == "__main__":
    unittest.main()
