"""W5 async enrichment: instant alerts get their words after the fact.

The bypass tier fires violence/dwell alerts in ~0s — which also stripped them
of the VLM's English description. Enrichment restores the words WITHOUT
restoring the wait: after the alert has fired and persisted, an idle gate
worker describes the evidence and the sink appends it to the stored event;
the API's stream loop turns that into alert.update. The contract these tests
hold: enrichment never gates, never pages, never outranks a real
verification, and its failures cost nothing but the description.
"""
import sqlite3
import threading
import time
import unittest
from unittest import mock

import numpy as np

from cvti.contracts import CandidateAlert
from cvti.serving.alert_queue import AlertQueue, QueuedAlert
from cvti.serving.alert_sink import AlertSink
from cvti.serving.gate_pool import GatePool
from cvti.verification.gate import VerificationGate


def _frame():
    return np.zeros((32, 32, 3), dtype=np.uint8)


def _candidate(detector="violence", rule="violence"):
    return CandidateAlert(rule_name=rule, priority="critical",
                          detector=detector, title="VIOLENCE: altercation",
                          person_id=None, object_label=None, timestamp=0.0)


def _queued(candidate):
    return QueuedAlert(camera_id="cam1", rule_name=candidate.rule_name,
                       priority=candidate.priority, title=candidate.title,
                       timestamp=0.0, track_id=None, zone=None,
                       object_label=None,
                       payload={"candidate": candidate, "frames": [_frame()],
                                "scene": {}, "enqueued_at": time.time()})


class DescribeTest(unittest.TestCase):
    def _gate(self):
        return VerificationGate(provider="ollama")

    def test_describe_returns_trimmed_plain_text(self):
        g = self._gate()
        with mock.patch.object(g, "_call_provider",
                               return_value='  "Two people struggling by the till."  '):
            out = g.describe([_frame()], _candidate())
        self.assertEqual(out, "Two people struggling by the till.")

    def test_describe_failure_is_an_empty_string_not_an_exception(self):
        g = self._gate()
        with mock.patch.object(g, "_call_provider",
                               side_effect=ConnectionError("ollama down")):
            self.assertEqual(g.describe([_frame()], _candidate()), "")

    def test_no_frames_means_no_call(self):
        g = self._gate()
        with mock.patch.object(g, "_call_provider") as call:
            self.assertEqual(g.describe([], _candidate()), "")
        call.assert_not_called()


class _Sink:
    """Verdict sink exposing the annotate seam the pool discovers."""

    def __init__(self):
        self.verdicts = []
        self.annotations = []

    def handle(self, alert, result):
        self.verdicts.append((alert, result))
        return 41 + len(self.verdicts)          # a fresh event id per persist

    def annotate_event(self, event_id, text):
        self.annotations.append((event_id, text))
        return True


class _DescribingGate:
    def __init__(self):
        self.describe_calls = 0

    def describe(self, frames, candidate):
        self.describe_calls += 1
        return "A person lingering by the door."

    def verify(self, frames, candidate, scene, examples=None):
        raise AssertionError("bypassed alerts must never reach verify()")


class PoolEnrichmentTest(unittest.TestCase):
    def _run(self, pool, alerts, want_annotations, timeout=4.0):
        for a in alerts:
            pool.queue.add(a)
        pool.start()
        deadline = time.time() + timeout
        sink = pool.on_verdict.__self__
        while len(sink.annotations) < want_annotations:
            if time.time() > deadline:
                break
            time.sleep(0.01)
        pool._stop.set()
        return sink

    def test_a_bypassed_alert_is_described_after_it_fires(self):
        sink = _Sink()
        gate = _DescribingGate()
        pool = GatePool(AlertQueue(), gate_factory=lambda: gate,
                        on_verdict=sink.handle)
        sink = self._run(pool, [_queued(_candidate())], want_annotations=1)
        self.assertEqual(len(sink.verdicts), 1, "the instant verdict comes first")
        self.assertTrue(sink.verdicts[0][1].confirmed)
        self.assertEqual(sink.annotations,
                         [(42, "A person lingering by the door.")])

    def test_enrichment_can_be_switched_off_site_wide(self):
        sink = _Sink()
        gate = _DescribingGate()
        pool = GatePool(AlertQueue(), gate_factory=lambda: gate,
                        on_verdict=sink.handle, enrich_bypassed=False)
        sink = self._run(pool, [_queued(_candidate())], want_annotations=1,
                         timeout=1.0)
        self.assertEqual(len(sink.verdicts), 1)
        self.assertEqual(sink.annotations, [])

    def test_a_gateless_sink_never_breaks_the_worker(self):
        # default on_verdict (no annotate_event) + gate without describe:
        # jobs are dropped, verdicts still flow
        class _Bare:
            def verify(self, *a, **k):
                raise AssertionError("bypass must not verify")
        pool = GatePool(AlertQueue(), gate_factory=_Bare)
        pool.queue.add(_queued(_candidate()))
        pool.start()
        deadline = time.time() + 3
        while pool.verified == 0 and time.time() < deadline:
            time.sleep(0.01)
        pool._stop.set()
        self.assertEqual(pool.verified, 1)
        self.assertEqual(pool.enriched, 0)


class AnnotateEventTest(unittest.TestCase):
    def _sink(self):
        s = AlertSink.__new__(AlertSink)
        s._db = sqlite3.connect(":memory:", check_same_thread=False)
        s._db.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, reason TEXT)")
        s._db.execute("INSERT INTO events (id, reason) VALUES "
                      "(7, 'VIOLENCE — measured-clean tier, auto-confirmed.')")
        s._db.commit()
        s._lock = threading.Lock()
        return s

    def test_description_is_appended_to_the_stored_reason(self):
        s = self._sink()
        self.assertTrue(s.annotate_event(7, "Two people shoving near the door."))
        reason = s._db.execute("SELECT reason FROM events WHERE id=7").fetchone()[0]
        self.assertIn("auto-confirmed", reason)
        self.assertIn("TrueSight: Two people shoving near the door.", reason)

    def test_re_annotation_replaces_not_stacks(self):
        s = self._sink()
        s.annotate_event(7, "First description.")
        s.annotate_event(7, "Better description.")
        reason = s._db.execute("SELECT reason FROM events WHERE id=7").fetchone()[0]
        self.assertEqual(reason.count("TrueSight:"), 1)
        self.assertIn("Better description.", reason)

    def test_unknown_event_and_empty_text_are_calm_noops(self):
        s = self._sink()
        self.assertFalse(s.annotate_event(999, "text"))
        self.assertFalse(s.annotate_event(7, ""))


if __name__ == "__main__":
    unittest.main()
