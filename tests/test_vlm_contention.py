"""The 11 Sep pilot collapse, as tests.

Four VLM callers (gate, English scan, scene mapping, self-test) time-shared
one saturated 4-core CPU and NONE ever finished inside its deadline — median
verdict latency was literally the 360s timeout, with 16 alerts queued behind
verdicts that were never going to land. These tests pin the three defences:
the process-wide VLM slot, the gate pool's circuit breaker, and the one-frame
cap for local verdicts.
"""
from __future__ import annotations

import threading
import time
import unittest

from cvti.contracts import CandidateAlert, VerificationResult
from cvti.serving.alert_queue import AlertQueue, QueuedAlert
from cvti.serving.gate_pool import GatePool
from cvti.verification import vlm_slot
from cvti.verification.vlm_slot import VLMBusy, slot


class VLMSlotTest(unittest.TestCase):
    def test_skip_mode_raises_while_another_thread_holds(self):
        started, release = threading.Event(), threading.Event()

        def hold():
            with slot("wait", who="verify"):
                started.set()
                release.wait(5)

        t = threading.Thread(target=hold, daemon=True)
        t.start()
        self.assertTrue(started.wait(5))
        with self.assertRaises(VLMBusy):
            with slot("skip", who="english-scan"):
                pass
        release.set()
        t.join(5)
        # Free again: a skip-mode caller now enters without raising.
        with slot("skip"):
            pass

    def test_reentrant_within_one_thread(self):
        # The self-test wraps gate.verify, which acquires the slot again on
        # the same thread — that must never deadlock or raise.
        with slot("wait", who="self-test"):
            with slot("skip", who="gate"):
                pass

    def test_wait_mode_serializes(self):
        order: list[str] = []

        def worker(name: str):
            with slot("wait", who=name):
                order.append(f"{name}-in")
                time.sleep(0.05)
                order.append(f"{name}-out")

        threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        # Never interleaved: whoever entered first also left first.
        self.assertEqual(order[0].split("-")[0], order[1].split("-")[0])
        self.assertEqual(order[2].split("-")[0], order[3].split("-")[0])

    def test_is_local(self):
        self.assertTrue(vlm_slot.is_local("http://localhost:11434/v1"))
        self.assertTrue(vlm_slot.is_local("http://127.0.0.1:11434/v1"))
        self.assertTrue(vlm_slot.is_local(""))  # empty = the default local host
        self.assertFalse(vlm_slot.is_local("https://openrouter.ai/api/v1"))
        self.assertFalse(vlm_slot.is_local("https://api.openai.com/v1"))


def _candidate(rule="theft", detector="video_action") -> CandidateAlert:
    return CandidateAlert(rule_name=rule, priority="high", detector=detector,
                          title="t", person_id=None, object_label=None,
                          timestamp=time.time())


def _queued(candidate: CandidateAlert) -> QueuedAlert:
    return QueuedAlert(camera_id="cam1", rule_name=candidate.rule_name,
                       priority=candidate.priority, title=candidate.title,
                       timestamp=time.time(),
                       payload={"candidate": candidate, "frames": [], "scene": {}})


class _TransportTimeoutGate:
    """A gate whose model never answers in time — the pilot box, in one class."""
    calls = 0

    def verify(self, frames, candidate, scene, examples=None):
        type(self).calls += 1
        return VerificationResult(
            confirmed=True, confidence=0.0, reason="UNVERIFIED",
            alert_priority=candidate.priority, timestamp="", raw_response="",
            error="transport: TimeoutError: timed out")


class _HealthyGate:
    def verify(self, frames, candidate, scene, examples=None):
        return VerificationResult(
            confirmed=True, confidence=0.9, reason="confirmed",
            alert_priority=candidate.priority, timestamp="", raw_response="{}")


class GateBreakerTest(unittest.TestCase):
    def _pool(self, gate_cls) -> GatePool:
        gate_cls.calls = 0
        return GatePool(AlertQueue(), gate_factory=gate_cls, workers=1,
                        on_verdict=lambda alert, result: None)

    def test_opens_after_consecutive_transport_failures_and_stops_calling(self):
        pool = self._pool(_TransportTimeoutGate)
        pool.start()
        try:
            for i in range(5):
                pool.queue.add(_queued(_candidate(rule=f"theft_{i}")))
            self.assertTrue(pool.drain(timeout=10))
        finally:
            pool.stop()
        # Exactly BREAKER_AFTER verdicts reached the model; the rest were
        # answered instantly by the breaker — and every one still surfaced.
        self.assertEqual(_TransportTimeoutGate.calls, GatePool.BREAKER_AFTER)
        self.assertEqual(pool.unverified, 5)
        self.assertTrue(pool.stats()["breaker"]["open"])
        self.assertEqual(pool.stats()["breaker"]["trips"], 1)

    def test_half_open_probe_closes_on_success(self):
        pool = self._pool(_TransportTimeoutGate)
        # Trip it synchronously via the bookkeeping methods.
        bad = _TransportTimeoutGate().verify([], _candidate(), {})
        pool._breaker_note(bad)
        pool._breaker_note(bad)
        self.assertTrue(pool._breaker_open())
        self.assertTrue(pool._breaker_blocks())
        # Cooldown elapses -> exactly one probe passes, others still blocked.
        pool._breaker_opened_at = time.time() - GatePool.BREAKER_COOLDOWN_S - 1
        self.assertFalse(pool._breaker_blocks())   # the probe
        self.assertTrue(pool._breaker_blocks())    # everyone else, meanwhile
        good = _HealthyGate().verify([], _candidate(), {})
        pool._breaker_note(good)
        self.assertFalse(pool._breaker_open())
        self.assertFalse(pool._breaker_blocks())

    def test_non_transport_errors_do_not_trip_it(self):
        pool = self._pool(_TransportTimeoutGate)
        parse_fail = VerificationResult(
            confirmed=True, confidence=0.0, reason="UNVERIFIED",
            alert_priority="high", timestamp="", raw_response="garbage",
            error="parse: no JSON found")
        for _ in range(4):
            pool._breaker_note(parse_fail)
        self.assertFalse(pool._breaker_open())

    def test_bypass_detectors_flow_while_open(self):
        pool = self._pool(_TransportTimeoutGate)
        verdicts: list = []
        pool.on_verdict = lambda alert, result: verdicts.append(result)
        pool.start()
        try:
            for i in range(3):
                pool.queue.add(_queued(_candidate(rule=f"theft_{i}")))
            self.assertTrue(pool.drain(timeout=10))
            self.assertTrue(pool._breaker_open())
            pool.queue.add(_queued(_candidate(rule="dwell", detector="presence")))
            self.assertTrue(pool.drain(timeout=10))
        finally:
            pool.stop()
        # The deterministic bypass still auto-confirms, breaker or not.
        self.assertEqual(verdicts[-1].raw_response, "bypass")
        self.assertTrue(verdicts[-1].confirmed)


class LocalFrameCapTest(unittest.TestCase):
    def _gate(self, provider, **kw):
        from cvti.verification.gate import VerificationGate
        return VerificationGate(provider=provider, model="m", **kw)

    def _frames(self, n):
        import numpy as np
        return [np.full((8, 8, 3), i, dtype="uint8") for i in range(n)]

    def _verify_and_count(self, gate, n_frames):
        sent = {}

        def fake_provider(prompt, frames_bytes, alert):
            sent["n"] = len(frames_bytes)
            return '{"confirmed": true, "confidence": 0.9, "reason": "ok"}'

        gate._call_provider = fake_provider
        gate.verify(self._frames(n_frames), _candidate())
        return sent["n"]

    def test_local_provider_caps_frames(self):
        from cvti.verification.gate import VerificationGate
        self.assertEqual(self._verify_and_count(self._gate("ollama"), 5),
                         VerificationGate.LOCAL_MAX_FRAMES)

    def test_explicit_one_frame_keeps_the_last_image(self):
        # The last image is the subject crop when one was appended — an
        # explicit cap of 1 must keep IT, not a context frame.
        import numpy as np
        gate = self._gate("ollama", max_frames=1)
        sent = {}

        def fake_provider(prompt, frames_bytes, alert):
            sent["bytes"] = frames_bytes
            return '{"confirmed": true, "confidence": 0.9, "reason": "ok"}'

        gate._call_provider = fake_provider
        frames = self._frames(4)
        frames[-1] = np.full((8, 8, 3), 255, dtype="uint8")  # the "crop"
        gate.verify(frames, _candidate())
        self.assertEqual(len(sent["bytes"]), 1)
        import cv2
        decoded = cv2.imdecode(np.frombuffer(sent["bytes"][0], dtype=np.uint8),
                               cv2.IMREAD_COLOR)
        self.assertGreater(decoded.mean(), 200)  # it's the white crop, not a dark frame

    def test_cloud_provider_keeps_all_frames(self):
        self.assertEqual(self._verify_and_count(self._gate("anthropic"), 5), 5)

    def test_explicit_max_frames_wins(self):
        self.assertEqual(
            self._verify_and_count(self._gate("ollama", max_frames=3), 5), 3)


if __name__ == "__main__":
    unittest.main()
