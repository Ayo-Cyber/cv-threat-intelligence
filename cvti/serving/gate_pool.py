"""Async VLM gate pool (plan.md Phase 8.3).

The local VLM gate is the scaling ceiling (~1-3s/verify). Running it inline
would stall detection for every camera. Instead, worker threads drain the
shared AlertQueue and verify out-of-band, rate-limited so a slow gate is never
overwhelmed. The detection loop only ever does the cheap `queue.add()`.

Each worker owns its own VerificationGate (call-count/save state is per worker).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from cvti.serving.alert_queue import AlertQueue, QueuedAlert
from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Called on every verdict: (alert, VerificationResult) -> None
VerdictHandler = Callable[[QueuedAlert, Any], None]

# Hardened policy: EVERYTHING goes through the VLM gate. Nothing auto-confirms.
# (Weapons/tamper used to bypass here, but the weapon model over-fires and a
# bypassed false positive becomes a confirmed alert — exactly the noise the gate
# exists to kill. The VLM now verifies covered-camera and weapon claims too, with
# detector-specific gate questions.) Left as an empty, overridable set so a
# genuinely deterministic detector could opt back in later if ever needed.
# 'presence' (loitering/dwell) is a deterministic geometric fact — someone stood in
# the zone past the dwell threshold. Auto-confirm it instead of asking the VLM (which
# would reject "a person standing on a street" as not-a-threat). Demo/loitering path.
# "camera_offline" joins it for a different reason: there is no frame to
# verify. The camera is unreachable — that IS the observation.
BYPASS_DETECTORS: set[str] = {"presence", "camera_offline"}


class GatePool:
    def __init__(self, queue: AlertQueue, *, gate_factory: Callable[[], Any],
                 workers: int = 1, min_interval: float = 0.0,
                 on_verdict: VerdictHandler | None = None,
                 examples_provider: Callable[[str, str], list] | None = None) -> None:
        self.queue = queue
        self.gate_factory = gate_factory
        self.workers = max(1, workers)
        self.min_interval = min_interval
        self.on_verdict = on_verdict or self._default_verdict
        # feedback loop: (camera, rule) -> recent operator-labeled examples for the gate
        self.examples_provider = examples_provider
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._active = 0  # verdicts currently in flight (for graceful drain)
        # stats (single-worker-safe; with >1 worker these are approximate)
        self.verified = 0
        self.confirmed = 0
        self.rejected = 0
        self.errors = 0
        # Last failure, kept for the System panel: a count alone doesn't tell an
        # operator whether Ollama is down or the model is returning garbage.
        self.last_error = ""
        self.last_error_at = 0.0
        self.last_success_at = 0.0
        # Verdicts the gate could NOT reach (fail-visible surfaces them as
        # UNVERIFIED alerts rather than raising). They are deliveries, not
        # verifications — counting them as verified would let /health report a
        # dead gate as healthy, which is the exact lie EP-04 exists to end.
        self.unverified = 0
        self.last_unverified_at = 0.0
        from cvti.health import component
        self._health = component("gate")
        # Verify latency, last 50 verdicts. /health reports the median: one slow
        # verdict is the model thinking, a slow median is a saturated gate.
        from collections import deque
        self._latencies: deque = deque(maxlen=50)

    def start(self) -> "GatePool":
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, name=f"gate-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def _default_verdict(self, alert: QueuedAlert, result: Any) -> None:
        if result is None:
            return
        tag = "CONFIRMED" if result.confirmed else "REJECTED "
        log.info(f"[{tag}] {alert.camera_id} :: {alert.rule_name} ({alert.priority.upper()}) "
              f"— {alert.title} | conf={result.confidence:.2f} | {result.reason}")

    @staticmethod
    def _bypass(candidate: Any, alert: QueuedAlert) -> Any:
        from cvti.contracts import VerificationResult
        return VerificationResult(
            confirmed=True, confidence=0.99,
            reason=f"{getattr(candidate, 'title', alert.rule_name)} — deterministic detector, auto-confirmed (no VLM needed).",
            alert_priority=alert.priority, timestamp=time.time(), raw_response="bypass")

    def _worker(self) -> None:
        gate = self.gate_factory()
        while not self._stop.is_set():
            batch = self.queue.drain(max_per_drain=1)
            if not batch:
                self._stop.wait(0.05)
                continue
            for alert in batch:
                # payload = {"candidate": CandidateAlert, "frames": [...], "scene": {...}}
                p = alert.payload or {}
                self._active += 1
                try:
                    candidate = p.get("candidate")
                    if getattr(candidate, "detector", "") in BYPASS_DETECTORS:
                        result = self._bypass(candidate, alert)   # deterministic -> instant
                    else:
                        examples = None
                        if self.examples_provider is not None:
                            try:
                                examples = self.examples_provider(alert.camera_id, alert.rule_name)
                            except Exception as exc:  # noqa: BLE001 - feedback lookup must never break the gate
                                log.debug("feedback example lookup failed", exc_info=True)
                                examples = None
                        _t0 = time.monotonic()
                        result = gate.verify(p.get("frames"), candidate, p.get("scene"),
                                             examples=examples)
                        self._latencies.append(time.monotonic() - _t0)
                    if result is not None and getattr(result, "errored", False):
                        # No verdict was reached. The alert was surfaced
                        # UNVERIFIED — that is delivery working, not the gate.
                        self.unverified += 1
                        self.last_unverified_at = time.time()
                        self.last_error = str(result.error)[:180]
                        self.last_error_at = self.last_unverified_at
                        self._health.failed(RuntimeError(result.error), log,
                                            "reaching a verdict")
                    else:
                        self.verified += 1
                        self.last_success_at = time.time()
                        self._health.ok()
                        if result is not None and result.confirmed:
                            self.confirmed += 1
                        else:
                            self.rejected += 1
                except Exception as exc:  # noqa: BLE001 - a gate error must not kill the worker
                    self.errors += 1
                    self._health.failed(exc, log, f"verifying {alert.rule_name}")
                    self.last_error = f"{alert.camera_id}::{alert.rule_name} — {str(exc)[:160]}"
                    self.last_error_at = time.time()
                    log.error(f"[gate error] {self.last_error}", exc_info=True)
                    result = None
                finally:
                    self._active -= 1
                self.on_verdict(alert, result)
                if self.min_interval:
                    self._stop.wait(self.min_interval)

    def drain(self, timeout: float = 120.0) -> bool:
        """Block until the queue is empty and no verdict is in flight, or timeout.

        With a real VLM (~12s/verify) the queue keeps draining after detection
        ends; call this before stop() so confirmed alerts aren't cut off. Returns
        True if fully drained, False if the timeout hit first.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.queue.pending_count == 0 and self._active == 0:
                return True
            time.sleep(0.2)
        return self.queue.pending_count == 0 and self._active == 0

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)

    def median_latency_s(self):
        if not self._latencies:
            return None
        ordered = sorted(self._latencies)
        return round(ordered[len(ordered) // 2], 2)

    def stats(self) -> dict:
        return {"verified": self.verified, "confirmed": self.confirmed,
                "rejected": self.rejected, "errors": self.errors,
                "unverified": self.unverified,
                "last_unverified_at": self.last_unverified_at,
                "last_success_at": self.last_success_at,
                "median_latency_s": self.median_latency_s(),
                "last_error": self.last_error, "last_error_at": self.last_error_at,
                "deduped": self.queue.dropped_duplicates, "pending": self.queue.pending_count}
