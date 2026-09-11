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

# The measured tier (bakeoff + per-rule breakdown, manifest e56b4277, 9 Sep 2026):
# 'violence' fired on 0 of 170 normal clips, while the VLM rejected 12 of its 79
# true positives — the judge only subtracts there, and it costs ~12s on the one
# alert where seconds matter most. Unlike the weapons rollback above, this entry
# is evidence-based re-entry: runs/eval/kpi/bakeoff_summary.json holds the data.
# A site file can veto or replace it with "verify_bypass": [...] — an empty list
# restores full gating; the deterministic set above always applies regardless.
MEASURED_BYPASS: set[str] = {"violence"}


def bypass_from_site(site: dict) -> set[str] | None:
    """The site file's "verify_bypass" list, or None to accept MEASURED_BYPASS."""
    raw = site.get("verify_bypass")
    return None if raw is None else {str(d) for d in raw}


class GatePool:
    # Circuit breaker (11 Sep pilot collapse): each transport timeout burns the
    # gate's FULL ceiling (up to 360s) of saturated CPU, and on a box where the
    # model can never answer in time, every queued alert pays it again — 16
    # alerts were waiting behind verdicts that were never going to land. After
    # BREAKER_AFTER consecutive transport failures the pool stops calling the
    # model: pending alerts surface UNVERIFIED instantly (fail-visible, same as
    # a timeout, minus the six minutes) and detection gets its CPU back. After
    # the cooldown ONE probe verify goes through; success closes the breaker.
    BREAKER_AFTER = 2
    BREAKER_COOLDOWN_S = 600.0

    def __init__(self, queue: AlertQueue, *, gate_factory: Callable[[], Any],
                 workers: int = 1, min_interval: float = 0.0,
                 on_verdict: VerdictHandler | None = None,
                 examples_provider: Callable[[str, str], list] | None = None,
                 bypass: set[str] | None = None,
                 enrich_bypassed: bool = True) -> None:
        self.queue = queue
        self.gate_factory = gate_factory
        self.workers = max(1, workers)
        # Effective bypass tier: the deterministic set is never removable (there
        # is nothing for a VLM to judge there); the measured tier yields to an
        # explicit site-file choice.
        self.bypass = BYPASS_DETECTORS | (MEASURED_BYPASS if bypass is None
                                          else set(bypass))
        # W5 async enrichment: a bypassed alert fires instantly but carries no
        # English description. When the queue is IDLE, a worker describes the
        # newest bypassed alerts after the fact and the sink annotates the
        # stored event (delivered to clients as alert.update). Strictly
        # best-effort and strictly second-class: a pending verification always
        # wins the worker, and a full deque just forgets the oldest job.
        self.enrich_bypassed = enrich_bypassed
        from collections import deque as _deque
        self._enrich_jobs: _deque = _deque(maxlen=16)
        self.enriched = 0
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
        # breaker state (guarded by _breaker_lock: workers race on it)
        self._breaker_lock = threading.Lock()
        self._consec_transport = 0
        self._breaker_opened_at = 0.0
        self._breaker_probing = False
        self.breaker_trips = 0

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
        det = getattr(candidate, "detector", "")
        why = ("deterministic detector, auto-confirmed (no VLM needed)"
               if det in BYPASS_DETECTORS else
               "measured-clean tier, auto-confirmed (0 false alarms on the eval "
               "normals; the VLM only rejected true positives here)")
        return VerificationResult(
            confirmed=True, confidence=0.99,
            reason=f"{getattr(candidate, 'title', alert.rule_name)} — {why}.",
            alert_priority=alert.priority, timestamp=time.time(), raw_response="bypass")

    # ---- circuit breaker ---------------------------------------------------

    def _breaker_blocks(self) -> bool:
        """True when this verify should NOT reach the model.

        Closed -> False. Open -> True until the cooldown elapses, then exactly
        one worker gets through as the half-open probe; everyone else keeps
        getting blocked until that probe reports back.
        """
        with self._breaker_lock:
            if self._consec_transport < self.BREAKER_AFTER:
                return False
            if time.time() - self._breaker_opened_at < self.BREAKER_COOLDOWN_S:
                return True
            if self._breaker_probing:
                return True
            self._breaker_probing = True    # this caller is the probe
            return False

    def _breaker_note(self, result: Any) -> None:
        """Feed a verdict's outcome back into the breaker."""
        transport_failed = bool(result is not None and getattr(result, "errored", False)
                                and str(getattr(result, "error", "")).startswith("transport:"))
        with self._breaker_lock:
            self._breaker_probing = False
            if not transport_failed:
                if self._consec_transport >= self.BREAKER_AFTER:
                    log.warning("[gate breaker] closed — the model is answering again")
                self._consec_transport = 0
                self._breaker_opened_at = 0.0
                return
            self._consec_transport += 1
            if self._consec_transport >= self.BREAKER_AFTER:
                self._breaker_opened_at = time.time()
                self.breaker_trips += 1
                log.error(
                    "[gate breaker] OPEN after %d consecutive transport failures — "
                    "verification paused for %.0fs; alerts surface UNVERIFIED "
                    "instantly instead of each burning a full timeout",
                    self._consec_transport, self.BREAKER_COOLDOWN_S)

    def _breaker_open(self) -> bool:
        with self._breaker_lock:
            return self._consec_transport >= self.BREAKER_AFTER

    def _breaker_result(self, candidate: Any, alert: QueuedAlert) -> Any:
        """The instant fail-visible verdict handed out while the breaker is open."""
        from cvti.contracts import VerificationResult
        from cvti.verification.gate import UNVERIFIED_REASON
        return VerificationResult(
            confirmed=True, confidence=0.0,
            reason=UNVERIFIED_REASON,
            alert_priority=alert.priority, timestamp=time.time(), raw_response="",
            error=(f"breaker: open after {self._consec_transport} consecutive "
                   "transport failures — verify skipped, alert surfaced unverified"))

    def _enrich_one(self, gate: Any) -> bool:
        """Describe ONE bypassed alert while the verification queue is idle.

        Returns False when there was nothing to do (the worker sleeps as
        before). Runs on the worker's own gate; any failure is logged at
        debug and the job is simply dropped — enrichment must never page,
        never gate, never keep a worker from real verdicts."""
        if not self.enrich_bypassed or not self._enrich_jobs:
            return False
        if self._breaker_open():
            # No opportunistic VLM work while the model can't even answer
            # verdicts — enrichment would burn the same doomed timeout.
            return False
        describe = getattr(gate, "describe", None)
        annotate = getattr(self.on_verdict, "__self__", None)
        annotate = getattr(annotate, "annotate_event", None)
        if describe is None or annotate is None:
            self._enrich_jobs.clear()
            return False
        event_id, frames, candidate = self._enrich_jobs.popleft()
        try:
            text = describe(frames, candidate)
            if text:
                annotate(event_id, text)
                self.enriched += 1
        except Exception:  # noqa: BLE001 - best-effort by contract
            log.debug("bypass enrichment failed for event %s", event_id,
                      exc_info=True)
        return True

    def _worker(self) -> None:
        gate = self.gate_factory()
        while not self._stop.is_set():
            batch = self.queue.drain(max_per_drain=1)
            if not batch:
                if not self._enrich_one(gate):
                    self._stop.wait(0.05)
                continue
            for alert in batch:
                # payload = {"candidate": CandidateAlert, "frames": [...], "scene": {...}}
                p = alert.payload or {}
                self._active += 1
                try:
                    candidate = p.get("candidate")
                    if getattr(candidate, "detector", "") in self.bypass:
                        result = self._bypass(candidate, alert)   # deterministic -> instant
                    else:
                        examples = None
                        if self.examples_provider is not None:
                            try:
                                examples = self.examples_provider(alert.camera_id, alert.rule_name)
                            except Exception as exc:  # noqa: BLE001 - feedback lookup must never break the gate
                                log.debug("feedback example lookup failed", exc_info=True)
                                examples = None
                        from cvti.serving.perf import BOARD
                        # How long the alert sat in the queue before a worker
                        # picked it up — on a starved box this, not inference,
                        # is usually the real "verification is slow".
                        # Wall clock comes from the payload's enqueued_at:
                        # alert.timestamp is often the FRAME time (seconds into
                        # a stream), and now-minus-frame-time printed epoch-sized
                        # waits in the field perf report (10 Sep diagnostics).
                        _enq = p.get("enqueued_at") or None
                        if _enq:
                            BOARD.observe("verify_wait", alert.camera_id,
                                          max(0.0, (time.time() - _enq) * 1000.0))
                        if self._breaker_blocks():
                            result = self._breaker_result(candidate, alert)
                        else:
                            _t0 = time.monotonic()
                            result = gate.verify(p.get("frames"), candidate, p.get("scene"),
                                                 examples=examples)
                            _dur = time.monotonic() - _t0
                            self._latencies.append(_dur)
                            BOARD.observe("verify_infer", alert.camera_id, _dur * 1000.0)
                            self._breaker_note(result)
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
                    with self._breaker_lock:
                        self._breaker_probing = False   # a crashed probe must not wedge the breaker
                    self.errors += 1
                    self._health.failed(exc, log, f"verifying {alert.rule_name}")
                    self.last_error = f"{alert.camera_id}::{alert.rule_name} — {str(exc)[:160]}"
                    self.last_error_at = time.time()
                    log.error(f"[gate error] {self.last_error}", exc_info=True)
                    result = None
                finally:
                    self._active -= 1
                event_id = self.on_verdict(alert, result)
                if (self.enrich_bypassed and event_id
                        and getattr(result, "raw_response", "") == "bypass"):
                    self._enrich_jobs.append(
                        (event_id, p.get("frames"), p.get("candidate")))
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
                "deduped": self.queue.dropped_duplicates, "pending": self.queue.pending_count,
                "breaker": {"open": self._breaker_open(),
                            "consecutive_transport_failures": self._consec_transport,
                            "opened_at": self._breaker_opened_at,
                            "trips": self.breaker_trips,
                            "cooldown_s": self.BREAKER_COOLDOWN_S}}
