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

# Called on every verdict: (alert, VerificationResult) -> None
VerdictHandler = Callable[[QueuedAlert, Any], None]


class GatePool:
    def __init__(self, queue: AlertQueue, *, gate_factory: Callable[[], Any],
                 workers: int = 1, min_interval: float = 0.0,
                 on_verdict: VerdictHandler | None = None) -> None:
        self.queue = queue
        self.gate_factory = gate_factory
        self.workers = max(1, workers)
        self.min_interval = min_interval
        self.on_verdict = on_verdict or self._default_verdict
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # stats (single-worker-safe; with >1 worker these are approximate)
        self.verified = 0
        self.confirmed = 0
        self.rejected = 0
        self.errors = 0

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
        print(f"[{tag}] {alert.camera_id} :: {alert.rule_name} ({alert.priority.upper()}) "
              f"— {alert.title} | conf={result.confidence:.2f} | {result.reason}")

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
                try:
                    result = gate.verify(p.get("frames"), p.get("candidate"), p.get("scene"))
                    self.verified += 1
                    if result is not None and result.confirmed:
                        self.confirmed += 1
                    else:
                        self.rejected += 1
                except Exception as exc:  # noqa: BLE001 - a gate error must not kill the worker
                    self.errors += 1
                    print(f"[gate error] {alert.camera_id}::{alert.rule_name} — {str(exc)[:120]}")
                    result = None
                self.on_verdict(alert, result)
                if self.min_interval:
                    self._stop.wait(self.min_interval)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)

    def stats(self) -> dict:
        return {"verified": self.verified, "confirmed": self.confirmed,
                "rejected": self.rejected, "errors": self.errors,
                "deduped": self.queue.dropped_duplicates, "pending": self.queue.pending_count}
