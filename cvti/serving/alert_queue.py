"""Alert queue with dedup + throttle (plan.md Phase 1: top-alert -> queue).

The single-stream detector processes only `candidate_alerts[0]`. With many
cameras that drops real concurrent threats and also floods the VLM. This queue:

- accepts every candidate from every camera,
- collapses duplicates by (camera, rule, track, zone, time-bucket),
- orders by priority so the gate verifies the most critical first,
- hands out at most `max_per_drain` candidates per cycle so a slow VLM gate
  (the true scaling ceiling) is never overwhelmed.

Pure Python and side-effect free so it is fully unit-testable without models.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Higher rank = more urgent; used to order the gate funnel.
_PRIORITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}


@dataclass(order=True)
class QueuedAlert:
    # Fields before `sort_index` are excluded from ordering; heap/sort uses
    # (-rank, timestamp) so criticals surface first, ties break oldest-first.
    sort_index: tuple = field(init=False, compare=True)
    camera_id: str = field(compare=False)
    rule_name: str = field(compare=False)
    priority: str = field(compare=False)
    title: str = field(compare=False)
    timestamp: float = field(compare=False)
    track_id: int | None = field(default=None, compare=False)
    zone: str | None = field(default=None, compare=False)
    object_label: str | None = field(default=None, compare=False)
    payload: Any = field(default=None, compare=False)  # e.g. the CandidateAlert

    def __post_init__(self) -> None:
        rank = _PRIORITY_RANK.get(self.priority, 0)
        self.sort_index = (-rank, self.timestamp)

    def signature(self, bucket_seconds: float) -> tuple:
        bucket = int(self.timestamp // bucket_seconds) if bucket_seconds > 0 else 0
        return (self.camera_id, self.rule_name, self.track_id, self.zone, bucket)


class AlertQueue:
    """Dedup + priority-ordered throttle in front of the VLM gate."""

    # 8s let one incident arrive as a dozen near-identical alerts. An operator
    # reads repeats as noise, so the same camera+rule+subject is collapsed for a
    # minute — one incident, one alert.
    def __init__(self, *, cooldown_seconds: float = 60.0, bucket_seconds: float = 2.0,
                 max_pending: int = 256) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.bucket_seconds = bucket_seconds
        self.max_pending = max_pending
        self._pending: list[QueuedAlert] = []
        self._last_seen: dict[tuple, float] = {}  # dedup signature -> last accept ts
        self.dropped_duplicates = 0
        # Producer (pipeline thread) and consumer (gate pool thread) touch this
        # concurrently, so every mutation is under the lock.
        self._lock = threading.Lock()

    def add(self, alert: QueuedAlert) -> bool:
        """Enqueue unless a matching alert was accepted within the cooldown.

        Returns True if enqueued, False if suppressed as a duplicate.
        """
        # The same (camera, rule, track, zone, object) should not re-fire every
        # frame; suppress within the cooldown. object_label matters for object
        # alerts (weapons) where track_id is often None.
        dedup_key = (alert.camera_id, alert.rule_name, alert.track_id, alert.zone,
                     alert.object_label)
        with self._lock:
            last = self._last_seen.get(dedup_key)
            if last is not None and (alert.timestamp - last) < self.cooldown_seconds:
                self.dropped_duplicates += 1
                return False
            self._last_seen[dedup_key] = alert.timestamp
            # A signature past its cooldown is dead weight (track ids never
            # recur), yet this dict grew one entry per alert forever — the
            # only unbounded structure in the queue. (RAM audit 24 Aug, #2.)
            if len(self._last_seen) > 4 * self.max_pending:
                cutoff = alert.timestamp - self.cooldown_seconds
                for k in [k for k, ts in self._last_seen.items() if ts < cutoff]:
                    self._last_seen.pop(k, None)
            self._pending.append(alert)
            # Bound memory: if a slow gate lets the backlog grow, keep the most urgent.
            if len(self._pending) > self.max_pending:
                self._pending.sort()
                self._pending = self._pending[: self.max_pending]
            # The count cap is byte-blind: 256 alerts can pin gigabytes of clip
            # JPEGs while a slow gate drains. Past half the cap, strip clip
            # frames from the alerts that will be VERIFIED LAST (drain-order
            # tail) — their clips would sit pinned longest. Evidence stills
            # survive; the replay clip is sacrificed before the box swaps.
            # (Audit #3.)
            if len(self._pending) > self.max_pending // 2:
                self._pending.sort()
                shed = 0
                for stale in self._pending[self.max_pending // 2:]:
                    if stale.payload and stale.payload.get("clip_frames"):
                        stale.payload["clip_frames"] = []
                        shed += 1
                if shed:
                    log.info("[queue] backlog past %d: dropped clip frames from %d "
                             "oldest pending alert(s) to bound memory",
                             self.max_pending // 2, shed)
        return True

    def drain(self, max_per_drain: int = 4) -> list[QueuedAlert]:
        """Pop up to `max_per_drain` most-urgent alerts for verification."""
        with self._lock:
            if not self._pending:
                return []
            self._pending.sort()
            out = self._pending[:max_per_drain]
            self._pending = self._pending[max_per_drain:]
        return out

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
