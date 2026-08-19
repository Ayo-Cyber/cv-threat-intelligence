"""Per-component health — because catching is not handling.

68 `except Exception` handlers keep one bad detector from killing the camera
loop, which is the right instinct. But a swallowed exception leaves no way to
tell "this detector correctly found nothing" apart from "this detector has
thrown on every frame for a week". Both look like silence.

Each component records what it processed and what it threw. The counters are
what the System panel shows, and what makes a persistent failure visible before
a customer notices it instead of after.

    from cvti.health import component

    health = component("detector.concealment")
    try:
        ...
        health.ok()
    except Exception as exc:
        health.failed(exc, log, "processing frame")   # rate-limited, exc_info
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

# A component is only called degraded once there is enough evidence to mean it.
# 1 error in 1 frame is not a 100% failure rate, it is one frame.
DEGRADED_RATE = 0.10
DEGRADED_MIN_SAMPLE = 20

# A failure on every frame must not fill the disk with its own traceback. Log
# the first few, then one in every hundred — the counter carries the true scale.
_LOG_FIRST = 3
_LOG_EVERY = 100

_lock = threading.Lock()
_registry: dict[str, "ComponentHealth"] = {}


@dataclass
class ComponentHealth:
    name: str
    processed: int = 0
    errors: int = 0
    last_error: str = ""
    last_error_at: float = 0.0
    first_error_at: float = 0.0
    suppressed_logs: int = 0
    _seen: dict = field(default_factory=dict, repr=False)

    def ok(self, n: int = 1) -> None:
        self.processed += n

    def failed(self, exc: BaseException, log=None, context: str = "") -> None:
        """Count an error and log it, rate-limited, with the traceback."""
        self.errors += 1
        now = time.time()
        if not self.first_error_at:
            self.first_error_at = now
        self.last_error_at = now
        self.last_error = f"{type(exc).__name__}: {str(exc)[:180]}"

        key = type(exc).__name__
        seen = self._seen.get(key, 0) + 1
        self._seen[key] = seen
        should_log = seen <= _LOG_FIRST or seen % _LOG_EVERY == 0
        if not should_log:
            self.suppressed_logs += 1
            return
        if log is not None:
            extra = f" (occurrence {seen})" if seen > 1 else ""
            where = f" while {context}" if context else ""
            log.error("[%s] failed%s%s", self.name, where, extra, exc_info=True)

    @property
    def error_rate(self) -> float:
        total = self.processed + self.errors
        return (self.errors / total) if total else 0.0

    @property
    def degraded(self) -> bool:
        """Failing often enough, and for long enough, to be worth interrupting someone."""
        return (self.processed + self.errors) >= DEGRADED_MIN_SAMPLE \
            and self.error_rate > DEGRADED_RATE

    def to_dict(self) -> dict:
        return {"name": self.name, "processed": self.processed, "errors": self.errors,
                "error_rate": round(self.error_rate, 4), "degraded": self.degraded,
                "last_error": self.last_error, "last_error_at": self.last_error_at,
                "suppressed_logs": self.suppressed_logs}


def component(name: str) -> ComponentHealth:
    """The health record for `name`, created on first use. Thread-safe."""
    with _lock:
        health = _registry.get(name)
        if health is None:
            health = _registry[name] = ComponentHealth(name=name)
        return health


def snapshot() -> dict:
    """Every component that has been touched, worst error rate first."""
    with _lock:
        items = [h.to_dict() for h in _registry.values()]
    items.sort(key=lambda d: (-d["error_rate"], -d["errors"], d["name"]))
    return {"components": items,
            "degraded": [d["name"] for d in items if d["degraded"]],
            "total_errors": sum(d["errors"] for d in items)}


def reset() -> None:
    """Tests only."""
    with _lock:
        _registry.clear()
