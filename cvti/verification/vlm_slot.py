"""One local VLM, one request at a time.

The pilot box died of contention, not of a slow model (11 Sep diagnostics):
the gate's verify, the English-rules scan, scene mapping and the daily
self-test each call the local Ollama server from their own thread. Three or
four vision requests time-sharing a saturated 4-core CPU means NONE finishes
inside its deadline — verify_infer showed a busy-fraction of 1.49 and a
median verdict latency exactly equal to the 360s timeout. Serialized, the
same machine at least finishes what it starts.

This module is the process-wide slot. Callers on the live alert path (the
gate, scene mapping at startup) WAIT for it; opportunistic callers (the
English scan that reruns every ~12s anyway, the daily self-test) SKIP their
cycle when it's busy instead of piling on. The lock is reentrant so a caller
that already holds the slot can call through code that acquires it again
(the self-test wraps gate.verify, which acquires internally).

Cloud gates are not serialized — their parallelism is not ours to budget.
`call_openai_compatible` engages the slot only for local base URLs.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from urllib.parse import urlparse

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]", ""}


class VLMBusy(RuntimeError):
    """The local VLM slot is held by another caller; this cycle should skip.

    Deliberately NOT a transport error: a skipped scan is an outcome to
    record, never an UNVERIFIED alert or a component failure.
    """


_lock = threading.RLock()
# Telemetry (best-effort, unlocked reads are fine for a status panel).
_holder: str = ""
_held_since: float = 0.0
_skips: int = 0


def is_local(base_url: str) -> bool:
    """True when `base_url` points at this machine (or is empty = default local)."""
    if not base_url:
        return True
    host = (urlparse(base_url).hostname or "").lower()
    return host in _LOCAL_HOSTS


@contextmanager
def slot(mode: str = "wait", *, who: str = ""):
    """Hold the local-VLM slot for the duration of one request.

    mode="wait"  — block until free (live path: verifies, scene mapping).
    mode="skip"  — raise VLMBusy immediately if another thread holds it
                   (periodic callers: English scans, the self-test).
    """
    global _holder, _held_since, _skips
    if mode == "skip":
        acquired = _lock.acquire(blocking=False)
        if not acquired:
            _skips += 1
            raise VLMBusy(f"local VLM busy ({_holder or 'another caller'} "
                          f"for {time.monotonic() - _held_since:.0f}s)")
    else:
        _lock.acquire()
    _holder, _held_since = (who or mode), time.monotonic()
    try:
        yield
    finally:
        _holder, _held_since = "", 0.0
        _lock.release()


def stats() -> dict:
    return {"holder": _holder,
            "held_s": round(time.monotonic() - _held_since, 1) if _holder else 0.0,
            "skips": _skips}
