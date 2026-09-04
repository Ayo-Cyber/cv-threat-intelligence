"""Per-stage performance truth — the numbers behind "it feels slow".

Every pilot conversation about speed so far has been adjectives ("slow",
"laggy", "a speed up") because the product measured nothing a customer could
send us. This module is the fix: each pipeline stage reports how long its unit
of work took, the board keeps a rolling window per (stage, camera), and the
engine writes the percentiles to perf_report.json beside gate_health.json.
The diagnostics bundle ships that file, so a support zip answers WHERE the
time goes — decode, detect, verification queue, model inference, or the
English-rule scanner — instead of inviting another guess.

Observations are a lock-guarded deque append: cheap enough to run always-on.
Nothing here may ever raise into the pipeline.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

from cvti.logging_setup import get_logger

log = get_logger(__name__)

WINDOW = 512          # per-(stage, key) samples kept; enough for honest p95s


class PerfBoard:
    """Rolling per-stage latency windows, summarised on demand."""

    def __init__(self, window: int = WINDOW) -> None:
        self._lock = threading.Lock()
        self._series: dict[tuple[str, str], deque] = {}
        self._window = window
        self._started = time.time()

    def observe(self, stage: str, key: str, ms: float) -> None:
        """Record one unit of work for `stage` on `key` (usually a camera id)."""
        try:
            with self._lock:
                series = self._series.get((stage, key))
                if series is None:
                    series = self._series[(stage, key)] = deque(maxlen=self._window)
                series.append((time.time(), float(ms)))
        except Exception:  # noqa: BLE001 - a metric must never hurt the pipeline
            log.debug("perf observation dropped", exc_info=True)

    @staticmethod
    def _percentile(values: list[float], frac: float) -> float:
        idx = min(len(values) - 1, max(0, int(round(frac * (len(values) - 1)))))
        return values[idx]

    def snapshot(self) -> dict:
        """{stage: {key: {count, mean_ms, p50_ms, p95_ms, max_ms, last_ms}}}."""
        with self._lock:
            items = {k: list(v) for k, v in self._series.items()}
        out: dict[str, dict] = {}
        for (stage, key), samples in sorted(items.items()):
            values = sorted(ms for _, ms in samples)
            if not values:
                continue
            out.setdefault(stage, {})[key] = {
                "count": len(values),
                "mean_ms": round(sum(values) / len(values), 1),
                "p50_ms": round(self._percentile(values, 0.50), 1),
                "p95_ms": round(self._percentile(values, 0.95), 1),
                "max_ms": round(values[-1], 1),
                "last_ms": round(samples[-1][1], 1),
                "last_at": round(samples[-1][0], 3),
            }
        return out


def _system() -> dict:
    """CPU and memory context for the same instant as the stage numbers —
    'verification is slow' means something different at 95% CPU."""
    out: dict = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_total_gb": round(vm.total / 2**30, 2),
            "memory_available_gb": round(vm.available / 2**30, 2),
            "memory_percent": vm.percent,
        }
        try:
            la = psutil.getloadavg()
            out["loadavg_1m"] = round(la[0], 2)
        except (AttributeError, OSError):
            pass
    except Exception:  # noqa: BLE001 - a probe must never hurt the report
        log.debug("system probe unavailable for perf report", exc_info=True)
    return out


BOARD = PerfBoard()


def write_report(output_dir: str | Path) -> Path | None:
    """Write perf_report.json next to gate_health.json. Best-effort."""
    try:
        doc = {
            "generated_at": time.time(),
            "window_per_stage": WINDOW,
            "stages": BOARD.snapshot(),
            "system": _system(),
        }
        target = Path(output_dir) / "perf_report.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(doc, indent=1))
        temporary.replace(target)
        return target
    except Exception:  # noqa: BLE001 - reporting must never hurt monitoring
        log.debug("perf report write failed", exc_info=True)
        return None
