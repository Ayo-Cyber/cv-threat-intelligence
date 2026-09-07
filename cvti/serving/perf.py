"""Per-stage performance truth — the numbers behind "it feels slow".

Every pilot conversation about speed so far has been adjectives ("slow",
"laggy", "a speed up") because the product measured nothing a customer could
send us. This module is the fix: each pipeline stage reports how long its unit
of work took, the board keeps a rolling window per (stage, camera), and the
engine writes the percentiles to perf_report.json beside gate_health.json.
The diagnostics bundle ships that file, so a support zip answers WHERE the
time goes — decode, detect, verification queue, model inference, or the
English-rule scanner — instead of inviting another guess.

A latency alone cannot rank stages against each other, because the stages do
not measure the same thing (7 Sep). `decode` times one frame at maybe 8/s;
`english_scan` times one scan cycle at 1 per 12s with a deliberate 120s budget,
so its p95 tops any raw ranking while costing almost no CPU; `detect_batch`
times a BATCH of N frames, and N varies. The window is 512 SAMPLES, not 512
seconds, so the same file holds a few seconds of decode beside an hour of
scans. Each series therefore reports the wall-clock span it covers and the
share of a thread it occupied over that span (`busy_fraction` = time spent /
time elapsed) — the one number that IS comparable across stages, and the one
`tools/perf_readout.py` ranks by.

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
        self._notes: dict[tuple[str, str], dict] = {}
        self._window = window
        self._started = time.time()

    def observe(self, stage: str, key: str, ms: float, *, units: int = 1) -> None:
        """Record one observation of `stage` on `key` (usually a camera id).

        `units` is how many units of work that one observation covered — one
        frame, normally, but `detect_batch` times a whole batch at once. Left
        at 1 the arithmetic is unchanged; passed the batch size, per_unit_ms
        stops reading as "detection costs 35ms a frame" when the 35ms in fact
        bought four of them.
        """
        try:
            with self._lock:
                series = self._series.get((stage, key))
                if series is None:
                    series = self._series[(stage, key)] = deque(maxlen=self._window)
                series.append((time.time(), float(ms), max(1, int(units))))
        except Exception:  # noqa: BLE001 - a metric must never hurt the pipeline
            log.debug("perf observation dropped", exc_info=True)

    def note(self, stage: str, key: str, **facts) -> None:
        """Attach non-latency context to a series (last writer wins).

        Some of what a support zip most needs is not a duration. The decoder
        already knows whether this machine sustains the frame rate asked of it
        — it computes `sustainable_fps` on every frame and then only LOGS it,
        so the one signal that separates "the box is too slow" from "the
        camera is pacing us" never reached the report the customer sends.
        """
        try:
            with self._lock:
                self._notes.setdefault((stage, key), {}).update(facts)
        except Exception:  # noqa: BLE001 - a metric must never hurt the pipeline
            log.debug("perf note dropped", exc_info=True)

    @staticmethod
    def _percentile(values: list[float], frac: float) -> float:
        idx = min(len(values) - 1, max(0, int(round(frac * (len(values) - 1)))))
        return values[idx]

    def snapshot(self) -> dict:
        """{stage: {key: {count, mean_ms, p50_ms, p95_ms, max_ms, last_ms, ...}}}.

        Beyond the percentiles, each series carries what makes it COMPARABLE to
        another stage: the wall-clock span its samples cover, the rate of work
        over that span, and `busy_fraction` — the share of one thread the stage
        occupied. Ranking by p95 alone names the wrong bottleneck (a 30s
        english_scan p95 is by design and costs ~0.4% of a core; a 35ms
        detect_batch at 8/s is 1.4 cores), which is the mistake this exists to
        make impossible.
        """
        with self._lock:
            items = {k: list(v) for k, v in self._series.items()}
            notes = {k: dict(v) for k, v in self._notes.items()}
        out: dict[str, dict] = {}
        for (stage, key), samples in sorted(items.items()):
            values = sorted(ms for _, ms, _u in samples)
            if not values:
                continue
            total_ms = sum(values)
            units = sum(u for _t, _ms, u in samples)
            first_at, last_at = samples[0][0], samples[-1][0]
            # The span the RETAINED samples cover. A full deque has already
            # dropped older ones, so this is the window's true reach — not
            # uptime, and never assumed to be 512 seconds.
            span_s = max(0.0, last_at - first_at)
            doc = {
                "count": len(values),
                "units": units,
                "mean_ms": round(total_ms / len(values), 1),
                "per_unit_ms": round(total_ms / units, 1),
                "p50_ms": round(self._percentile(values, 0.50), 1),
                "p95_ms": round(self._percentile(values, 0.95), 1),
                "max_ms": round(values[-1], 1),
                "last_ms": round(samples[-1][1], 1),
                "last_at": round(last_at, 3),
                "first_at": round(first_at, 3),
                "span_s": round(span_s, 3),
            }
            # One sample, or several inside the same clock tick, cannot yield a
            # rate. Say so with None rather than divide by ~0 and report a
            # stage as a thousand cores.
            if span_s > 0.05 and len(values) > 1:
                doc["rate_per_s"] = round(units / span_s, 3)
                doc["busy_fraction"] = round((total_ms / 1000.0) / span_s, 3)
            else:
                doc["rate_per_s"] = None
                doc["busy_fraction"] = None
            extra = notes.get((stage, key))
            if extra:
                doc.update(extra)
            out.setdefault(stage, {})[key] = doc
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
            # Without a core count, "detection occupies 1.4 threads" cannot be
            # read as a verdict: that is comfortable on 8 cores and terminal
            # on 2. The readout needs the denominator.
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
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
