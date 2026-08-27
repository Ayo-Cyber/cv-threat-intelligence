"""Memory guard — notice pressure early and shed load instead of thrashing.

An edge box runs whatever the customer bought, and the detection stack is heavy
(YOLO + pose + VideoMAE + a local VLM, plus a decode thread and frame buffers per
camera). When it runs out, the machine doesn't fail cleanly — it swaps, and
everything crawls until someone kills a process by hand.

So the engine watches its own footprint and degrades on purpose, worst-cost
first, announcing every step:

    warn      trim the replay buffers, drop target fps            (quality dips)
    critical  disable the heaviest model, then shed a camera      (coverage dips)

Losing a camera is bad; a swapping box that misses every camera is worse. The
guard only ever RELEASES load — it never grows it back automatically, because
flapping between states under pressure is its own failure mode.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from cvti.logging_setup import get_logger

log = get_logger(__name__)

OK, WARN, CRITICAL = "ok", "warn", "critical"


@dataclass
class MemorySample:
    rss_gb: float
    available_gb: float
    percent_used: float
    swap_gb: float = 0.0

    def level(self, warn_available_gb: float, critical_available_gb: float) -> str:
        if self.available_gb <= critical_available_gb:
            return CRITICAL
        if self.available_gb <= warn_available_gb:
            return WARN
        return OK

    def to_dict(self) -> dict:
        return {"rss_gb": round(self.rss_gb, 2), "available_gb": round(self.available_gb, 2),
                "percent_used": round(self.percent_used, 1), "swap_gb": round(self.swap_gb, 2)}


def sample_memory() -> MemorySample:
    """Current process RSS + system-wide availability."""
    try:
        import psutil
        proc = psutil.Process()
        vm = psutil.virtual_memory()
        try:
            swap = psutil.swap_memory().used / 1e9
        except Exception as exc:  # noqa: BLE001 - swap is unavailable in some containers
            log.debug("swap metrics unavailable on this platform", exc_info=True)
            swap = 0.0
        return MemorySample(proc.memory_info().rss / 1e9, vm.available / 1e9,
                            vm.percent, swap)
    except Exception as exc:  # noqa: BLE001 - never let measurement break the engine
        log.debug("resource module unavailable", exc_info=True)
        import resource
        import sys
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = raw / 1e9 if sys.platform == "darwin" else raw / 1e6   # bytes vs KB
        return MemorySample(rss, float("inf"), 0.0, 0.0)             # can't see the system


@dataclass
class MemoryGuard:
    """Samples memory and fires escalating, one-shot mitigations."""

    warn_available_gb: float = 2.0
    critical_available_gb: float = 1.0
    interval: float = 15.0
    # Each action is tried once, in order, as pressure escalates. () -> str|None,
    # returning a human-readable description of what it gave up.
    warn_actions: list = field(default_factory=list)
    critical_actions: list = field(default_factory=list)
    on_sample: Callable[[MemorySample, str], None] | None = None

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warn_idx = 0
        self._crit_idx = 0
        self.level = OK
        self.samples = 0
        self.mitigations: list[str] = []
        self.last: MemorySample | None = None

    # --- one pass (called by the loop; also directly in tests) ------------
    def check(self, sample: MemorySample | None = None) -> str:
        s = sample or sample_memory()
        self.last = s
        self.samples += 1
        level = s.level(self.warn_available_gb, self.critical_available_gb)
        if level != self.level:
            if level != OK:
                log.info(f"[memory] {level.upper()}: {s.available_gb:.1f}G available "
                      f"(engine using {s.rss_gb:.1f}G, swap {s.swap_gb:.1f}G)")
            else:
                log.info(f"[memory] recovered: {s.available_gb:.1f}G available")
            self.level = level
        if level == CRITICAL:
            self._apply(self.critical_actions, "critical")
        elif level == WARN:
            self._apply(self.warn_actions, "warn")
        if self.on_sample:
            try:
                self.on_sample(s, level)
            except Exception as exc:  # noqa: BLE001
                log.warning("on_sample callback failed", exc_info=True)
                pass
        return level

    def _apply(self, actions: list, tier: str) -> None:
        """Take ONE more step at this tier — never everything at once, so the
        lightest mitigation gets a chance to work before we shed a camera."""
        idx = self._crit_idx if tier == "critical" else self._warn_idx
        if idx >= len(actions):
            return
        action = actions[idx]
        if tier == "critical":
            self._crit_idx += 1
        else:
            self._warn_idx += 1
        try:
            what = action()
        except Exception as exc:  # noqa: BLE001 - a failed mitigation must not crash the engine
            log.error(f"[memory] mitigation failed: {str(exc)[:110]}", exc_info=True)
            return
        if what:
            self.mitigations.append(what)
            log.info(f"[memory] mitigation ({tier}): {what}")

    # --- background loop --------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.check()
            except Exception as exc:  # noqa: BLE001
                log.error(f"[memory] check failed: {str(exc)[:110]}", exc_info=True)

    def start(self) -> "MemoryGuard":
        s = sample_memory()
        self.last = s
        log.warning(f"[memory] guard on — {s.available_gb:.1f}G available, engine {s.rss_gb:.1f}G "
              f"(warn <{self.warn_available_gb}G, critical <{self.critical_available_gb}G)")
        self._thread = threading.Thread(target=self._loop, name="memory-guard", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def status(self) -> dict:
        return {"level": self.level, "samples": self.samples,
                "mitigations": list(self.mitigations),
                "memory": self.last.to_dict() if self.last else None}


def build_default_actions(pipe: Any, states: dict | None = None) -> tuple[list, list]:
    """Mitigations ordered cheapest-first: give up quality before coverage."""
    states = states or {}

    def trim_buffers() -> str | None:
        trimmed = 0
        for st in states.values():
            buf = getattr(st, "_clip_buffer", None)
            touched = False
            if buf is not None and buf.maxlen and buf.maxlen > 16:
                while len(buf) > 16:
                    buf.popleft()
                touched = True
            # The RAW frame buffer is the big one — ~62 MB/camera at 1080p
            # (10 undownscaled BGR frames) — and the guard never touched it.
            # Under pressure, halve it: evidence pre-roll gets shorter, the
            # box stops swapping. (RAM audit 24 Aug.)
            fbuf = getattr(st, "_frame_buffer", None)
            if fbuf is not None and len(fbuf) > 4:
                while len(fbuf) > 4:
                    fbuf.popleft()
                touched = True
            trimmed += 1 if touched else 0    # count CAMERAS, not buffers
        return f"trimmed frame/replay buffers on {trimmed} camera(s)" if trimmed else None

    def lower_fps() -> str | None:
        cur = getattr(pipe, "target_fps", 0)
        if cur and cur > 2:
            pipe.target_fps = max(2.0, cur / 2)
            for d in getattr(pipe, "_decoders", {}).values():
                d.target_fps = pipe.target_fps
                d._min_period = 1.0 / pipe.target_fps
            return f"target fps {cur:.0f} -> {pipe.target_fps:.0f}"
        return None

    def smaller_frames() -> str | None:
        cur = getattr(pipe, "imgsz", 0)
        if cur and cur > 320:
            pipe.imgsz = 320
            return f"detector input {cur} -> 320"
        return None

    def drop_video_action() -> str | None:
        n = 0
        for st in states.values():
            if getattr(st, "_video_runtime", None) is not None:
                st._video_runtime = None
                st.video_action = False
                n += 1
        return f"disabled the video-action model on {n} camera(s)" if n else None

    def shed_camera() -> str | None:
        decoders = getattr(pipe, "_decoders", {})
        if len(decoders) <= 1:
            return None                     # never go to zero cameras
        cam_id = sorted(decoders)[-1]
        try:
            decoders[cam_id].stop()
        except Exception as exc:  # noqa: BLE001
            log.warning("stopping a camera to free memory failed", exc_info=True)
            pass
        decoders.pop(cam_id, None)
        return f"stopped camera '{cam_id}' to free memory"

    return ([trim_buffers, lower_fps, smaller_frames],
            [drop_video_action, shed_camera])
