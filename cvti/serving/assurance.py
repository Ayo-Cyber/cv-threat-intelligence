"""Daily proof of life (EP-04-T4).

Silence is ambiguous — a quiet night and a dead engine produce the same user
experience. Traditional alarm panels solved this decades ago with a blinking
light. This module is the blinking light's two halves:

**The daily all-normal message.** The system says "I am fine" once a day, on by
default, so silence stops being the success signal. A user who stops receiving
it knows something — which is the entire point.

**The daily self-test.** An assumption is not a control: this *exercises* the
chain — take a real frame from a live camera, put a synthetic candidate through
the real verification gate, then deliver a real notification — and raises an
alert if any hop fails. It answers "would an alert actually reach a phone right
now?" with evidence instead of hope.

The self-test event is deliberately kept out of events.db: it is not an
incident, and counting it would pollute the Value screen's numbers.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from cvti.health import component
from cvti.logging_setup import get_logger

log = get_logger(__name__)

SELF_TEST_RULE = "self_test"
# Sent mid-morning: late enough that a night shift isn't woken by routine,
# early enough that a dead box is noticed the same working day.
DEFAULT_SEND_HOUR = 9


def _self_test_event(ok: bool, detail: str) -> dict:
    now = time.time()
    return {
        "ts": now, "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "camera_id": "self_test", "rule": SELF_TEST_RULE,
        "priority": "low" if ok else "high",
        "confidence": 1.0, "zone": None, "track_id": None, "object_label": None,
        "reason": detail, "evidence_dir": None,
    }


class Assurance:
    """Runs the daily message and the daily self-test on one thread.

    Everything is injected — frame source, gate factory, notifier, status
    provider, clock — so the schedule and both outcomes are testable without a
    camera, a model, or a real day passing.
    """

    def __init__(self, *,
                 latest_frame: Callable[[], Any],          # a real frame, or None
                 gate_factory: Callable[[], Any],
                 notifier: Any,                            # .notify(event)
                 status_provider: Callable[[], dict],      # the /health doc
                 daily_normal: bool = True,                # opt-out, per site
                 send_hour: int = DEFAULT_SEND_HOUR,
                 clock: Callable[[], float] = time.time) -> None:
        self.latest_frame = latest_frame
        self.gate_factory = gate_factory
        self.notifier = notifier
        self.status_provider = status_provider
        self.daily_normal = daily_normal
        self.send_hour = send_hour
        self.clock = clock
        self._health = component("self_test")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sent_day = ""                 # YYYY-MM-DD of the last daily message
        self._last_test_day = ""
        self.last_result: dict = {}              # surfaced in /health

    # --- the self-test -----------------------------------------------------
    def run_self_test(self) -> dict:
        """Exercise camera -> gate -> notification. Returns what happened."""
        started = self.clock()
        result = {"ok": False, "at": started, "steps": {}}

        # 1. A real frame from a real camera. No frame is itself a finding.
        frame = None
        try:
            frame = self.latest_frame()
        except Exception as exc:  # noqa: BLE001 - a probe failure is a result, not a crash
            log.warning("self-test: frame probe failed", exc_info=True)
            result["steps"]["frame"] = f"error: {str(exc)[:120]}"
        if frame is None:
            result["steps"].setdefault("frame", "no frame available from any camera")
            return self._finish(result, "self-test failed: no camera frame available")
        result["steps"]["frame"] = "ok"

        # 2. The real gate, with a synthetic candidate. ANY verdict proves the
        # chain; only "no verdict" (errored) is a failure — this is a plumbing
        # test, not an accuracy test.
        try:
            from cvti.contracts import CandidateAlert
            candidate = CandidateAlert(
                rule_name=SELF_TEST_RULE, priority="low", detector=SELF_TEST_RULE,
                title="SELF TEST — scheduled end-to-end check", person_id=None,
                object_label=None, timestamp=started)
            gate = self.gate_factory()
            verdict = gate.verify([frame], candidate,
                                  {"environment_type": "self-test",
                                   "scene_description": "Scheduled system self-test."})
            if getattr(verdict, "errored", False):
                result["steps"]["gate"] = f"no verdict: {verdict.error[:120]}"
                return self._finish(result, f"self-test failed: gate gave no verdict ({verdict.error[:80]})")
            result["steps"]["gate"] = f"verdict in chain (confirmed={verdict.confirmed})"
        except Exception as exc:  # noqa: BLE001
            log.warning("self-test: gate step failed", exc_info=True)
            result["steps"]["gate"] = f"error: {str(exc)[:120]}"
            return self._finish(result, f"self-test failed at the gate: {str(exc)[:80]}")

        # 3. A real notification, so delivery is proven, not presumed.
        try:
            self.notifier.notify(_self_test_event(
                True, "✅ Self-test passed — camera, verification and notification "
                      "all working end to end."))
            result["steps"]["notify"] = "ok"
        except Exception as exc:  # noqa: BLE001
            log.warning("self-test: notification step failed", exc_info=True)
            result["steps"]["notify"] = f"error: {str(exc)[:120]}"
            return self._finish(result, f"self-test failed at notification: {str(exc)[:80]}")

        result["ok"] = True
        result["seconds"] = round(self.clock() - started, 1)
        self._health.ok()
        self.last_result = result
        log.info("self-test passed in %.1fs", result["seconds"])
        return result

    def _finish(self, result: dict, headline: str) -> dict:
        """A failed self-test is itself an alert — that is the whole point."""
        result["seconds"] = round(self.clock() - result["at"], 1)
        self.last_result = result
        self._health.failed(RuntimeError(headline), log, "running the daily self-test")
        try:
            self.notifier.notify(_self_test_event(False, f"⚠️ {headline}"))
        except Exception:  # noqa: BLE001 - if notify is the broken hop, it cannot carry the news
            log.error("self-test failure could not be notified — the notification "
                      "path itself is down", exc_info=True)
        return result

    # --- the daily message ---------------------------------------------------
    def send_daily_normal(self) -> bool:
        if not self.daily_normal:
            return False
        status = {}
        try:
            status = self.status_provider() or {}
        except Exception:  # noqa: BLE001
            log.warning("daily message: status unavailable", exc_info=True)
        state = status.get("status", "unknown")
        cams = status.get("cameras") or []
        live = sum(1 for c in cams if c.get("state") == "connected")
        if state == "ok":
            text = (f"✅ All systems normal — {live} of {len(cams)} camera(s) live, "
                    f"verification healthy, disk "
                    f"{status.get('disk', {}).get('used_pct', '?')}% used.")
        else:
            reasons = "; ".join(status.get("reasons", [])[:3]) or "see the System panel"
            text = f"⚠️ Argus is {state}: {reasons}"
        try:
            event = _self_test_event(state == "ok", text)
            event["rule"] = "daily_status"
            event["priority"] = "low" if state == "ok" else "high"
            self.notifier.notify(event)
            return True
        except Exception:  # noqa: BLE001
            log.error("daily status message failed to send", exc_info=True)
            return False

    # --- scheduling ----------------------------------------------------------
    def tick(self) -> None:
        """One scheduler pass. Runs both jobs at most once per local day, after
        send_hour. Idempotent, so the caller's interval doesn't matter."""
        now = self.clock()
        local = time.localtime(now)
        today = time.strftime("%Y-%m-%d", local)
        if local.tm_hour < self.send_hour:
            return
        if self._last_test_day != today:
            self._last_test_day = today
            self.run_self_test()
        if self._last_sent_day != today:
            self._last_sent_day = today
            self.send_daily_normal()

    def start(self, interval: float = 300.0) -> "Assurance":
        def loop() -> None:
            while not self._stop.wait(interval):
                try:
                    self.tick()
                except Exception:  # noqa: BLE001 - assurance must never stop the engine
                    log.error("assurance tick failed", exc_info=True)
        self._thread = threading.Thread(target=loop, name="assurance", daemon=True)
        self._thread.start()
        log.info("assurance: daily self-test and status message scheduled after "
                 "%02d:00 local (daily message %s)", self.send_hour,
                 "on" if self.daily_normal else "opted out")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
