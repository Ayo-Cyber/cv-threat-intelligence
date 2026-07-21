"""Alert sink + notifier (plan.md Phase 9 / P0).

Confirmed alerts in the serving path used to only `print()` and vanish. This
persists every CONFIRMED alert to a SQLite event store WITH an evidence bundle
(the frames the gate saw + a metadata.json), and pushes a notification
(console / webhook / Telegram). Wire it into `GatePool(on_verdict=sink.handle)`.

Stdlib + cv2 only (sqlite3, urllib) so it runs on an offline edge box with no
extra pip deps.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Notifiers — each exposes notify(event: dict) -> None
# ---------------------------------------------------------------------------

class ConsoleNotifier:
    def notify(self, event: dict) -> None:
        print(f"[NOTIFY] {event['priority'].upper()} {event['rule']} on {event['camera_id']} "
              f"(conf {event['confidence']:.2f}) — evidence: {event['evidence_dir']}")


class WebhookNotifier:
    """POST the event as JSON to any webhook (Slack/Discord/custom endpoint)."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout

    def notify(self, event: dict) -> None:
        import urllib.request
        text = (f"⚠️ {event['priority'].upper()} — {event['rule']} on "
                f"{event['camera_id']} (conf {event['confidence']:.2f}): {event['reason']}")
        payload = {"text": text, "event": event}
        try:
            req = urllib.request.Request(
                self.url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 - a notify failure must not kill the gate
            print(f"[notify webhook error] {str(exc)[:120]}")


class TelegramNotifier:
    """Send to a Telegram chat via the bot API (token + chat_id)."""

    def __init__(self, token: str, chat_id: str, timeout: float = 5.0) -> None:
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id
        self.timeout = timeout

    def notify(self, event: dict) -> None:
        import urllib.parse
        import urllib.request
        text = (f"⚠️ {event['priority'].upper()} — {event['rule']} on "
                f"{event['camera_id']} (conf {event['confidence']:.2f})\n{event['reason']}")
        data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text}).encode()
        try:
            urllib.request.urlopen(self.url, data=data, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[notify telegram error] {str(exc)[:120]}")


def build_notifier(spec: str) -> Any:
    """spec: 'console' | 'webhook:<url>' | 'telegram:<token>:<chat_id>'."""
    if not spec or spec == "console":
        return ConsoleNotifier()
    if spec.startswith("webhook:"):
        return WebhookNotifier(spec[len("webhook:"):])
    if spec.startswith("telegram:"):
        _, token, chat_id = spec.split(":", 2)
        return TelegramNotifier(token, chat_id)
    print(f"[alert-sink] unknown notifier '{spec}', using console")
    return ConsoleNotifier()


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, iso TEXT, camera_id TEXT, rule TEXT, priority TEXT,
    confidence REAL, reason TEXT, track_id INTEGER, zone TEXT,
    object_label TEXT, evidence_dir TEXT
);
"""


class AlertSink:
    """Persist + notify CONFIRMED alerts. `handle(alert, result)` is the
    GatePool on_verdict callback (also prints every verdict, confirmed or not)."""

    def __init__(self, output_dir: str | Path, *, notifier: Any = None,
                 save_evidence: bool = True, iso_now: str | None = None) -> None:
        self.root = Path(output_dir)
        self.events_dir = self.root / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "events.db"
        self.notifier = notifier or ConsoleNotifier()
        self.save_evidence = save_evidence
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.execute(_SCHEMA)
        self._db.commit()
        self.persisted = 0

    def handle(self, alert: Any, result: Any) -> None:
        if result is None:
            return
        tag = "CONFIRMED" if result.confirmed else "REJECTED "
        print(f"[{tag}] {alert.camera_id} :: {alert.rule_name} ({alert.priority.upper()}) "
              f"— {alert.title} | conf={result.confidence:.2f} | {result.reason}")
        if not result.confirmed:
            return
        try:
            self._persist(alert, result)
        except Exception as exc:  # noqa: BLE001 - persistence must not kill the gate
            print(f"[alert-sink error] {str(exc)[:140]}")

    def _persist(self, alert: Any, result: Any) -> None:
        ts = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        payload = alert.payload or {}
        ev_dir = self.events_dir / f"{stamp}_{alert.camera_id}_{alert.rule_name}"
        ev_dir.mkdir(parents=True, exist_ok=True)

        frames = payload.get("frames") or []
        if self.save_evidence and frames:
            import cv2
            for i, f in enumerate(frames):
                cv2.imwrite(str(ev_dir / f"frame_{i:02d}.jpg"), f)

        event = {
            "ts": ts, "iso": iso, "camera_id": alert.camera_id, "rule": alert.rule_name,
            "priority": alert.priority, "confidence": float(result.confidence),
            "reason": result.reason, "track_id": alert.track_id, "zone": alert.zone,
            "object_label": alert.object_label, "evidence_dir": str(ev_dir),
        }
        (ev_dir / "event.json").write_text(json.dumps(event, indent=2))

        with self._lock:
            self._db.execute(
                "INSERT INTO events (ts,iso,camera_id,rule,priority,confidence,reason,"
                "track_id,zone,object_label,evidence_dir) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (ts, iso, alert.camera_id, alert.rule_name, alert.priority,
                 float(result.confidence), result.reason, alert.track_id, alert.zone,
                 alert.object_label, str(ev_dir)))
            self._db.commit()
        self.persisted += 1
        self.notifier.notify(event)

    def close(self) -> None:
        with self._lock:
            self._db.close()
