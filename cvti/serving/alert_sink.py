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
import uuid
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Notifiers — each exposes notify(event: dict) -> None
# ---------------------------------------------------------------------------

def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """Build a multipart/form-data body (stdlib only). files: {name: (filename, bytes)}."""
    boundary = "----cvti" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += f"{value}\r\n".encode()
    for name, (filename, data) in files.items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: image/jpeg\r\n\r\n').encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"

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
    """Send to a Telegram chat via the bot API (token + chat_id). Attaches the
    evidence frames as a photo album so the alert arrives WITH pictures."""

    def __init__(self, token: str, chat_id: str, timeout: float = 8.0) -> None:
        self.base = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id
        self.timeout = timeout

    def _caption(self, event: dict) -> str:
        return (f"⚠️ {event['priority'].upper()} — {event['rule']} on "
                f"{event['camera_id']} (conf {event['confidence']:.2f})\n{event['reason']}")

    def _frames(self, event: dict, cap: int = 3) -> list[Path]:
        d = Path(event.get("evidence_dir") or "")
        if not d.exists():
            return []
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        return imgs[:cap]

    def notify(self, event: dict) -> None:
        import urllib.parse
        import urllib.request
        frames = self._frames(event)
        try:
            if not frames:
                data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": self._caption(event)}).encode()
                urllib.request.urlopen(f"{self.base}/sendMessage", data=data, timeout=self.timeout)
                return
            self._send_photos(frames, self._caption(event))
        except Exception as exc:  # noqa: BLE001 - a notify failure must not kill the gate
            print(f"[notify telegram error] {str(exc)[:140]}")

    def _send_photos(self, frames: list[Path], caption: str) -> None:
        """One photo -> sendPhoto; several -> sendMediaGroup (album)."""
        import urllib.request
        if len(frames) == 1:
            fields = {"chat_id": self.chat_id, "caption": caption}
            files = {"photo": (frames[0].name, frames[0].read_bytes())}
            body, ctype = _multipart(fields, files)
            req = urllib.request.Request(f"{self.base}/sendPhoto", data=body, headers={"Content-Type": ctype})
            urllib.request.urlopen(req, timeout=self.timeout)
            return
        media, files = [], {}
        for i, fr in enumerate(frames):
            key = f"photo{i}"
            item = {"type": "photo", "media": f"attach://{key}"}
            if i == 0:
                item["caption"] = caption          # caption goes on the first item
            media.append(item)
            files[key] = (fr.name, fr.read_bytes())
        body, ctype = _multipart({"chat_id": self.chat_id, "media": json.dumps(media)}, files)
        req = urllib.request.Request(f"{self.base}/sendMediaGroup", data=body, headers={"Content-Type": ctype})
        urllib.request.urlopen(req, timeout=self.timeout)


class WhatsAppNotifier:
    """Send to WhatsApp via Twilio's API. Credentials come from env vars (never
    committed): TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM
    (defaults to the Twilio sandbox number), WHATSAPP_TO (e.g. +234...)."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str,
                 to_number: str, timeout: float = 6.0) -> None:
        import base64
        self.url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        self.auth = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
        self.from_ = from_number if from_number.startswith("whatsapp:") else f"whatsapp:{from_number}"
        self.to = to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}"
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "WhatsAppNotifier":
        import os
        sid = os.environ.get("TWILIO_ACCOUNT_SID")
        tok = os.environ.get("TWILIO_AUTH_TOKEN")
        frm = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")  # Twilio sandbox
        to = os.environ.get("WHATSAPP_TO")
        if not (sid and tok and to):
            raise RuntimeError("WhatsApp needs TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, WHATSAPP_TO env vars")
        return cls(sid, tok, frm, to)

    def notify(self, event: dict) -> None:
        import urllib.parse
        import urllib.request
        text = (f"⚠️ {event['priority'].upper()} — {event['rule']} on "
                f"{event['camera_id']} (conf {event['confidence']:.2f})\n{event['reason']}")
        data = urllib.parse.urlencode({"From": self.from_, "To": self.to, "Body": text}).encode()
        try:
            req = urllib.request.Request(self.url, data=data,
                                         headers={"Authorization": f"Basic {self.auth}"}, method="POST")
            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[notify whatsapp error] {str(exc)[:140]}")


class MultiNotifier:
    """Fan an alert out to several notifiers; one failing never blocks the rest."""
    def __init__(self, notifiers: list) -> None:
        self.notifiers = notifiers

    def notify(self, event: dict) -> None:
        for n in self.notifiers:
            try:
                n.notify(event)
            except Exception as exc:  # noqa: BLE001
                print(f"[alert-sink] {type(n).__name__} failed: {str(exc)[:100]}")


def _build_one(spec: str) -> Any:
    spec = spec.strip()
    if not spec or spec == "console":
        return ConsoleNotifier()
    if spec.startswith("webhook:"):
        return WebhookNotifier(spec[len("webhook:"):])
    if spec.startswith("telegram:"):
        _, token, chat_id = spec.split(":", 2)
        return TelegramNotifier(token, chat_id)
    if spec == "whatsapp":
        try:
            return WhatsAppNotifier.from_env()
        except Exception as exc:  # noqa: BLE001
            print(f"[alert-sink] whatsapp unavailable ({exc}); using console")
            return ConsoleNotifier()
    print(f"[alert-sink] unknown notifier '{spec}', using console")
    return ConsoleNotifier()


def build_notifier(spec: str) -> Any:
    """spec: one channel, or several comma-separated (e.g. 'console,whatsapp').
    Each channel: 'console' | 'webhook:<url>' | 'telegram:<token>:<chat_id>' |
    'whatsapp' (Twilio creds from env)."""
    parts = [s for s in (spec or "console").split(",") if s.strip()]
    if len(parts) <= 1:
        return _build_one(parts[0] if parts else "console")
    return MultiNotifier([_build_one(p) for p in parts])


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, iso TEXT, camera_id TEXT, rule TEXT, priority TEXT,
    confidence REAL, reason TEXT, track_id INTEGER, zone TEXT,
    object_label TEXT, evidence_dir TEXT,
    review TEXT,          -- NULL=new, 'ack', 'true', 'false' (operator label)
    reviewed_at TEXT,
    latency_s REAL        -- detection -> TrueSight-verified wall-clock seconds
);
CREATE TABLE IF NOT EXISTS suppressions (
    -- Every alert TrueSight REJECTED — the false alarms the operator never sees.
    -- This is the noise-suppression story, so we keep the count + a sample reason.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, iso TEXT, camera_id TEXT, rule TEXT, confidence REAL, reason TEXT
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
        self._db.executescript(_SCHEMA)
        # Migrate older DBs that predate the latency column.
        try:
            self._db.execute("ALTER TABLE events ADD COLUMN latency_s REAL")
        except sqlite3.OperationalError:
            pass       # already there
        self._db.commit()
        self.persisted = 0
        self.suppressed = 0

    def handle(self, alert: Any, result: Any) -> None:
        if result is None:
            return
        tag = "CONFIRMED" if result.confirmed else "REJECTED "
        print(f"[{tag}] {alert.camera_id} :: {alert.rule_name} ({alert.priority.upper()}) "
              f"— {alert.title} | conf={result.confidence:.2f} | {result.reason}")
        if not result.confirmed:
            try:
                self._record_suppression(alert, result)   # the noise-suppression story
            except Exception as exc:  # noqa: BLE001
                print(f"[alert-sink error] {str(exc)[:140]}")
            return
        try:
            self._persist(alert, result)
        except Exception as exc:  # noqa: BLE001 - persistence must not kill the gate
            print(f"[alert-sink error] {str(exc)[:140]}")

    def _record_suppression(self, alert: Any, result: Any) -> None:
        ts = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
        with self._lock:
            self._db.execute(
                "INSERT INTO suppressions (ts,iso,camera_id,rule,confidence,reason) "
                "VALUES (?,?,?,?,?,?)",
                (ts, iso, alert.camera_id, alert.rule_name,
                 float(getattr(result, "confidence", 0.0)), getattr(result, "reason", "")))
            self._db.commit()
        self.suppressed += 1

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
            # Prefer a REAL continuous video of the event window; fall back to the
            # held-stills slideshow when no continuous buffer was captured.
            clip_frames = payload.get("clip_frames") or []
            clip_fps = float(payload.get("clip_fps") or 0.0)
            if clip_frames and clip_fps > 0:
                self._write_video_clip(ev_dir / "clip.mp4", clip_frames, clip_fps)
            else:
                self._write_clip(ev_dir / "clip.mp4", frames)

        # detection -> verified wall-clock latency (queue wait + TrueSight time).
        enq = payload.get("enqueued_at")
        latency_s = round(ts - enq, 2) if isinstance(enq, (int, float)) else None

        event = {
            "ts": ts, "iso": iso, "camera_id": alert.camera_id, "rule": alert.rule_name,
            "priority": alert.priority, "confidence": float(result.confidence),
            "reason": result.reason, "track_id": alert.track_id, "zone": alert.zone,
            "object_label": alert.object_label, "evidence_dir": str(ev_dir),
            "latency_s": latency_s,
        }
        (ev_dir / "event.json").write_text(json.dumps(event, indent=2))

        with self._lock:
            self._db.execute(
                "INSERT INTO events (ts,iso,camera_id,rule,priority,confidence,reason,"
                "track_id,zone,object_label,evidence_dir,latency_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, iso, alert.camera_id, alert.rule_name, alert.priority,
                 float(result.confidence), result.reason, alert.track_id, alert.zone,
                 alert.object_label, str(ev_dir), latency_s))
            self._db.commit()
        self.persisted += 1
        self.notifier.notify(event)

    def _write_video_clip(self, path: Path, jpeg_frames: list, src_fps: float,
                          container_fps: int = 24) -> None:
        """Write a REAL video of the event from the continuous JPEG window.

        The pipeline captures frames at `src_fps` (e.g. ~4). We re-time them onto a
        player-friendly `container_fps` by repeating each source frame, so the clip
        plays at TRUE real-time speed (duration = n/src_fps) but decodes smoothly in
        any player — real footage of the event, not a slideshow of stills.
        """
        import cv2
        import numpy as np
        if not jpeg_frames:
            return
        imgs = []
        for jb in jpeg_frames:
            im = cv2.imdecode(np.frombuffer(jb, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                imgs.append(im)
        if not imgs:
            return
        h, w = imgs[0].shape[:2]
        repeat = max(1, int(round(container_fps / max(src_fps, 0.1))))
        for fourcc in ("avc1", "mp4v"):        # prefer H.264, fall back to MPEG-4
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), container_fps, (w, h))
            if vw.isOpened():
                for im in imgs:
                    out = im if im.shape[:2] == (h, w) else cv2.resize(im, (w, h))
                    for _ in range(repeat):
                        vw.write(out)
                vw.release()
                return

    def _write_clip(self, path: Path, frames: list, fps: int = 12,
                    hold_seconds: float = 0.5) -> None:
        """Encode the evidence frames into a short MP4 (archival + Telegram video).

        Evidence is only a handful of stills, so a naive low-fps clip flashes past
        too fast to read. Instead we hold each frame for ~`hold_seconds` at a normal
        container fps — a slower, smoother playback that any player handles cleanly.
        """
        import cv2
        if not frames:
            return
        h, w = frames[0].shape[:2]
        repeat = max(1, int(round(fps * hold_seconds)))   # frames to hold each still
        for fourcc in ("avc1", "mp4v"):        # prefer H.264, fall back to MPEG-4
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if vw.isOpened():
                for f in frames:
                    out = f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h))
                    for _ in range(repeat):
                        vw.write(out)
                vw.release()
                return

    def close(self) -> None:
        with self._lock:
            self._db.close()
