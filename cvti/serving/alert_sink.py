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
from cvti.logging_setup import get_logger

log = get_logger(__name__)


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
        log.info(f"[NOTIFY] {event['priority'].upper()} {event['rule']} on {event['camera_id']} "
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
            log.error(f"[notify webhook error] {str(exc)[:120]}", exc_info=True)


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
        # Lead with the annotated subject shot when there is one — on a phone the
        # first picture is what gets looked at, and it should point at the person.
        subject = [p for p in imgs if p.name == "subject.jpg"]
        rest = [p for p in imgs if p.name != "subject.jpg"]
        return (subject + rest)[:cap]

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
            log.error(f"[notify telegram error] {str(exc)[:140]}", exc_info=True)

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
            log.error(f"[notify whatsapp error] {str(exc)[:140]}", exc_info=True)


class MultiNotifier:
    """Fan an alert out to several notifiers; one failing never blocks the rest."""
    def __init__(self, notifiers: list) -> None:
        self.notifiers = notifiers

    def notify(self, event: dict) -> None:
        for n in self.notifiers:
            try:
                n.notify(event)
            except Exception as exc:  # noqa: BLE001
                log.error(f"[alert-sink] {type(n).__name__} failed: {str(exc)[:100]}", exc_info=True)


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
            log.warning(f"[alert-sink] whatsapp unavailable ({exc}); using console", exc_info=True)
            return ConsoleNotifier()
    log.warning(f"[alert-sink] unknown notifier '{spec}', using console")
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
    latency_s REAL,       -- detection -> TrueSight-verified wall-clock seconds
    bbox TEXT,            -- "x1,y1,x2,y2" of the subject when it fired (may be NULL)
    unverified INTEGER DEFAULT 0,  -- 1 = the gate never reached a verdict (fail-visible)
    gate_error TEXT,      -- why, when unverified
    legal_hold INTEGER DEFAULT 0,  -- 1 = exempt from retention purge, set by an operator
    state TEXT DEFAULT 'new',      -- new -> acknowledged -> resolved (cvti/triage.py)
    owner TEXT,                    -- who claimed it; visible to every operator
    acknowledged_at REAL,
    resolved_at REAL,
    outcome TEXT,                  -- real | false_alarm | inconclusive
    note TEXT                      -- free text captured at resolution
);

-- What the gate threw away, per day. Only confirmed alerts become rows in
-- `events`, so without this the product's central claim — "raw detectors would
-- have shown you 201 alerts, we showed you 28" — has no persistent evidence
-- behind it once the engine restarts.
CREATE TABLE IF NOT EXISTS suppression_daily (
    day TEXT PRIMARY KEY,        -- YYYY-MM-DD, local time
    shown INTEGER DEFAULT 0,     -- verified, confirmed, put in front of a human
    rejected INTEGER DEFAULT 0,  -- verified and thrown away
    deduped INTEGER DEFAULT 0,   -- same event already queued; never verified
    errors INTEGER DEFAULT 0,    -- the gate failed; counted, never hidden
    updated_at REAL
);
"""


class AlertSink:
    """Persist + notify CONFIRMED alerts. `handle(alert, result)` is the
    GatePool on_verdict callback (also prints every verdict, confirmed or not)."""

    def __init__(self, output_dir: str | Path, *, notifier: Any = None,
                 save_evidence: bool = True, iso_now: str | None = None,
                 routing_path: str | Path | None = "configs/routing.json") -> None:
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
        try:
            self._db.execute("ALTER TABLE events ADD COLUMN bbox TEXT")
        except sqlite3.OperationalError:
            pass
        for _col, _type in (("unverified", "INTEGER DEFAULT 0"), ("gate_error", "TEXT"),
                            ("legal_hold", "INTEGER DEFAULT 0"), ("state", "TEXT"),
                            ("owner", "TEXT"), ("acknowledged_at", "REAL"),
                            ("resolved_at", "REAL"), ("outcome", "TEXT"), ("note", "TEXT")):
            try:
                self._db.execute(f"ALTER TABLE events ADD COLUMN {_col} {_type}")
            except sqlite3.OperationalError:
                pass   # already there
        self._db.commit()
        self.persisted = 0
        # Routing: send each alert to the channels its rule names (severity, camera,
        # time-of-day…), falling back to the site-wide notifier. Escalation re-sends
        # anything still unacknowledged after the rule's deadline.
        from cvti.serving.routing import EscalationTracker, RoutingPolicy
        self.routing = RoutingPolicy.load(routing_path) if routing_path else RoutingPolicy()
        self.escalations = EscalationTracker(is_acknowledged=self._is_acknowledged)
        self._notifier_cache: dict = {}
        self.routed = 0
        self.escalated = 0
        if self.routing.rules:
            log.info(f"[routing] {len(self.routing.rules)} rule(s) loaded "
                  f"(default: {self.routing.default})")
        # Feedback loop: a calibration file (written by the FeedbackManager from
        # operator labels) tells us which (camera, rule) pairs are chronically wrong.
        # Demoted pairs are still stored (so the operator keeps correcting them) but
        # don't page anyone. Hot-reloaded when the file changes.
        from cvti.feedback.calibration import Calibration
        self._calib_path = self.root / "calibration.json"
        self._calib_mtime = 0.0
        self.calibration = Calibration()
        self._reload_calibration()

    # --- suppression ledger ----------------------------------------------
    def record_suppression(self, *, shown: int = 0, rejected: int = 0,
                           deduped: int = 0, errors: int = 0, day: str | None = None) -> None:
        """Add a delta to today's suppression row.

        Deltas, not totals: the gate pool's counters reset with the process, and
        a day's figure has to survive a restart to be worth quoting.
        """
        if not (shown or rejected or deduped or errors):
            return
        day = day or time.strftime("%Y-%m-%d")
        with self._lock:
            self._db.execute(
                "INSERT INTO suppression_daily (day, shown, rejected, deduped, errors, updated_at) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(day) DO UPDATE SET "
                "shown=shown+excluded.shown, rejected=rejected+excluded.rejected, "
                "deduped=deduped+excluded.deduped, errors=errors+excluded.errors, "
                "updated_at=excluded.updated_at",
                (day, shown, rejected, deduped, errors, time.time()))
            self._db.commit()

    def _reload_calibration(self) -> None:
        from cvti.feedback.calibration import Calibration
        try:
            mtime = self._calib_path.stat().st_mtime
        except OSError:
            return
        if mtime != self._calib_mtime:
            self.calibration = Calibration.load(self._calib_path)
            self._calib_mtime = mtime
            demoted = self.calibration.demoted_keys()
            if demoted:
                log.info(f"[calibration] loaded — demoting (no page): {', '.join(demoted)}")

    def handle(self, alert: Any, result: Any) -> None:
        if result is None:
            return
        # Three outcomes, not two. UNVERIFIED means the gate never reached a
        # verdict — logging it as CONFIRMED would overstate what we know, and as
        # REJECTED would claim a judgement nobody made.
        if getattr(result, "errored", False):
            tag, level = "UNVERIFIED", log.warning
        elif result.confirmed:
            tag, level = "CONFIRMED", log.info
        else:
            tag, level = "REJECTED ", log.info
        level("[%s] %s :: %s (%s) — %s | conf=%.2f | %s", tag, alert.camera_id,
              alert.rule_name, alert.priority.upper(), alert.title,
              result.confidence, result.reason)
        if getattr(result, "errored", False) and result.error:
            log.warning("[gate unavailable] %s :: %s — %s",
                        alert.camera_id, alert.rule_name, result.error)
        if not result.confirmed:
            return
        try:
            self._persist(alert, result)
        except Exception as exc:  # noqa: BLE001 - persistence must not kill the gate
            log.error(f"[alert-sink error] {str(exc)[:140]}", exc_info=True)

    # --- routing ---------------------------------------------------------
    def _is_acknowledged(self, event_id: Any) -> bool:
        """Has an operator handled this alert yet? (any review label clears it)"""
        try:
            with self._lock:
                row = self._db.execute(
                    "SELECT review FROM events WHERE id=?", (event_id,)).fetchone()
            return bool(row and row[0] and row[0] != "new")
        except Exception as exc:  # noqa: BLE001
            log.warning("acknowledgement lookup failed; treating as unacknowledged", exc_info=True)
            return False

    def _notifier_for(self, spec: str) -> Any:
        """Build (and cache) a notifier per channel spec — building a Telegram or
        Twilio client per alert would be wasteful."""
        if spec not in self._notifier_cache:
            self._notifier_cache[spec] = build_notifier(spec)
        return self._notifier_cache[spec]

    def _dispatch(self, event: dict, event_id: Any = None) -> None:
        """Send an event to whichever channels its routing rule names."""
        spec, rule_name = self.routing.channels_for(event)
        target = self._notifier_for(spec) if self.routing.rules else self.notifier
        try:
            target.notify(event)
            self.routed += 1
        except Exception as exc:  # noqa: BLE001 - a channel failing must not kill the sink
            log.error(f"[routing] '{rule_name}' -> {spec} failed: {str(exc)[:110]}", exc_info=True)
        if event_id is not None:
            rule = self.routing.match(event)
            if rule:
                self.escalations.register(event_id, event, rule)

    def run_escalations(self) -> int:
        """Re-send alerts nobody acknowledged in time. Call periodically."""
        sent = 0
        for item in self.escalations.due():
            ev = dict(item["event"])
            ev["escalated"] = True
            ev["reason"] = f"ESCALATED (unacknowledged): {ev.get('reason', '')}"
            try:
                self._notifier_for(item["to"]).notify(ev)
                sent += 1
                self.escalated += 1
                log.info(f"[routing] escalated {ev.get('camera_id')}::{ev.get('rule')} "
                      f"-> {item['to']} (rule '{item['rule']}')")
            except Exception as exc:  # noqa: BLE001
                log.error(f"[routing] escalation to {item['to']} failed: {str(exc)[:110]}", exc_info=True)
        return sent

    @staticmethod
    def _annotate(frame: Any, bbox: tuple | None, label: str, priority: str = "high") -> Any:
        """Draw the subject's box on a copy of the frame, so evidence shows WHO."""
        if not bbox:
            return frame
        import cv2
        colours = {"critical": (60, 60, 220), "high": (60, 140, 240),
                   "medium": (60, 200, 240), "low": (160, 160, 160)}   # BGR
        col = colours.get(str(priority).lower(), colours["high"])
        out = frame.copy()
        h, w = out.shape[:2]
        x1, y1, x2, y2 = (max(0, min(int(v), lim)) for v, lim in
                          zip(bbox, (w, h, w, h)))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = y1 - 8 if y1 > th + 12 else min(h - 4, y2 + th + 8)
            cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), col, -1)
            cv2.putText(out, label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _write_evidence(self, ev_dir: Path, payload: dict) -> None:
        """Write the event's frames + a clip.mp4.

        The app plays a smooth IMAGE cine-loop (no video-codec dependency — QtWebEngine
        can't always decode the mp4), so we save the whole CONTINUOUS window as
        individual JPEGs (they're already JPEG bytes -> written straight to disk). The
        mp4 is still written for archival / Telegram."""
        import cv2
        frames = payload.get("frames") or []
        clip_frames = payload.get("clip_frames") or []
        if clip_frames:
            for i, jb in enumerate(clip_frames):
                (ev_dir / f"frame_{i:03d}.jpg").write_bytes(jb)
        else:
            for i, f in enumerate(frames):
                cv2.imwrite(str(ev_dir / f"frame_{i:03d}.jpg"), f)
        # A single annotated "who" shot: the frame closest to the moment it fired,
        # with the subject boxed. Deliberately NOT drawn on every clip frame — the
        # subject moves, so a fixed box across the window would point at the wrong
        # place and mislead the operator.
        bbox = payload.get("bbox")
        if bbox and frames:
            cand = payload.get("candidate")
            label = f"#{getattr(cand, 'person_id', '')} {getattr(cand, 'title', '')}".strip()
            shot = self._annotate(frames[-1], bbox, label[:40],
                                  getattr(cand, "priority", "high"))
            cv2.imwrite(str(ev_dir / "subject.jpg"), shot)

        clip_fps = float(payload.get("clip_fps") or 0.0)
        if len(clip_frames) >= 6:
            fps = clip_fps if 1.0 <= clip_fps <= 60.0 else 4.0
            self._write_video_clip(ev_dir / "clip.mp4", clip_frames, fps)
        elif frames:
            self._write_clip(ev_dir / "clip.mp4", frames)

    def _persist(self, alert: Any, result: Any) -> None:
        ts = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        payload = alert.payload or {}
        ev_dir = self.events_dir / f"{stamp}_{alert.camera_id}_{alert.rule_name}"
        ev_dir.mkdir(parents=True, exist_ok=True)

        frames = payload.get("frames") or []
        if self.save_evidence and frames:
            self._write_evidence(ev_dir, payload)

        # detection -> verified wall-clock latency (queue wait + TrueSight time).
        enq = payload.get("enqueued_at")
        latency_s = round(ts - enq, 2) if isinstance(enq, (int, float)) else None

        event = {
            "ts": ts, "iso": iso, "camera_id": alert.camera_id, "rule": alert.rule_name,
            "priority": alert.priority, "confidence": float(result.confidence),
            "reason": result.reason, "track_id": alert.track_id, "zone": alert.zone,
            "object_label": alert.object_label, "evidence_dir": str(ev_dir),
            "latency_s": latency_s,
            "bbox": ",".join(str(int(v)) for v in payload["bbox"]) if payload.get("bbox") else None,
            "unverified": 1 if getattr(result, "errored", False) else 0,
            "gate_error": getattr(result, "error", "") or None,
        }
        (ev_dir / "event.json").write_text(json.dumps(event, indent=2))

        with self._lock:
            cur = self._db.execute(
                "INSERT INTO events (ts,iso,camera_id,rule,priority,confidence,reason,"
                "track_id,zone,object_label,evidence_dir,latency_s,bbox,unverified,gate_error) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, iso, alert.camera_id, alert.rule_name, alert.priority,
                 float(result.confidence), result.reason, alert.track_id, alert.zone,
                 alert.object_label, str(ev_dir), latency_s, event["bbox"],
                 event["unverified"], event["gate_error"]))
            event_id = cur.lastrowid
            self._db.commit()
        self.persisted += 1
        event["id"] = event_id
        # Feedback loop: chronically-wrong (camera, rule) pairs are stored but not paged.
        self._reload_calibration()
        if self.calibration.demoted(alert.camera_id, alert.rule_name):
            log.warning(f"[calibrated] {alert.camera_id} :: {alert.rule_name} demoted by feedback "
                  f"— stored, NOT notified")
        else:
            # Routed to the channels this alert's rule names; registers escalation
            # if nobody acknowledges it in time.
            self._dispatch(event, event_id)

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
