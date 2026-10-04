"""Email alerts over SMTP — the channel a control room already has.

Spec (one channel inside the site's `notify` string):

    email:host=smtp.gmail.com&port=587&user=ops%40site.com&password=...&from=ops%40site.com&to=a%40x.com%3Bb%40y.com&security=starttls

The body is URL-encoded key=value pairs, so passwords with colons and
addresses with commas survive the colon- and comma-delimited notify spec
(a Telegram token broke the same way on 11 Sep). Recipients are separated
with ';'. `security` is starttls (587), ssl (465) or none (25, internal relays).

Delivery runs on its own thread, like Telegram: the gate's verdict callback
never waits on an SMTP handshake. Every send logs a line either way —
"[notify email] delivered" or "[notify email error]" — because a silent
notifier is indistinguishable from a working one (20 Sep).
"""
from __future__ import annotations

import html
import queue
import smtplib
import threading
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from cvti.logging_setup import get_logger

log = get_logger(__name__)

SECURITY_MODES = ("starttls", "ssl", "none")
DEFAULT_PORTS = {"starttls": 587, "ssl": 465, "none": 25}
MAX_FRAMES = 3
MAX_CLIP_MB = 8.0          # mail servers reject big attachments; the photos carry the alert

_PRIORITY_COLOURS = {
    "critical": "#b91c1c",
    "high": "#c2410c",
    "medium": "#a16207",
    "low": "#4b5563",
}


def parse_email_spec(body: str) -> dict | None:
    """The 'email:' spec body -> settings, or None when it cannot deliver.

    Incomplete settings return None so the caller can fall back LOUDLY; an
    email channel with no host or no recipient must never look configured.
    """
    fields = {k: v[-1].strip() for k, v in parse_qs(body or "", keep_blank_values=True).items()}
    host = fields.get("host", "")
    sender = fields.get("from", "") or fields.get("user", "")
    recipients = [r.strip() for r in (fields.get("to", "").replace(",", ";")).split(";") if r.strip()]
    security = (fields.get("security") or "starttls").lower()
    if security not in SECURITY_MODES:
        security = "starttls"
    if not host or not sender or not recipients:
        return None
    try:
        port = int(fields.get("port") or DEFAULT_PORTS[security])
    except ValueError:
        port = DEFAULT_PORTS[security]
    return {
        "host": host, "port": port, "username": fields.get("user", ""),
        "password": fields.get("password", ""), "sender": sender,
        "recipients": recipients, "security": security,
    }


def _humanise_rule(rule: str) -> str:
    text = (rule or "alert").replace("baseline_", "").replace("_candidate", "").replace("_", " ")
    return text.strip().upper() or "ALERT"


def render_html(event: dict, frame_cids: list[str] | None = None) -> str:
    """A simple, self-contained alert message: inline CSS only, no scripts,
    every value escaped. Reads on a phone and in Outlook."""
    priority = str(event.get("priority") or "high").lower()
    colour = _PRIORITY_COLOURS.get(priority, _PRIORITY_COLOURS["high"])
    title = html.escape(_humanise_rule(str(event.get("rule") or "")))
    camera = html.escape(str(event.get("camera_id") or "unknown camera"))
    when = html.escape(str(event.get("iso") or time.strftime("%Y-%m-%dT%H:%M:%S")).replace("T", " "))
    reason = html.escape(str(event.get("reason") or "")).replace("\n", "<br>")
    conf = event.get("confidence")
    conf_text = f"{float(conf) * 100:.0f}%" if isinstance(conf, (int, float)) and conf > 0 else "pending"
    zone = html.escape(str(event.get("zone") or "")) or "—"
    link = str(event.get("link") or "")
    rows = [
        ("Camera", camera),
        ("Time", when),
        ("Priority", html.escape(priority.upper())),
        ("Confidence", html.escape(conf_text)),
        ("Zone", zone),
    ]
    row_html = "".join(
        f'<tr><td style="padding:6px 0;color:#6b7280;font-size:13px;width:110px;">{k}</td>'
        f'<td style="padding:6px 0;color:#111827;font-size:14px;font-weight:600;">{v}</td></tr>'
        for k, v in rows
    )
    images = "".join(
        f'<img src="cid:{cid}" alt="Evidence frame {i + 1}" width="100%" '
        f'style="display:block;border-radius:8px;margin:0 0 10px;max-width:560px;">'
        for i, cid in enumerate(frame_cids or [])
    )
    if not images:
        images = ('<p style="margin:0;color:#6b7280;font-size:13px;">No evidence frame yet — '
                  'the verified alert with pictures follows.</p>')
    button = (
        f'<a href="{html.escape(link, quote=True)}" style="display:inline-block;background:{colour};'
        f'color:#ffffff;text-decoration:none;font-weight:600;padding:12px 20px;border-radius:8px;'
        f'font-size:14px;">Open this alert</a>'
    ) if link else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Argus alert: {title} on {camera}</title></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:24px 12px;">
<tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e5e7eb;">
  <tr><td style="background:{colour};color:#ffffff;padding:18px 24px;">
    <div style="font-size:12px;letter-spacing:0.08em;text-transform:uppercase;opacity:0.9;">Argus · {html.escape(priority.upper())} alert</div>
    <div style="font-size:22px;font-weight:700;margin-top:4px;">{title}</div>
    <div style="font-size:14px;margin-top:2px;opacity:0.95;">{camera} · {when}</div>
  </td></tr>
  <tr><td style="padding:20px 24px 8px;">
    <p style="margin:0 0 14px;color:#111827;font-size:15px;line-height:1.5;">{reason}</p>
    <table role="presentation" cellpadding="0" cellspacing="0" style="width:100%;border-top:1px solid #e5e7eb;border-bottom:1px solid #e5e7eb;">{row_html}</table>
  </td></tr>
  <tr><td style="padding:16px 24px 8px;">{images}</td></tr>
  <tr><td style="padding:8px 24px 24px;">{button}</td></tr>
  <tr><td style="background:#f9fafb;color:#6b7280;font-size:12px;padding:14px 24px;border-top:1px solid #e5e7eb;">
    Sent by Argus from this site's own computer. Reply to this message does not reach the camera operator.
  </td></tr>
</table>
</td></tr></table>
</body></html>"""


def render_text(event: dict) -> str:
    link = event.get("link")
    base = (f"{str(event.get('priority', 'high')).upper()} — {_humanise_rule(str(event.get('rule') or ''))} "
            f"on {event.get('camera_id')} at {event.get('iso', '')}\n{event.get('reason', '')}")
    return f"{base}\n\nOpen this alert: {link}" if link else base


class EmailNotifier:
    """Send each alert as one HTML email with the evidence frames inline."""

    QUEUE_MAX = 200

    def __init__(self, host: str, port: int, username: str, password: str, sender: str,
                 recipients: list[str], security: str = "starttls", timeout: float = 15.0,
                 *, background: bool = True) -> None:
        self.host, self.port = host, int(port)
        self.username, self.password = username, password
        self.sender, self.recipients = sender, list(recipients)
        self.security = security if security in SECURITY_MODES else "starttls"
        self.timeout = timeout
        self.sent = 0
        self.dropped = 0
        self.last_error = ""
        self._queue: "queue.Queue | None" = None
        self._worker: threading.Thread | None = None
        if background:
            self._queue = queue.Queue(maxsize=self.QUEUE_MAX)
            self._worker = threading.Thread(target=self._drain, name="email-notifier", daemon=True)
            self._worker.start()

    @classmethod
    def from_spec(cls, body: str, **kw: Any) -> "EmailNotifier | None":
        settings = parse_email_spec(body)
        return cls(**settings, **kw) if settings else None

    # -- queue -------------------------------------------------------------
    def notify(self, event: dict) -> None:
        if self._queue is None:
            self._deliver(event)
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1
            log.warning(f"[notify email] queue full ({self.QUEUE_MAX}); dropped "
                        f"{event.get('rule')} on {event.get('camera_id')}")

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self._deliver(item)

    def flush(self, timeout: float = 30.0) -> None:
        if self._queue is None:
            return
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)

    def close(self) -> None:
        self.flush()
        if self._queue is not None and self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=5.0)

    # -- one message ---------------------------------------------------------
    def _frames(self, event: dict) -> list[Path]:
        ev = event.get("evidence_dir")
        if not ev:
            return []
        d = Path(ev)
        if not d.exists():
            return []
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        subject = [p for p in imgs if p.name == "subject.jpg"]
        rest = [p for p in imgs if p.name != "subject.jpg"]
        return (subject + rest)[:MAX_FRAMES]

    def _clip(self, event: dict) -> Path | None:
        ev = event.get("evidence_dir")
        if not ev:
            return None
        clip = Path(ev) / "clip.mp4"
        try:
            if clip.exists() and clip.stat().st_size <= MAX_CLIP_MB * 1024 * 1024:
                return clip
        except OSError:
            return None
        return None

    def build_message(self, event: dict) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = (f"[Argus] {str(event.get('priority', 'high')).upper()} — "
                          f"{_humanise_rule(str(event.get('rule') or ''))} on {event.get('camera_id')}")
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="argus.local")
        msg["X-Argus-Priority"] = str(event.get("priority") or "")
        msg["X-Argus-Camera"] = str(event.get("camera_id") or "")
        frames = self._frames(event)
        cids = [make_msgid(domain="argus.local") for _ in frames]
        msg.set_content(render_text(event))
        msg.add_alternative(render_html(event, [c.strip("<>") for c in cids]), subtype="html")
        html_part = msg.get_payload()[-1]
        for path, cid in zip(frames, cids):
            data = path.read_bytes()
            subtype = "png" if path.suffix.lower() == ".png" else "jpeg"
            html_part.add_related(data, maintype="image", subtype=subtype, cid=cid,
                                  filename=path.name)
        clip = self._clip(event)
        if clip is not None:
            msg.add_attachment(clip.read_bytes(), maintype="video", subtype="mp4", filename=clip.name)
        return msg

    def _connect(self) -> smtplib.SMTP:
        if self.security == "ssl":
            server: smtplib.SMTP = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout)
        else:
            server = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
            server.ehlo()
            if self.security == "starttls":
                server.starttls()
                server.ehlo()
        if self.username:
            server.login(self.username, self.password)
        return server

    def _deliver(self, event: dict) -> None:
        what = f"{event.get('rule')} on {event.get('camera_id')}"
        try:
            msg = self.build_message(event)
            with self._connect() as server:
                server.send_message(msg)
            self.sent += 1
            self.last_error = ""
            log.info(f"[notify email] delivered {what} to {len(self.recipients)} recipient(s) "
                     f"via {self.host}:{self.port}")
        except Exception as exc:  # noqa: BLE001 - a notify failure must not kill the gate
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            log.error(f"[notify email error] {what}: {self.last_error}", exc_info=True)
