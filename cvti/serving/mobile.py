"""The mobile response view (EP-06-T3): act on an alert from where you stand.

Today the notification is mobile but the response is not — every action means
returning to a desk, which is backwards for a job whose defining feature is
walking around. From the guard's seat: "I get told about things I can't do
anything about from where I'm standing." A guard who feels that stops trusting
the alerts, and a monitoring product the monitor ignores has zero value.

This is deliberately not an app. It is one HTML page served by the engine over
the site's own network — no cloud, no store, no install. Telegram alerts
deep-link straight to the specific alert.

Security posture, because this is the first Argus surface that leaves
localhost:

- **Every route except /login requires a session**, backed by the same account
  store and lockout as the desktop app. There is no unauthenticated path to a
  frame, an alert, or an action — the plan calls this the one sequencing
  mistake that would be genuinely damaging, and it is tested explicitly.
- The session cookie is HttpOnly + SameSite=Lax; actions are POSTs carrying a
  CSRF token derived from (but not equal to) the session, so a hostile page on
  another origin can neither read nor forge it.
- Actions go through the same state machine (`cvti.triage`) and audit trail as
  the desktop — the phone is another door into the same room, not a side room.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import sqlite3
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cvti import triage
from cvti.logging_setup import get_logger
from cvti.security.accounts import AccountStore, AuthError
from cvti.security.audit import AuditLog
from cvti.security.permissions import REVIEW_ALERTS, VIEW_ALERTS, allows

log = get_logger(__name__)

DEFAULT_PORT = 8710
SESSION_COOKIE = "argus_session"

_SEV = {"critical": "#ef4444", "high": "#f59e0b", "medium": "#fad881", "low": "#8d94a6"}

_CSS = """
*{box-sizing:border-box;margin:0}
body{background:#0a1326;color:#edf1f8;font:15px/1.45 system-ui,-apple-system,sans-serif;
  padding:0 0 40px;max-width:520px;margin:0 auto}
a{color:#36a0da;text-decoration:none}
.top{display:flex;align-items:center;gap:9px;padding:12px 16px;border-bottom:1px solid #1f2c48;
  position:sticky;top:0;background:#0f1c36;z-index:5}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none}
.pad{padding:16px}
.pill{display:inline-block;padding:4px 11px;border-radius:20px;font-size:11px;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase}
.pill.unv{background:transparent;border:1px dashed #8d94a6;color:#8d94a6}
h1{font-size:22px;font-weight:680;margin:9px 0 2px}
.mut{color:#8d94a6}.sub{color:#5f6b84;font-size:12px}
.mono{font-family:ui-monospace,Menlo,monospace}
.frames{display:flex;gap:8px;overflow-x:auto;scroll-snap-type:x mandatory;
  margin:13px -16px 0;padding:0 16px}
.frames img{width:88%;flex:none;border-radius:11px;border:1px solid #1f2c48;
  scroll-snap-align:center}
.panel{background:#0f1c36;border:1px solid #1f2c48;border-radius:11px;padding:13px 15px;
  margin-top:13px}
.plabel{font-size:10px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
  color:#36a0da;margin-bottom:6px}
button,.btnlink{display:block;width:100%;padding:15px;margin-top:10px;border-radius:11px;
  border:1px solid #2a3a5c;background:#13223f;color:#edf1f8;font-size:15px;font-weight:650;
  text-align:center;cursor:pointer;min-height:48px}
button.pri{background:#36a0da;border-color:#36a0da;color:#04121f}
.btnrow{display:flex;gap:9px}.btnrow form{flex:1}
input[type=text],input[type=password]{width:100%;padding:13px;margin-top:8px;border-radius:10px;
  border:1px solid #2a3a5c;background:#0a1326;color:#edf1f8;font-size:15px;min-height:48px}
.err{color:#ef4444;font-size:13px;margin-top:9px;min-height:16px}
.own{margin-top:12px;font-size:13.5px}
.row{padding:11px 0;border-bottom:1px solid #1f2c48;font-size:14px}
"""


def _page(title: str, body: str, *, netname: str = "on site network") -> bytes:
    return (f"<!doctype html><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{_CSS}</style>"
            f"<div class='top'><span class='dot' style='background:#22c55e'></span>"
            f"<b>Argus</b><span style='flex:1'></span>"
            f"<span class='sub mono'>{html.escape(netname)}</span></div>"
            f"{body}").encode("utf-8")


def _csrf_for(token: str) -> str:
    # Derived from the session but not the session: forms can carry it while the
    # HttpOnly cookie stays unreadable to scripts and other origins.
    return hashlib.sha256(("csrf:" + token).encode()).hexdigest()[:24]


def lan_ip() -> str:
    """Best-effort LAN address for the deep-link. Falls back to hostname."""
    import socket
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("10.255.255.255", 1))
        ip = probe.getsockname()[0]
        probe.close()
        return ip
    except OSError:
        return socket.gethostbyname(socket.gethostname())


class MobileServer:
    def __init__(self, output_dir: str | Path, *, port: int = DEFAULT_PORT,
                 host: str = "0.0.0.0", frame_base: Path | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.host, self.port = host, port
        self.frame_base = frame_base
        self.accounts = AccountStore(self.output_dir / "auth.db")
        self.audit = AuditLog(self.output_dir / "audit.db")
        self._server = None
        self._thread = None

    # --- data ----------------------------------------------------------------
    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.output_dir / "events.db")
        con.row_factory = sqlite3.Row
        triage.ensure_columns(con)
        return con

    def _event(self, event_id: int) -> dict | None:
        con = self._con()
        try:
            row = con.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
            return dict(row) if row else None
        finally:
            con.close()

    def _frame_paths(self, event: dict) -> list:
        ev_dir = event.get("evidence_dir")
        if not ev_dir:
            return []
        path = Path(ev_dir)
        if not path.is_absolute() and self.frame_base:
            path = self.frame_base / path
        if not path.exists():
            return []
        subject = [path / "subject.jpg"] if (path / "subject.jpg").exists() else []
        return subject + sorted(path.glob("frame_*.jpg"))[:6]

    # --- views ---------------------------------------------------------------
    def _login_page(self, error: str = "", next_url: str = "/") -> bytes:
        return _page("Argus — sign in", f"""
<div class='pad'><h1>Sign in</h1>
<div class='mut' style='font-size:13px'>Same account as the console.</div>
<form method='post' action='/login'>
<input type='hidden' name='next' value='{html.escape(next_url)}'>
<input type='text' name='username' placeholder='name' autocomplete='username'>
<input type='password' name='password' placeholder='password' autocomplete='current-password'>
<div class='err'>{html.escape(error)}</div>
<button class='pri' type='submit'>Sign in</button>
</form></div>""")

    def _alert_page(self, event: dict, user, csrf: str, context: dict) -> bytes:
        eid = event["id"]
        sev = _SEV.get(event.get("priority") or "low", "#8d94a6")
        state = triage.state_of(event)
        unverified = bool(event.get("unverified"))
        pill = (f"<span class='pill unv'>Unverified</span>" if unverified else
                f"<span class='pill' style='background:{sev};color:#04121f'>"
                f"{html.escape((event.get('priority') or '').upper())}</span>")
        frames = "".join(
            f"<img src='/frame/{eid}/{i}.jpg' alt='evidence frame {i + 1}'>"
            for i in range(len(self._frame_paths(event)))) or \
            "<div class='panel mut'>No evidence frames on disk.</div>"

        if state == triage.RESOLVED:
            actions = (f"<div class='panel'><div class='plabel'>Concluded</div>"
                       f"{html.escape(event.get('outcome') or '')} by "
                       f"{html.escape(event.get('owner') or '?')}"
                       + (f"<div class='mut' style='margin-top:6px'>“"
                          f"{html.escape(event.get('note') or '')}”</div>"
                          if event.get("note") else "") + "</div>")
        else:
            owner = event.get("owner")
            claim = "" if owner else (
                f"<form method='post' action='/ack/{eid}'>"
                f"<input type='hidden' name='csrf' value='{csrf}'>"
                f"<button class='pri' type='submit'>I’m on it</button></form>"
                f"<div class='sub' style='margin-top:6px'>Claims this alert. Everyone "
                f"else sees your name against it.</div>")
            own = (f"<div class='own'><b>{html.escape(owner)}</b> is on this</div>"
                   if owner else "")
            actions = f"""{own}{claim}
<form method='post' action='/resolve/{eid}'>
<input type='hidden' name='csrf' value='{csrf}'>
<input type='text' name='note' placeholder='note for the handover (optional)'>
<div class='btnrow' style='margin-top:2px'>
<button type='submit' name='outcome' value='real'
  style='border-color:#ef4444;color:#ef4444;background:transparent'>Real</button>
<button type='submit' name='outcome' value='false_alarm'>False alarm</button>
</div></form>"""

        reason_label = ("What the detector claimed — NOT verified"
                        if unverified else "Why TrueSight confirmed this")
        unv_note = ("<div class='panel' style='border-style:dashed'><b>This has not "
                    "been checked by anything.</b> The verifier could not decide — "
                    "review it yourself.</div>" if unverified else "")
        queue = (f"<div class='sub' style='text-align:center;margin-top:16px'>"
                 f"{context.get('waiting', 0)} more waiting"
                 + "".join(f" · {html.escape(h.get('owner') or '?')} has "
                           f"{html.escape(h.get('rule') or '')}"
                           for h in (context.get("held") or [])[:1]) + "</div>")
        return _page(f"Argus — alert {eid}", f"""
<div class='pad'>
<div style='display:flex;gap:9px;align-items:center'>{pill}
<span class='mut mono' style='font-size:12px'>{html.escape(event.get('iso') or '')}
 · {html.escape(event.get('camera_id') or '')}</span></div>
<h1>{html.escape(event.get('rule') or '')}</h1>
<div class='frames'>{frames}</div>
{unv_note}
<div class='panel'><div class='plabel'>{reason_label}</div>
{html.escape(event.get('reason') or '')}</div>
{actions}
{queue}
<div style='text-align:center;margin-top:14px'><a href='/'>← all alerts</a>
 · <a href='/logout'>sign out</a></div>
</div>""")

    def _now_page(self, user, csrf: str) -> bytes:
        con = self._con()
        try:
            out = triage.needs_attention(con, min_priority="low")
        finally:
            con.close()
        if out["now"]:
            return self._alert_page(out["now"], user, csrf,
                                    {"waiting": max(0, out["waiting"] - 1),
                                     "held": out["held"]})
        held = "".join(
            f"<div class='row'>{html.escape(h.get('owner') or '?')} is on "
            f"<b class='mono'>{html.escape(h.get('rule') or '')}</b> · "
            f"{html.escape(h.get('camera_id') or '')}</div>"
            for h in out["held"]) or ""
        return _page("Argus", f"""
<div class='pad' style='text-align:center;padding-top:60px'>
<div style='font-size:42px'>✓</div>
<h1 style='font-size:19px'>Nothing needs you right now</h1>
<div class='mut' style='font-size:13.5px'>New alerts appear here the moment the
system confirms one.</div>
<div style='text-align:left;margin-top:24px'>{held}</div>
<div style='margin-top:26px'><a href='/logout'>sign out</a></div>
</div>""")

    # --- serving ---------------------------------------------------------------
    def start(self) -> "MobileServer":
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                return

            # -- plumbing --
            def _send(self, code, body, ctype="text/html; charset=utf-8", headers=None):
                data = body if isinstance(body, bytes) else body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _redirect(self, to, headers=None):
                self.send_response(303)
                self.send_header("Location", to)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _session(self):
                cookies = self.headers.get("Cookie", "")
                for part in cookies.split(";"):
                    name, _, value = part.strip().partition("=")
                    if name == SESSION_COOKIE:
                        return value
                return ""

            def _user(self):
                return server.accounts.session_user(self._session())

            def _form(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode("utf-8", "replace")
                return {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}

            def _csrf_ok(self, form) -> bool:
                return hmac.compare_digest(form.get("csrf", ""),
                                           _csrf_for(self._session()))

            # -- routes --
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                if path == "/login":
                    self._send(200, server._login_page())
                    return
                user = self._user()
                if user is None:
                    # Every other route. No session, no content — not even 404
                    # detail, which would confirm which alert ids exist.
                    self._redirect(f"/login?next={urllib.parse.quote(self.path)}")
                    return
                if not allows(user.role, VIEW_ALERTS):
                    self._send(403, _page("Argus", "<div class='pad'>This account "
                                          "cannot view alerts.</div>"))
                    return
                csrf = _csrf_for(self._session())
                if path == "/" or path == "":
                    self._send(200, server._now_page(user, csrf))
                elif path.startswith("/alert/"):
                    event = server._event(int(path.split("/")[2]))
                    if event is None:
                        self._send(404, _page("Argus", "<div class='pad'>No such "
                                              "alert.</div>"))
                    else:
                        con = server._con()
                        try:
                            ctx = triage.needs_attention(con, min_priority="low")
                        finally:
                            con.close()
                        self._send(200, server._alert_page(event, user, csrf,
                                                           {"waiting": ctx["waiting"],
                                                            "held": ctx["held"]}))
                elif path.startswith("/frame/"):
                    _, _, eid, name = path.split("/", 3)
                    event = server._event(int(eid))
                    index = int(name.split(".")[0])
                    paths = server._frame_paths(event or {})
                    if event and 0 <= index < len(paths):
                        self._send(200, paths[index].read_bytes(), "image/jpeg")
                    else:
                        self._send(404, _page("Argus", "<div class='pad'>No such "
                                              "frame.</div>"))
                elif path == "/logout":
                    server.accounts.close_session(self._session())
                    self._redirect("/login", {"Set-Cookie":
                                              f"{SESSION_COOKIE}=; Max-Age=0; Path=/"})
                else:
                    self._send(404, _page("Argus", "<div class='pad'>Not found."
                                          "</div>"))

            def do_POST(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                form = self._form()
                if path == "/login":
                    try:
                        user = server.accounts.authenticate(
                            form.get("username", ""), form.get("password", ""))
                    except AuthError as exc:
                        server.audit.record(form.get("username") or "<unknown>",
                                            "login", detail={"outcome": "refused",
                                                             "via": "mobile"})
                        self._send(200, server._login_page(str(exc),
                                                           form.get("next", "/")))
                        return
                    token = server.accounts.open_session(user.username)
                    server.audit.record(user.username, "login",
                                        detail={"outcome": "success", "via": "mobile"})
                    target = form.get("next") or "/"
                    if not target.startswith("/"):
                        target = "/"          # no open redirects off this host
                    self._redirect(target, {"Set-Cookie":
                                            f"{SESSION_COOKIE}={token}; HttpOnly; "
                                            f"SameSite=Lax; Path=/"})
                    return
                user = self._user()
                if user is None:
                    self._redirect("/login")
                    return
                if not allows(user.role, REVIEW_ALERTS) or not self._csrf_ok(form):
                    self._send(403, _page("Argus", "<div class='pad'>Refused.</div>"))
                    return
                try:
                    if path.startswith("/ack/"):
                        eid = int(path.split("/")[2])
                        con = server._con()
                        try:
                            triage.acknowledge(con, eid, user.username)
                        finally:
                            con.close()
                        server.audit.record(user.username, "alert_resolution",
                                            f"event:{eid}",
                                            {"transition": "acknowledged",
                                             "via": "mobile"})
                        self._redirect(f"/alert/{eid}")
                    elif path.startswith("/resolve/"):
                        eid = int(path.split("/")[2])
                        con = server._con()
                        try:
                            triage.resolve(con, eid, user.username,
                                           form.get("outcome", ""),
                                           form.get("note", "").strip())
                        finally:
                            con.close()
                        server.audit.record(user.username, "alert_resolution",
                                            f"event:{eid}",
                                            {"transition": "resolved",
                                             "outcome": form.get("outcome"),
                                             "via": "mobile"})
                        self._redirect("/")
                    else:
                        self._send(404, _page("Argus", "<div class='pad'>Not found."
                                              "</div>"))
                except triage.TriageError as exc:
                    # e.g. "sam is already on alert 12" — say it, plainly.
                    self._send(200, _page("Argus", f"<div class='pad'><h1 style="
                                          f"'font-size:18px'>Couldn’t do that"
                                          f"</h1><div class='mut'>{html.escape(str(exc))}"
                                          f"</div><a class='btnlink' href='{html.escape(self.path.replace('/ack/', '/alert/').replace('/resolve/', '/alert/'))}'"
                                          f">Back to the alert</a></div>"))

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="mobile", daemon=True)
        self._thread.start()
        log.info("mobile response view on http://%s:%d/ (authenticated; every "
                 "route requires sign-in)", lan_ip(), self.port)
        return self

    def base_url(self) -> str:
        return f"http://{lan_ip()}:{self.port}"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
