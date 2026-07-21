"""Operator event inbox (plan.md Phase 9 / EPIC C).

A local web view over the alert sink's events.db: confirmed alerts newest-first,
each with its evidence frames + the gate's reason. So an operator can actually
review what fired — not just get a ping. Stdlib only (http.server + sqlite3), so
it runs on the edge box with no extra deps.

    python -m cvti.serving.inbox --db runs/serving/events.db --port 8080
    # then open http://localhost:8080
"""
from __future__ import annotations

import argparse
import html
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

_DB_PATH = "runs/serving/events.db"

_PRIORITY_COLOR = {"critical": "#e0655b", "high": "#e2a63c", "medium": "#5aa2e6", "low": "#7f8a9a"}

_CSS = """
*{box-sizing:border-box} body{margin:0;background:#0d1117;color:#e9edf3;
 font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5}
.wrap{max-width:1000px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:22px;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:#8c97a7;font-size:13px;font-family:ui-monospace,Menlo,monospace;margin-bottom:22px}
.empty{color:#8c97a7;background:#151b24;border:1px solid #232d3a;border-radius:12px;padding:40px;text-align:center}
.ev{background:#151b24;border:1px solid #232d3a;border-radius:12px;padding:16px 18px;margin-bottom:14px;
 border-left:4px solid var(--pc)}
.hd{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.pill{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;font-weight:700;letter-spacing:.06em;
 text-transform:uppercase;padding:3px 9px;border-radius:99px;color:var(--pc);
 background:color-mix(in srgb,var(--pc) 16%,transparent)}
.rule{font-weight:650;font-size:16px} .cam{color:#8c97a7;font-size:13px;font-family:ui-monospace,Menlo,monospace}
.when{margin-left:auto;color:#8c97a7;font-size:12px;font-family:ui-monospace,Menlo,monospace}
.reason{color:#c6cfdb;font-size:14px;margin:8px 0 12px}
.frames{display:flex;gap:8px;overflow-x:auto} .frames img{height:130px;border-radius:8px;border:1px solid #232d3a;flex:none}
.meta{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#6f7a8a;margin-top:10px}
"""


def _events(limit: int = 200) -> list[dict]:
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    except sqlite3.OperationalError:
        rows = []                         # no events table yet
    con.close()
    return [dict(r) for r in rows]


def _frames_for(event: dict) -> list[str]:
    d = Path(event.get("evidence_dir") or "")
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def _page() -> bytes:
    events = _events()
    cards = []
    for e in events:
        pc = _PRIORITY_COLOR.get(str(e.get("priority", "")).lower(), "#5aa2e6")
        imgs = "".join(
            f'<img src="/img/{e["id"]}/{html.escape(f)}" alt="evidence" loading="lazy">'
            for f in _frames_for(e))
        cards.append(f"""
        <div class="ev" style="--pc:{pc}">
          <div class="hd">
            <span class="pill">{html.escape(str(e.get("priority","")))}</span>
            <span class="rule">{html.escape(str(e.get("rule","")))}</span>
            <span class="cam">{html.escape(str(e.get("camera_id","")))}</span>
            <span class="when">{html.escape(str(e.get("iso","")))}</span>
          </div>
          <div class="reason">{html.escape(str(e.get("reason","") or ""))}</div>
          <div class="frames">{imgs or '<span class="cam">no frames saved</span>'}</div>
          <div class="meta">conf {float(e.get("confidence") or 0):.2f}
            · track {e.get("track_id")} · zone {e.get("zone")} · obj {e.get("object_label")}</div>
        </div>""")
    body = "".join(cards) or '<div class="empty">No confirmed alerts yet. Alerts appear here as the gate confirms them.</div>'
    doc = (f'<!doctype html><html><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<meta http-equiv="refresh" content="10"><title>CVTI — Event Inbox</title>'
           f'<style>{_CSS}</style></head><body><div class="wrap">'
           f'<h1>CVTI — Event Inbox</h1>'
           f'<div class="sub">{len(events)} confirmed alert(s) · auto-refreshes every 10s · {html.escape(_DB_PATH)}</div>'
           f'{body}</div></body></html>')
    return doc.encode()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet
        pass

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            data = _page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path.startswith("/img/"):
            self._serve_image(path)
            return
        self.send_error(404)

    def _serve_image(self, path: str) -> None:
        try:
            _, _, event_id, fname = path.split("/", 3)
        except ValueError:
            self.send_error(404)
            return
        con = sqlite3.connect(_DB_PATH)
        row = con.execute("SELECT evidence_dir FROM events WHERE id=?", (event_id,)).fetchone()
        con.close()
        if not row:
            self.send_error(404)
            return
        base = Path(row[0]).resolve()
        target = (base / Path(fname).name).resolve()          # strip any path traversal
        if base not in target.parents or not target.is_file():
            self.send_error(404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    global _DB_PATH
    p = argparse.ArgumentParser(description="CVTI operator event inbox (web view over events.db).")
    p.add_argument("--db", default="runs/serving/events.db")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    _DB_PATH = args.db
    print(f"[inbox] serving {args.db} at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    HTTPServer((args.host, args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
