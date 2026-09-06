"""Read helpers over the engine's own outputs.

Everything the read-only API serves comes from files the engine already
writes — the events database, gate_health.json, and the site config. Reading
them here (rather than through the console's permission-gated methods) keeps
the API decoupled from the console's single-session model and adds zero load
to the detection path.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from cvti.logging_setup import get_logger
from cvti.utils import redact_credentials

log = get_logger(__name__)


# ---- health -----------------------------------------------------------------

def read_health(db_path: str) -> dict:
    """The engine's health doc, or a truthful 'unknown' when it hasn't run."""
    path = Path(db_path).parent / "gate_health.json"
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"status": "unknown", "reasons": ["engine has not reported yet"],
                "cameras": [], "gate": {}, "engine": {"phase": "stopped"}}
    return doc


def monitor_state(db_path: str) -> dict:
    """Derive running/starting/stopped from the health doc's freshness — the
    same heartbeat truth the console uses, without owning the subprocess."""
    doc = read_health(db_path)
    generated = float(doc.get("generated_at") or 0)
    age = time.time() - generated if generated else None
    phase = str((doc.get("engine") or {}).get("phase") or "")
    fresh = age is not None and age < 30
    starting = phase.startswith("starting") and (age is not None and age < 90)
    return {
        "running": bool(fresh or starting),
        "starting": bool(starting),
        "phase": phase or ("stopped" if not fresh else "monitoring"),
        "health_age_s": round(age, 1) if age is not None else None,
    }


# ---- cameras ----------------------------------------------------------------

def read_cameras(site_path: str, db_path: str) -> list[dict]:
    """Configured cameras (credential-redacted) merged with live link state
    from the health doc."""
    try:
        from cvti.serving.onboarding import list_cameras
        cams = list_cameras(site_path)
    except Exception:  # noqa: BLE001 - a missing/invalid site is empty, not a crash
        log.debug("could not read site cameras; returning empty", exc_info=True)
        cams = []
    health = read_health(db_path)
    link = {c.get("camera_id"): c for c in (health.get("cameras") or [])}
    out = []
    for c in cams:
        cid = str(c.get("id"))
        live = link.get(cid, {})
        out.append({
            "id": cid,
            "source": redact_credentials(str(c.get("source", ""))),
            "view_only": bool(c.get("view_only")),
            "state": live.get("state", "unknown"),
            "last_frame_age_s": live.get("last_frame_age_s"),
            "reconnects": live.get("reconnects"),
            "ingest": live.get("ingest"),
        })
    return out


# ---- events -----------------------------------------------------------------

_EVENT_COLUMNS = ("id", "ts", "iso", "camera_id", "rule", "priority",
                  "confidence", "reason", "track_id", "zone", "object_label",
                  "evidence_dir", "review", "reviewed_at")


def _verdict_from_row(row: dict) -> str:
    """The event's verification verdict for the API's flat shape."""
    review = row.get("review")
    if review in ("true", "ack"):
        return "confirmed"
    if review == "false":
        return "rejected"
    # An engine-confirmed alert with no operator label is still 'confirmed' by
    # TrueSight; unverified alerts are written with a marker in reason/priority.
    return "confirmed"


def _to_api_event(row: dict) -> dict:
    eid = f"evt_{row['id']}"
    return {
        "id": eid,
        "ts": row.get("ts"),
        "iso": row.get("iso"),
        "camera_id": row.get("camera_id"),
        "rule": row.get("rule"),
        "priority": row.get("priority"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
        "zone": row.get("zone"),
        "verdict": _verdict_from_row(row),
        "evidence": {
            "dir": row.get("evidence_dir"),
            "thumb": f"/api/v1/events/{eid}/evidence/thumb" if row.get("evidence_dir") else None,
            "clip": bool(row.get("evidence_dir")),
        },
        "triage": {
            "state": "new" if not row.get("review") else row.get("review"),
        },
    }


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    con.row_factory = sqlite3.Row
    return con


def read_events(db_path: str, *, limit: int = 50, cursor: int | None = None,
                camera: str | None = None, priority: str | None = None) -> dict:
    """Cursor-paged events, newest first. Cursor is the last id seen."""
    if not Path(db_path).exists():
        return {"events": [], "next_cursor": None}
    where, params = [], []
    if cursor is not None:
        where.append("id < ?"); params.append(cursor)
    if camera:
        where.append("camera_id = ?"); params.append(camera)
    if priority:
        where.append("priority = ?"); params.append(priority)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT * FROM events{clause} ORDER BY id DESC LIMIT ?"
    params.append(max(1, min(limit, 200)))
    try:
        con = _connect(db_path)
    except sqlite3.OperationalError:
        return {"events": [], "next_cursor": None, "error": "events database unavailable"}
    try:
        rows = [dict(r) for r in con.execute(sql, params).fetchall()]
    except sqlite3.OperationalError as exc:
        con.close()
        if "no such table" in str(exc):
            return {"events": [], "next_cursor": None}   # fresh site, genuinely quiet
        return {"events": [], "next_cursor": None, "error": "events database unavailable"}
    con.close()
    events = [_to_api_event(r) for r in rows]
    next_cursor = rows[-1]["id"] if len(rows) == params[-1] else None
    return {"events": events, "next_cursor": next_cursor}


def read_event(db_path: str, event_id: str) -> dict | None:
    raw_id = event_id.removeprefix("evt_")
    if not raw_id.isdigit() or not Path(db_path).exists():
        return None
    try:
        con = _connect(db_path)
        row = con.execute("SELECT * FROM events WHERE id = ?", (int(raw_id),)).fetchone()
        con.close()
    except sqlite3.OperationalError:
        return None
    return _to_api_event(dict(row)) if row else None


def read_triage(db_path: str) -> dict:
    """Counts for the triage header: to-review, total, and a priority split."""
    if not Path(db_path).exists():
        return {"to_review": 0, "total": 0, "by_priority": {}}
    try:
        con = _connect(db_path)
    except sqlite3.OperationalError:
        return {"to_review": 0, "total": 0, "by_priority": {}}
    try:
        total = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        to_review = con.execute(
            "SELECT COUNT(*) FROM events WHERE review IS NULL").fetchone()[0]
        rows = con.execute(
            "SELECT priority, COUNT(*) FROM events GROUP BY priority").fetchall()
        by_priority = {r[0]: r[1] for r in rows}
    except sqlite3.OperationalError:
        con.close()
        return {"to_review": 0, "total": 0, "by_priority": {}}
    con.close()
    return {"to_review": to_review, "total": total, "by_priority": by_priority}


def max_event_id(db_path: str) -> int:
    """Highest event id, for the WebSocket's new-alert poll. 0 when none."""
    if not Path(db_path).exists():
        return 0
    try:
        con = _connect(db_path)
        row = con.execute("SELECT MAX(id) FROM events").fetchone()
        con.close()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0
