"""Persistent health history — one row a minute (EP quick-win, 25 Aug).

The health doc was a snapshot; nothing remembered it. But the pilot
agreement's FIRST success criterion is monitoring uptime, which a snapshot
cannot measure. This is the smallest thing that can: a per-minute row in its
own SQLite (its own file — the sink's lock never waits on us), pruned to 90
days.

Uptime is honest by construction: rows only exist when the engine was alive
to write them, so uptime = rows present / minutes in the window. A crashed
engine can't fake its own attendance.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from cvti.logging_setup import get_logger

log = get_logger(__name__)

DB_NAME = "health_history.db"
MIN_INTERVAL_S = 60.0
KEEP_DAYS = 90

_SCHEMA = """CREATE TABLE IF NOT EXISTS health_minutes (
    ts REAL PRIMARY KEY,
    status TEXT,
    cameras_total INTEGER,
    cameras_connected INTEGER,
    gate_reachable INTEGER,     -- 1/0/NULL (NULL = no traffic yet)
    disk_pct REAL,
    reasons TEXT                -- json list
)"""


def _db(output_dir: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(Path(output_dir) / DB_NAME), timeout=2.0)
    con.execute(_SCHEMA)
    return con


def record(output_dir: str | Path, doc: dict, now: float | None = None) -> bool:
    """Append one row, at most once per MIN_INTERVAL_S. Returns True if written."""
    now = now or time.time()
    con = _db(output_dir)
    try:
        last = con.execute("SELECT MAX(ts) FROM health_minutes").fetchone()[0] or 0.0
        if now - last < MIN_INTERVAL_S:
            return False
        cams = doc.get("cameras") or []
        gate = doc.get("gate") or {}
        reachable = gate.get("reachable")
        disk = (doc.get("disk") or {}).get("used_pct")
        con.execute(
            "INSERT OR REPLACE INTO health_minutes VALUES (?,?,?,?,?,?,?)",
            (now, doc.get("status"), len(cams),
             sum(1 for c in cams if c.get("state") == "connected"),
             None if reachable is None else int(bool(reachable)),
             disk, json.dumps((doc.get("reasons") or [])[:6])))
        # prune opportunistically — cheap, and only when we actually wrote
        con.execute("DELETE FROM health_minutes WHERE ts < ?",
                    (now - KEEP_DAYS * 86400,))
        con.commit()
        return True
    except sqlite3.OperationalError:
        log.debug("health history write skipped", exc_info=True)
        return False
    finally:
        con.close()


def stats(output_dir: str | Path, since: float, now: float | None = None) -> dict:
    """Measured uptime + camera availability over [since, now).

    uptime_pct: minutes the engine proved it was alive / minutes in the window.
    camera_availability_pct: mean of connected/total across those minutes.
    Empty dict when there is no history — callers must treat that as
    'unmeasured', never as zero.
    """
    now = now or time.time()
    path = Path(output_dir) / DB_NAME
    if not path.exists():
        return {}
    con = _db(output_dir)
    try:
        rows = con.execute(
            "SELECT cameras_total, cameras_connected FROM health_minutes "
            "WHERE ts >= ? AND ts < ?", (since, now)).fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()
    if not rows:
        return {}
    expected_minutes = max(1.0, (now - since) / 60.0)
    avail = [c / t for t, c in rows if t]
    return {
        "samples": len(rows),
        "uptime_pct": round(min(1.0, len(rows) / expected_minutes) * 100, 1),
        "camera_availability_pct":
            round(sum(avail) / len(avail) * 100, 1) if avail else None,
    }
