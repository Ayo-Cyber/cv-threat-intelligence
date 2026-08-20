"""Append-only, tamper-evident audit log.

This is the task that makes footage evidentially useful. Video with no chain of
custody and no tamper-evident access record is materially weaker if a customer
needs it in a dispute or a prosecution. It also answers the single most
important security question this product can be asked: *who disabled that
detector, and when?*

Two properties, and they are different things:

**Append-only** — no code path in this module updates or deletes an entry.
There is no `update()`, no `delete()`, and SQLite triggers refuse both even to
a direct `UPDATE`/`DELETE`, so a future contributor cannot add one by accident.

**Tamper-evident** — each entry carries a hash over its own contents and the
previous entry's hash. Editing an old row breaks every hash after it. This does
not *prevent* someone with the disk from rewriting the whole chain; it makes a
partial edit — the realistic attack, from inside the app or by someone with a
SQLite browser — detectable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from cvti.logging_setup import get_logger

log = get_logger(__name__)

GENESIS = "0" * 64

# The seven classes the plan requires. Kept as constants so a typo is a
# NameError rather than an entry nobody can find later.
LOGIN = "login"
FOOTAGE_ACCESS = "footage_access"
CONFIG_CHANGE = "config_change"
ALERT_RESOLUTION = "alert_resolution"
EVIDENCE_EXPORT = "evidence_export"
PURGE = "purge"
ROLE_CHANGE = "role_change"

ACTIONS = (LOGIN, FOOTAGE_ACCESS, CONFIG_CHANGE, ALERT_RESOLUTION,
           EVIDENCE_EXPORT, PURGE, ROLE_CHANGE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    iso TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    detail TEXT,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);

-- Append-only, enforced by the database rather than by convention. A future
-- contributor cannot add an UPDATE path without deliberately dropping these.
CREATE TRIGGER IF NOT EXISTS audit_no_update
BEFORE UPDATE ON audit
BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: entries cannot be modified');
END;

CREATE TRIGGER IF NOT EXISTS audit_no_delete
BEFORE DELETE ON audit
BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: entries cannot be deleted');
END;
"""


@dataclass
class AuditEntry:
    seq: int
    ts: float
    iso: str
    actor: str
    action: str
    target: str
    detail: dict
    prev_hash: str
    hash: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def entry_hash(ts: float, actor: str, action: str, target: str,
               detail: dict, prev_hash: str) -> str:
    """Hash over the entry's own contents plus the previous hash."""
    payload = json.dumps(
        {"ts": round(ts, 6), "actor": actor, "action": action, "target": target,
         "detail": detail, "prev": prev_hash},
        sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only log, in its own database.

    Separate from `events.db` deliberately: the record of who looked at the
    evidence must not live in the file the evidence lives in, or a single
    deletion removes both the footage and the proof of who touched it.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            log.debug("could not tighten permissions on %s", self.db_path, exc_info=True)

    def _last_hash(self) -> str:
        row = self._db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        return row["hash"] if row else GENESIS

    def record(self, actor: str, action: str, target: str = "",
               detail: dict | None = None) -> AuditEntry:
        """Append one entry. There is no counterpart that removes one."""
        if action not in ACTIONS:
            # Not fatal: an unrecognised action still belongs in the log. Losing
            # the record because the name was unexpected would be the worse bug.
            log.warning("audit: unrecognised action %r — recording it anyway", action)
        ts = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
        detail = detail or {}
        prev = self._last_hash()
        digest = entry_hash(ts, actor, action, target, detail, prev)
        cur = self._db.execute(
            "INSERT INTO audit (ts, iso, actor, action, target, detail, prev_hash, hash) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (ts, iso, actor, action, target, json.dumps(detail, default=str), prev, digest))
        self._db.commit()
        log.info("audit: %s %s %s", actor, action, target or "")
        return AuditEntry(cur.lastrowid, ts, iso, actor, action, target, detail, prev, digest)

    def entries(self, limit: int = 200, action: str = "", actor: str = "") -> list[AuditEntry]:
        sql = "SELECT * FROM audit"
        where, params = [], []
        if action:
            where.append("action = ?")
            params.append(action)
        if actor:
            where.append("actor = ?")
            params.append(actor)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(int(limit))
        out = []
        for r in self._db.execute(sql, params):
            try:
                detail = json.loads(r["detail"] or "{}")
            except ValueError:
                detail = {"raw": r["detail"]}
            out.append(AuditEntry(r["seq"], r["ts"], r["iso"], r["actor"], r["action"],
                                  r["target"] or "", detail, r["prev_hash"], r["hash"]))
        return out

    def verify(self) -> dict:
        """Walk the chain. Reports the first entry that does not check out.

        An edit to an old row changes its hash, which breaks the link every
        later row depends on — so a partial rewrite is visible even though a
        full one is not preventable.
        """
        prev = GENESIS
        checked = 0
        for r in self._db.execute("SELECT * FROM audit ORDER BY seq ASC"):
            try:
                detail = json.loads(r["detail"] or "{}")
            except ValueError:
                detail = {"raw": r["detail"]}
            expected = entry_hash(r["ts"], r["actor"], r["action"], r["target"] or "",
                                  detail, r["prev_hash"])
            if r["prev_hash"] != prev:
                return {"ok": False, "checked": checked, "broken_at": r["seq"],
                        "reason": "chain link does not match the previous entry "
                                  "— an entry was inserted or removed"}
            if r["hash"] != expected:
                return {"ok": False, "checked": checked, "broken_at": r["seq"],
                        "reason": "entry contents do not match its hash — it was edited"}
            prev = r["hash"]
            checked += 1
        return {"ok": True, "checked": checked, "head": prev}

    def export(self, dest: str | Path) -> Path:
        """Write the whole chain out for an auditor. Owner-only, upstream."""
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [e.to_dict() for e in reversed(self.entries(limit=10 ** 9))]
        path.write_text(json.dumps(
            {"exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "verification": self.verify(), "entries": rows}, indent=2, default=str))
        log.info("audit log exported: %d entr(ies) -> %s", len(rows), path)
        return path

    def close(self) -> None:
        self._db.close()
