"""Retention, purge, and legal hold.

Two problems, one mechanism.

*Operationally*: an unattended edge PC with no purge fills its disk, and when it
does, writes fail and evidence stops being recorded at exactly the moment it is
most needed — with nobody watching.

*Legally*: storage limitation is not optional under GDPR/NDPR. A system with no
deletion path cannot honour an erasure request or answer a procurement
questionnaire.

The tension to respect, and the reason this is not a cron job with `find -mtime`:
**blind time-based deletion would destroy the exact records a customer needs.**
An incident nobody has reviewed yet, or one explicitly flagged for a case, must
survive its own expiry. Retention deletes what is old *and* settled.

Deletion order is deliberate. Files go first, then the row. A row without files
is a visible, recoverable inconsistency; files without a row are personal data
nobody can find, account for, or erase on request.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from cvti.health import component
from cvti.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_RETENTION_DAYS = 30
DEFAULT_DISK_WARN_PCT = 85.0
DEFAULT_DISK_CRITICAL_PCT = 95.0
# Long enough that a purge is never the thing keeping the box busy, short enough
# that a site filling up is caught the same day.
DEFAULT_INTERVAL_SECONDS = 3600.0


@dataclass
class RetentionPolicy:
    days: int = DEFAULT_RETENTION_DAYS
    disk_warn_pct: float = DEFAULT_DISK_WARN_PCT
    disk_critical_pct: float = DEFAULT_DISK_CRITICAL_PCT
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    enabled: bool = True

    @classmethod
    def from_site(cls, meta: dict) -> "RetentionPolicy":
        def num(key, default):
            try:
                return type(default)(meta.get(key) if meta.get(key) is not None else default)
            except (TypeError, ValueError):
                return default
        return cls(days=max(1, num("retention_days", DEFAULT_RETENTION_DAYS)),
                   disk_warn_pct=num("disk_warn_pct", DEFAULT_DISK_WARN_PCT),
                   disk_critical_pct=num("disk_critical_pct", DEFAULT_DISK_CRITICAL_PCT))

    def to_dict(self) -> dict:
        return {"days": self.days, "disk_warn_pct": self.disk_warn_pct,
                "disk_critical_pct": self.disk_critical_pct, "enabled": self.enabled}


def disk_status(path: str | Path, policy: RetentionPolicy | None = None) -> dict:
    """Free space where evidence is written, and whether that is a problem yet."""
    policy = policy or RetentionPolicy()
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return {"available": False}
    used_pct = 100.0 * usage.used / usage.total if usage.total else 0.0
    level = ("critical" if used_pct >= policy.disk_critical_pct
             else "warning" if used_pct >= policy.disk_warn_pct else "ok")
    return {"available": True, "used_pct": round(used_pct, 1),
            "free_gb": round(usage.free / 2**30, 1),
            "total_gb": round(usage.total / 2**30, 1), "level": level}


class RetentionManager:
    """Purges expired, settled evidence. Never touches anything on legal hold."""

    def __init__(self, output_dir: str | Path, policy: RetentionPolicy | None = None,
                 db_path: str | Path | None = None) -> None:
        self.root = Path(output_dir)
        self.policy = policy or RetentionPolicy()
        self.db_path = Path(db_path) if db_path else (self.root / "events.db")
        self.events_dir = self.root / "events"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._health = component("retention")
        self.last_run: dict = {}

    # --- what may be deleted -------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _hold_clause() -> str:
        """SQL for 'this must survive its own expiry'.

        An OPEN incident — one nobody has resolved — must not be deleted on a
        timer while the question is still live. That includes an alert a guard
        has claimed but not concluded: acknowledging is the start of the work,
        not the end of it. `legal_hold` is the explicit exemption an operator
        sets for a case. Purge deletes what is old AND settled; settled means
        resolved.
        """
        return ("(COALESCE(legal_hold, 0) = 1 OR "
                " COALESCE(state, CASE WHEN review IN ('true','false') "
                "   THEN 'resolved' ELSE 'open' END) != 'resolved')")

    def expired(self, now: float | None = None, limit: int = 0) -> list[sqlite3.Row]:
        now = now if now is not None else time.time()
        cutoff = now - self.policy.days * 86400
        sql = ("SELECT id, ts, iso, camera_id, rule, evidence_dir, review, "
               "COALESCE(legal_hold, 0) AS legal_hold FROM events "
               f"WHERE ts < ? AND NOT {self._hold_clause()} ORDER BY ts ASC")
        if limit:
            sql += f" LIMIT {int(limit)}"
        con = self._connect()
        try:
            return con.execute(sql, (cutoff,)).fetchall()
        except sqlite3.OperationalError:
            return []              # no events table yet
        finally:
            con.close()

    def held(self, now: float | None = None) -> dict:
        """Expired-but-retained, so the reason is visible rather than surprising."""
        now = now if now is not None else time.time()
        cutoff = now - self.policy.days * 86400
        con = self._connect()
        try:
            row = con.execute(
                "SELECT SUM(CASE WHEN COALESCE(legal_hold,0)=1 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN COALESCE(legal_hold,0)=0 THEN 1 ELSE 0 END) "
                f"FROM events WHERE ts < ? AND {self._hold_clause()}", (cutoff,)).fetchone()
            return {"legal_hold": row[0] or 0, "unreviewed": row[1] or 0}
        except sqlite3.OperationalError:
            return {"legal_hold": 0, "unreviewed": 0}
        finally:
            con.close()

    # --- deletion ------------------------------------------------------------
    def _delete_event(self, con: sqlite3.Connection, row: sqlite3.Row) -> bool:
        """Files first, then the row. Returns False if the files did not go.

        Deleting the row first would leave personal data on disk that nothing
        references — unfindable, unaccountable, and impossible to honour an
        erasure request against. A row whose files are already gone is merely
        untidy, and the next pass retries it.
        """
        ev_dir = row["evidence_dir"]
        if ev_dir:
            path = Path(ev_dir)
            if not path.is_absolute():
                path = self.root / path
            # Never delete outside our own events directory, whatever the row says.
            try:
                path.relative_to(self.events_dir.resolve())
            except ValueError:
                try:
                    path.relative_to(self.events_dir)
                except ValueError:
                    log.warning("retention: refusing to delete %s — outside %s",
                                path, self.events_dir)
                    return False
            if path.exists():
                try:
                    shutil.rmtree(path)
                except OSError as exc:
                    self._health.failed(exc, log, f"deleting evidence for event {row['id']}")
                    return False
        con.execute("DELETE FROM events WHERE id = ?", (row["id"],))
        return True

    def purge(self, now: float | None = None, dry_run: bool = False) -> dict:
        """Delete expired, settled events. Returns what happened."""
        if not self.policy.enabled:
            return {"skipped": "retention disabled"}
        rows = self.expired(now)
        result = {"examined": len(rows), "deleted": 0, "failed": 0,
                  "held": self.held(now), "dry_run": dry_run,
                  "retention_days": self.policy.days, "orphans_removed": 0,
                  "would_delete": [r["id"] for r in rows] if dry_run else []}
        if dry_run:
            self.last_run = {**result, "at": time.time()}
            return result
        if not rows:
            # An orphan exists independently of expiry — a site where nothing
            # has aged out yet can still be holding evidence no record points
            # to, which is the copy nobody can find or erase on request.
            result["orphans_removed"] = self.sweep_orphans()
            self.last_run = {**result, "at": time.time()}
            return result

        con = self._connect()
        try:
            for row in rows:
                try:
                    if self._delete_event(con, row):
                        con.commit()          # per event: a failure part-way leaves
                        result["deleted"] += 1  # everything before it consistently gone
                    else:
                        con.rollback()
                        result["failed"] += 1
                except sqlite3.Error as exc:
                    con.rollback()
                    result["failed"] += 1
                    self._health.failed(exc, log, f"purging event {row['id']}")
        finally:
            con.close()

        result["orphans_removed"] = self.sweep_orphans()
        self._health.ok(result["deleted"])
        log.info("retention: deleted %d of %d expired event(s), %d failed, "
                 "held %s, %d orphan dir(s) removed",
                 result["deleted"], result["examined"], result["failed"],
                 result["held"], result["orphans_removed"])
        self.last_run = {**result, "at": time.time()}
        return result

    def sweep_orphans(self) -> int:
        """Evidence directories with no event row.

        These are the dangerous ones: personal data nobody can find through the
        product, so nobody can account for it or erase it on request. Left behind
        by an interrupted purge, a crash mid-write, or a deleted database.
        """
        if not self.events_dir.exists():
            return 0
        con = self._connect()
        try:
            known = {Path(r[0]).name for r in
                     con.execute("SELECT evidence_dir FROM events WHERE evidence_dir IS NOT NULL")}
        except sqlite3.OperationalError:
            return 0                          # no table: cannot tell orphan from live
        finally:
            con.close()

        removed = 0
        for entry in self.events_dir.iterdir():
            if not entry.is_dir() or entry.name in known:
                continue
            try:
                shutil.rmtree(entry)
                removed += 1
                log.warning("retention: removed orphaned evidence directory %s", entry.name)
            except OSError as exc:
                self._health.failed(exc, log, f"removing orphan {entry.name}")
        return removed

    def emergency_purge(self, now: float | None = None) -> dict:
        """Disk is nearly full. Delete oldest-first until it is not.

        Still refuses to touch anything on hold. A full disk stops evidence being
        written at all, so doing nothing is not the safe option — but destroying
        an open case to free space is not either.
        """
        status = disk_status(self.root, self.policy)
        if not status.get("available") or status["level"] != "critical":
            return {"triggered": False, "disk": status}

        log.warning("retention: disk at %.1f%% — emergency purge, oldest first",
                    status["used_pct"])
        deleted = 0
        con = self._connect()
        try:
            while True:
                status = disk_status(self.root, self.policy)
                if status.get("level") != "critical":
                    break
                rows = self._oldest_deletable(con, limit=25)
                if not rows:
                    log.error("retention: disk at %.1f%% and nothing is deletable — "
                              "everything remaining is unreviewed or on legal hold",
                              status.get("used_pct", 0.0))
                    break
                for row in rows:
                    if self._delete_event(con, row):
                        con.commit()
                        deleted += 1
                    else:
                        con.rollback()
        finally:
            con.close()
        return {"triggered": True, "deleted": deleted, "disk": disk_status(self.root, self.policy)}

    def _oldest_deletable(self, con: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
        try:
            return con.execute(
                "SELECT id, evidence_dir FROM events "
                f"WHERE NOT {self._hold_clause()} ORDER BY ts ASC LIMIT ?", (limit,)).fetchall()
        except sqlite3.OperationalError:
            return []

    # --- scheduling ----------------------------------------------------------
    def start(self) -> "RetentionManager":
        if not self.policy.enabled:
            log.info("retention: disabled by policy — nothing will be purged")
            return self

        def loop() -> None:
            while not self._stop.wait(self.policy.interval_seconds):
                try:
                    self.emergency_purge()
                    self.purge()
                except Exception as exc:  # noqa: BLE001 - purge must never stop the engine
                    self._health.failed(exc, log, "scheduled purge")

        self._thread = threading.Thread(target=loop, name="retention", daemon=True)
        self._thread.start()
        log.info("retention: %d-day policy active, checking every %.0fs",
                 self.policy.days, self.policy.interval_seconds)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def status(self) -> dict:
        return {"policy": self.policy.to_dict(), "disk": disk_status(self.root, self.policy),
                "held": self.held(), "last_run": self.last_run}
