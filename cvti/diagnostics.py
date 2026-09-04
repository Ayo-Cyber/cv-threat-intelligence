"""Diagnostics bundle — what a customer sends us when something is wrong.

The hard rule here is what does NOT go in. This is a surveillance product: the
evidence directory holds images of identifiable people, and the events database
holds where they were and what a model thought they were doing. A support bundle
that quietly includes those turns "send us your logs" into an unlawful transfer
of personal data, and the customer would have no way to know.

So the bundle carries logs and counts, never frames, clips, or event rows. The
manifest states that explicitly, so the person sending it can verify the claim
rather than trust it.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

from cvti.logging_setup import get_logger, resolve_log_dir

log = get_logger(__name__)

# Anything matching these is personal data and never enters a bundle.
EXCLUDED = ("*.jpg", "*.jpeg", "*.png", "*.mp4", "*.avi", "*.mov", "*.db",
            "*.db-wal", "*.db-shm")


def _version() -> str:
    try:
        from cvti.utils import argus_version
        return argus_version()
    except Exception:  # noqa: BLE001 - a version lookup must never break support
        log.debug("version lookup failed", exc_info=True)
        return "unknown"


def _disk(path: Path) -> dict:
    try:
        usage = shutil.disk_usage(path)
        return {"total_gb": round(usage.total / 2**30, 1),
                "free_gb": round(usage.free / 2**30, 1),
                "used_pct": round(100 * usage.used / usage.total, 1)}
    except OSError:
        return {}


def _event_counts(db_path: Path) -> dict:
    """Aggregate counts only — never rows, reasons, or camera identities."""
    if not db_path.exists():
        return {"database": "absent"}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            total = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            unreviewed = con.execute(
                "SELECT COUNT(*) FROM events WHERE review IS NULL").fetchone()[0]
            oldest, newest = con.execute("SELECT MIN(ts), MAX(ts) FROM events").fetchone()
            out = {"events_total": total, "events_unreviewed": unreviewed,
                   "oldest_event_ts": oldest, "newest_event_ts": newest}
            try:
                row = con.execute(
                    "SELECT SUM(shown), SUM(rejected), SUM(deduped), SUM(errors) "
                    "FROM suppression_daily").fetchone()
                out["suppression_totals"] = {
                    "shown": row[0] or 0, "rejected": row[1] or 0,
                    "deduped": row[2] or 0, "errors": row[3] or 0}
            except sqlite3.OperationalError:
                out["suppression_totals"] = None      # ledger predates this build
            return out
        finally:
            con.close()
    except sqlite3.Error as exc:
        return {"database": f"unreadable: {str(exc)[:120]}"}


def health_snapshot(output_dir: str | Path) -> dict:
    """Everything about the deployment that is not about the people in frame."""
    out_dir = Path(output_dir)
    gate_health: dict | str = "absent"
    try:
        gate_health = json.loads((out_dir / "gate_health.json").read_text())
    except (OSError, ValueError):
        pass

    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "frozen": bool(getattr(sys, "frozen", False)),
        },
        "argus": {
            "version": _version(),
            "log_level": os.environ.get("ARGUS_LOG_LEVEL", "INFO"),
            "mock_gate_allowed": os.environ.get("ARGUS_ALLOW_MOCK_GATE") == "1",
            "output_dir": str(out_dir),
        },
        "gate_health": gate_health,
        "events": _event_counts(out_dir / "events.db"),
        "disk": _disk(out_dir if out_dir.exists() else Path.home()),
    }


def build_bundle(output_dir: str | Path, dest: str | Path | None = None) -> Path:
    """Zip logs + a health snapshot. Returns the archive path.

    Never includes evidence frames, clips, or the events database — see EXCLUDED.
    """
    out_dir = Path(output_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest_path = Path(dest) if dest else (out_dir / f"argus-diagnostics-{stamp}.zip")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = health_snapshot(out_dir)
    log_dir = resolve_log_dir(out_dir)
    included: list[str] = []

    with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if log_dir.exists():
            for entry in sorted(log_dir.iterdir()):
                if not entry.is_file():
                    continue
                if any(entry.match(pattern) for pattern in EXCLUDED):
                    continue           # defensive: nothing personal should be here anyway
                zf.write(entry, f"logs/{entry.name}")
                included.append(f"logs/{entry.name}")
        # The engine subprocess's stdout/stderr — where its tracebacks land.
        # It lives beside events.db, not in the log dir, so every support
        # bundle before 3 Sep shipped WITHOUT the one file that explains a
        # crash loop (both pilot debugging sessions needed it hand-fetched).
        for name in ("monitor.log", "monitor.log.1"):
            candidate = out_dir / name
            if candidate.is_file():
                try:
                    zf.write(candidate, f"logs/{name}")
                    included.append(f"logs/{name}")
                except OSError:
                    log.warning("could not add %s to the bundle", name, exc_info=True)

        # Stage-by-stage timing percentiles (decode / detect / verify queue /
        # verify inference / English scans) plus CPU and memory at capture.
        # This is the file that turns "it's slow" into a named stage — the
        # entire point of the 4 Sep instrumentation build. Counts only,
        # never content.
        for name in ("perf_report.json", "gate_health.json"):
            candidate = out_dir / name
            if candidate.is_file():
                try:
                    zf.write(candidate, name)
                    included.append(name)
                except OSError:
                    log.warning("could not add %s to the bundle", name, exc_info=True)

        zf.writestr("health.json", json.dumps(snapshot, indent=2, default=str))
        included.append("health.json")
        zf.writestr("MANIFEST.txt",
                    "Argus diagnostics bundle\n"
                    f"captured: {snapshot['captured_at']}\n\n"
                    "CONTAINS: application logs and a health snapshot (counts, versions,\n"
                    "disk, gate status).\n\n"
                    "DOES NOT CONTAIN: evidence frames, video clips, the events database,\n"
                    "or any image of any person. Only aggregate counts are included.\n\n"
                    "Files:\n" + "\n".join(f"  {name}" for name in included) + "\n")

    log.info("diagnostics bundle written to %s (%d file(s))", dest_path, len(included))
    return dest_path
