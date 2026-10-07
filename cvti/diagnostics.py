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
import re
import shutil
import sqlite3
import sys
import time
import zipfile
import uuid
from pathlib import Path

from cvti.logging_setup import get_logger, resolve_log_dir

log = get_logger(__name__)

# Anything matching these is personal data and never enters a bundle.
EXCLUDED = ("*.jpg", "*.jpeg", "*.png", "*.mp4", "*.avi", "*.mov", "*.db",
            "*.db-wal", "*.db-shm")

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_LOG_FILES = 20
_PRIVATE_KEY = re.compile(r"password|passwd|secret|token|authorization|api[_-]?key|credential|image|jpeg|frame_uri|frames|thumbnail", re.I)


def redact_text(text: str) -> str:
    text = re.sub(r"(?i)([a-z][a-z0-9+.-]*://)[^\s/\"<>]+@", r"\1[REDACTED]@", text)
    text = re.sub(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9+/=._-]+", r"\1 [REDACTED]", text)
    text = re.sub(r"(?i)([?&](?:token|key|password|secret|api_key|access_token)=)[^&\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r'''(?ix)(["']?(?:password|passwd|secret|token|authorization|api_key|access_token)["']?\s*[:=]\s*)(?:"[^"\n]*"|'[^'\n]*'|[^\s,;}&]+)''', r"\1[REDACTED]", text)
    return re.sub(r"data:(?:image|video)/[^\s\"']+", "[MEDIA OMITTED]", text, flags=re.I)


def redact_value(value):
    if isinstance(value, dict):
        return {key: "[REDACTED]" if _PRIVATE_KEY.search(key) else redact_value(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value


def _log_tail(path: Path) -> str:
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - MAX_FILE_BYTES))
        data = stream.read(MAX_FILE_BYTES)
    if size > MAX_FILE_BYTES:
        # Drop a partial leading line rather than exposing half a credential.
        data = data.partition(b"\n")[2]
        return "[Older log lines omitted]\n" + redact_text(data.decode("utf-8", errors="replace"))
    return redact_text(data.decode("utf-8", errors="replace"))


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
        health_file = out_dir / "gate_health.json"
        if not health_file.is_symlink() and health_file.stat().st_size <= MAX_FILE_BYTES:
            gate_health = json.loads(health_file.read_text())
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


def build_bundle(output_dir: str | Path, dest: str | Path | None = None,
                 *, site_path: str | Path | None = None) -> Path:
    """Zip logs + a health snapshot. Returns the archive path.

    Never includes evidence frames, clips, or the events database — see EXCLUDED.
    """
    out_dir = Path(output_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    dest_path = Path(dest) if dest else (out_dir / f"argus-diagnostics-{stamp}.zip")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = redact_value(health_snapshot(out_dir))
    log_dir = resolve_log_dir(out_dir)
    included: list[str] = []

    with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if log_dir.exists():
            for entry in sorted(log_dir.iterdir()):
                if len(included) >= MAX_LOG_FILES:
                    break
                if entry.is_symlink() or not entry.is_file() or not re.fullmatch(r".+\.log(?:\.\d+)?", entry.name):
                    continue
                if any(entry.match(pattern) for pattern in EXCLUDED):
                    continue           # defensive: nothing personal should be here anyway
                try:
                    zf.writestr(f"logs/{entry.name}", _log_tail(entry))
                    included.append(f"logs/{entry.name}")
                except OSError:
                    log.warning("could not read a diagnostic log", exc_info=True)
        # The engine subprocess's stdout/stderr — where its tracebacks land.
        # It lives beside events.db, not in the log dir, so every support
        # bundle before 3 Sep shipped WITHOUT the one file that explains a
        # crash loop (both pilot debugging sessions needed it hand-fetched).
        for name in ("monitor.log", "monitor.log.1"):
            candidate = out_dir / name
            if candidate.is_file() and not candidate.is_symlink() and f"logs/{name}" not in included:
                try:
                    zf.writestr(f"logs/{name}", _log_tail(candidate))
                    included.append(f"logs/{name}")
                except OSError:
                    log.warning("could not add %s to the bundle", name, exc_info=True)

        # Stage-by-stage timing percentiles (decode / detect / verify queue /
        # verify inference / English scans) plus CPU and memory at capture.
        # This is the file that turns "it's slow" into a named stage — the
        # entire point of the 4 Sep instrumentation build. Counts only,
        # never content.
        # english_rules_status.json is the scanner's own account of every
        # "describe in English" rule — scans, matches, errors, last outcome.
        # Three pilot reports said those rules "don't run"; the file that
        # says whether they even scanned was not in the zip until 22 Sep.
        for name in ("perf_report.json", "gate_health.json", "english_rules_status.json"):
            candidate = out_dir / name
            if candidate.is_file() and not candidate.is_symlink() and candidate.stat().st_size <= MAX_FILE_BYTES:
                try:
                    value = json.loads(candidate.read_text(encoding="utf-8"))
                    zf.writestr(name, json.dumps(redact_value(value), indent=2))
                    included.append(name)
                except (OSError, ValueError):
                    log.warning("could not add %s to the bundle", name, exc_info=True)

        if site_path is not None:
            from cvti.api.sources import DETECTOR_FLAGS
            try:
                site = json.loads(Path(site_path).read_text(encoding="utf-8"))
                cameras = [{"id": camera.get("id"),
                            "detectors": {flag: bool(camera.get(flag)) for flag in DETECTOR_FLAGS},
                            "has_zones": bool(camera.get("zones")),
                            "view_only": bool(camera.get("view_only"))}
                           for camera in site.get("cameras", [])]
                zf.writestr("camera_configuration.json", json.dumps(redact_value(cameras), indent=2))
                included.append("camera_configuration.json")
            except (OSError, ValueError, TypeError, AttributeError):
                log.warning("camera configuration summary unavailable", exc_info=True)

        zf.writestr("health.json", json.dumps(snapshot, indent=2, default=str))
        included.append("health.json")
        zf.writestr("MANIFEST.txt",
                    "Argus diagnostics bundle\n"
                    f"captured: {snapshot['captured_at']}\n\n"
                    "CONTAINS: application logs and a health snapshot (counts, versions,\n"
                    "disk, gate status).\n\n"
                    "DOES NOT CONTAIN: evidence frames, video clips, the events database,\n"
                    "or any image of any person. Known credential fields are redacted.\n"
                    "Logs and health may contain camera names, internal addresses and\n"
                    "operational text. Inspect this archive before sharing. No upload\n"
                    "is performed. Logs are limited to recent tails; large/invalid\n"
                    "status files and symlinks are omitted.\n\n"
                    "Files:\n" + "\n".join(f"  {name}" for name in included) + "\n")

    log.info("diagnostics bundle written to %s (%d file(s))", dest_path, len(included))
    return dest_path
