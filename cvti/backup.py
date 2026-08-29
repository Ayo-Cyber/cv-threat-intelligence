"""Config backup and one-click restore (EP-08-T1, ARCH-08).

If the edge PC's disk dies, the hardware is replaceable and the evidence has
its own retention story — but re-drawing zones and re-tuning rules for twenty
cameras is hours of skilled work that vanishes silently. Configuration backup
is cheap, so its absence is not defensible.

What a backup contains: the site file (cameras, detector toggles, custom
English rules, notify/retention/value settings), every zone file, every
generated per-camera rules file, and the routing policy. What it deliberately
does NOT contain: `auth.db` (accounts are per-install security material — a
fresh install creates a fresh owner), `audit.db` (the tamper-evident log must
not be replayable onto another machine), and evidence (that is retention's
jurisdiction; see `backup_evidence` for the optional NAS copy).

Backups are plain zips with a manifest — recoverable with any unzip tool if
Argus itself is what died. docs/RUNBOOK.md is the disaster-recovery procedure.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path

from cvti.logging_setup import get_logger

log = get_logger(__name__)

KEEP_VERSIONS = 14          # ~2 weeks of dailies
MANIFEST = "argus-backup.json"

# repo-relative trees that ride along when present
_CONFIG_TREES = ("configs/zones", "configs/rules")
_CONFIG_FILES = ("configs/routing.json", "configs/feeds.json")


def _default_dir() -> Path:
    from cvti.utils import user_data_dir
    return user_data_dir() / "backups"


def backup_config(site_path: str | Path, dest_dir: str | Path | None = None) -> dict:
    """Write one versioned config backup zip. Returns {ok, path, entries}."""
    site_path = Path(site_path)
    dest = Path(dest_dir) if dest_dir else _default_dir()
    dest.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = dest / f"argus-config-{stamp}.zip"

    entries: list[str] = []
    tmp = out.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        def put(path: Path, arcname: str):
            zf.write(path, arcname)
            entries.append(arcname)

        if site_path.exists():
            put(site_path, "site.json")
        # Zones/rules live in the repo's configs/ during development and in the
        # USER DATA DIR on installs (where the app has been writing them since
        # the Program Files fixes). Collecting only the cwd-relative tree meant
        # an installed app's backup captured ZERO files — '0 files · 0.3 KB' on
        # the pilot's Settings page (29 Aug): a backup button that made empty
        # zips of exactly the configuration it promised to protect.
        from cvti.utils import user_data_dir
        seen: set = set()
        roots = [Path("."), user_data_dir()]
        tree_names = {"configs/zones": "zones", "configs/rules": "rules"}
        for tree, alias in tree_names.items():
            for root in roots:
                d = root / tree if root == Path(".") else root / alias
                if not d.is_dir():
                    continue
                for f in sorted(d.glob("*.json")):
                    arc = f"{tree}/{f.name}"
                    if arc not in seen:
                        seen.add(arc)
                        put(f, arc)
        for rel in _CONFIG_FILES:
            for cand in (Path(rel), user_data_dir() / "feeds" / Path(rel).name):
                if cand.exists():
                    put(cand, rel)
                    break
        zf.writestr(MANIFEST, json.dumps({
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "site_path": str(site_path),
            "entries": entries,
            "format": 1,
        }, indent=2))
    tmp.rename(out)
    _prune(dest)
    log.info("config backup written: %s (%d entries)", out, len(entries))
    return {"ok": True, "path": str(out), "entries": len(entries)}


def _prune(dest: Path, keep: int = KEEP_VERSIONS) -> None:
    old = sorted(dest.glob("argus-config-*.zip"))[:-keep]
    for f in old:
        try:
            f.unlink()
        except OSError:
            log.warning("could not prune old backup %s", f, exc_info=True)


def list_backups(dest_dir: str | Path | None = None) -> list[dict]:
    dest = Path(dest_dir) if dest_dir else _default_dir()
    out = []
    for f in sorted(dest.glob("argus-config-*.zip"), reverse=True):
        row = {"path": str(f), "size_kb": round(f.stat().st_size / 1024, 1)}
        try:
            with zipfile.ZipFile(f) as zf:
                m = json.loads(zf.read(MANIFEST))
            row.update(created_at=m.get("created_at"), entries=len(m.get("entries", [])))
        except Exception:  # noqa: BLE001 - an unreadable zip is still listed, marked
            log.warning("unreadable backup %s", f, exc_info=True)
            row["error"] = "unreadable"
        out.append(row)
    return out


def restore_config(zip_path: str | Path, site_path: str | Path) -> dict:
    """One-click restore. The CURRENT config is backed up first — restoring
    must never be the operation that destroys the only good copy."""
    zip_path, site_path = Path(zip_path), Path(site_path)
    if not zip_path.exists():
        return {"ok": False, "error": f"backup not found: {zip_path}"}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            manifest = json.loads(zf.read(MANIFEST))
            names = set(zf.namelist())
            safety = None
            if site_path.exists():
                safety = backup_config(site_path).get("path")
            restored = 0
            for name in manifest.get("entries", []):
                if name not in names:
                    continue
                target = site_path if name == "site.json" else Path(name)
                if target != site_path and (".." in name or name.startswith("/")):
                    return {"ok": False, "error": f"unsafe path in backup: {name}"}
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
                restored += 1
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        return {"ok": False, "error": f"not a valid Argus backup: {str(exc)[:120]}"}
    log.info("config restored from %s (%d files); prior config saved to %s",
             zip_path, restored, safety)
    return {"ok": True, "restored": restored, "previous_config_backup": safety,
            "note": "applies the next time monitoring starts"}


def check_events_db(db_path: str | Path) -> dict:
    """Startup integrity check with automatic quarantine (EP-08-T1).

    A corrupt events.db must never take the product down or silently masquerade
    as a quiet site. If SQLite says the file is bad, it is moved aside intact
    (evidence for recovery attempts) and a fresh store starts — LOUDLY.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return {"ok": True, "state": "fresh"}
    try:
        con = sqlite3.connect(db_path)
        result = con.execute("PRAGMA integrity_check").fetchone()[0]
        con.close()
        if result == "ok":
            return {"ok": True, "state": "ok"}
    except sqlite3.DatabaseError as exc:
        result = str(exc)
    quarantine = db_path.with_name(
        f"{db_path.stem}.corrupt-{time.strftime('%Y%m%d_%H%M%S')}{db_path.suffix}")
    try:
        db_path.rename(quarantine)
    except OSError as exc:
        return {"ok": False, "state": "corrupt",
                "error": f"integrity check failed AND quarantine failed: {exc}"}
    log.error("events.db failed integrity check (%s) — quarantined to %s; "
              "a fresh store will be created", str(result)[:120], quarantine)
    return {"ok": False, "state": "quarantined", "quarantined_to": str(quarantine),
            "detail": str(result)[:200]}


def backup_evidence(db_path: str | Path, dest_dir: str | Path) -> dict:
    """Optional evidence copy to an external drive / customer NAS — never our
    cloud, never by default. Incremental: event folders already present at the
    destination are skipped; the DB is snapshotted each run."""
    db_path, dest = Path(db_path), Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    src_events = db_path.parent / "events"
    if src_events.is_dir():
        for ev in sorted(src_events.iterdir()):
            if not ev.is_dir():
                continue
            target = dest / "events" / ev.name
            if target.exists():
                skipped += 1
                continue
            shutil.copytree(ev, target)
            copied += 1
    if db_path.exists():
        con = sqlite3.connect(db_path)
        try:
            bkp = sqlite3.connect(dest / "events.db")
            con.backup(bkp)          # consistent snapshot even mid-write
            bkp.close()
        finally:
            con.close()
    log.info("evidence backup: %d new event folder(s), %d already present, db snapshotted",
             copied, skipped)
    return {"ok": True, "copied": copied, "skipped": skipped, "dest": str(dest)}
