"""Runtime resource path — works in dev and inside a PyInstaller bundle."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# PyInstaller sets both `frozen` and `_MEIPASS`; other freezers set only the
# first. Reading `_MEIPASS` unguarded turns that into an AttributeError at
# import time, which takes the whole app down before anything can log why.
_SOURCE_BASE = Path(__file__).parent.parent.resolve()
if getattr(sys, "frozen", False):
    _BASE = Path(getattr(sys, "_MEIPASS", _SOURCE_BASE))
else:
    _BASE = _SOURCE_BASE


def resource_path(relative: str) -> Path:
    """Absolute path to a read-only bundled resource."""
    return _BASE / relative


def user_data_dir() -> Path:
    """Writable per-user directory for Argus data (configs, logs, events).

    The product is Argus; the directory used to say CVTI (the repo's working
    name). Renaming without migrating would silently orphan every existing
    site's events.db and logs, so the first call after an upgrade moves the
    old directory into place — a same-volume rename, atomic enough that a
    concurrently spawned engine sees either the old path or the new one,
    never a half-copy.
    """
    if sys.platform == "darwin":
        support = Path.home() / "Library" / "Application Support"
        base, legacy = support / "Argus", support / "CVTI"
    elif sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", str(Path.home())))
        base, legacy = appdata / "Argus", appdata / "CVTI"
    else:
        base, legacy = Path.home() / ".argus", Path.home() / ".cvti"
    if not base.exists() and legacy.is_dir():
        try:
            legacy.rename(base)
        except OSError:
            pass          # e.g. the other process won the race — base now exists
    base.mkdir(parents=True, exist_ok=True)
    return base


def writable_configs_dir() -> Path:
    """
    Return a writable configs directory.
    On first run, seeds it by copying the bundled defaults there.
    """
    dest = user_data_dir() / "configs"
    dest.mkdir(parents=True, exist_ok=True)

    src = resource_path("configs")
    if src.exists():
        for f in src.iterdir():
            if f.suffix in {".json", ".yaml"} and not (dest / f.name).exists():
                shutil.copy2(f, dest / f.name)

    return dest
