"""Runtime resource path — works in dev and inside a PyInstaller bundle."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    _BASE = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    _BASE = Path(__file__).parent.parent.resolve()


def resource_path(relative: str) -> Path:
    """Absolute path to a read-only bundled resource."""
    return _BASE / relative


def user_data_dir() -> Path:
    """Writable per-user directory for CVTI data (configs, logs, etc.)."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "CVTI"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", str(Path.home()))) / "CVTI"
    else:
        base = Path.home() / ".cvti"
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
