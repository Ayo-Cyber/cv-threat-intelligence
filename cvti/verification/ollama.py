"""Ollama helpers for the local (offline) verification backend.

The verification gate talks to Ollama over its OpenAI-compatible endpoint
(http://localhost:11434/v1). These helpers cover the operational bits the gate
itself doesn't: checking the server is up, checking a model is present, and
pulling it on first run.

Nothing here imports heavy deps — it's plain stdlib HTTP so it stays cheap to
import from the app's startup path.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from typing import Callable, Iterator
from urllib import error as urlerror
from urllib import request as urlrequest

from cvti.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b-it-qat"


def server_up(host: str = DEFAULT_HOST, timeout: float = 2.0) -> bool:
    """True if an Ollama server answers at `host`."""
    try:
        with urlrequest.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urlerror.URLError, OSError):
        return False


def installed_models(host: str = DEFAULT_HOST, timeout: float = 5.0) -> list[str]:
    """Names of models already pulled on the server (empty if unreachable)."""
    try:
        with urlrequest.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urlerror.URLError, OSError, json.JSONDecodeError):
        return []
    return [m.get("name", "") for m in data.get("models", [])]


def has_model(model: str, host: str = DEFAULT_HOST) -> bool:
    """True if `model` (with or without an explicit :tag) is present."""
    names = installed_models(host)
    return any(n == model or n.split(":", 1)[0] == model.split(":", 1)[0] for n in names)


def pull_model(
    model: str,
    host: str = DEFAULT_HOST,
    on_progress: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """Pull `model`, yielding human-readable status lines as it streams.

    Uses Ollama's /api/pull streaming endpoint so the app can show progress.
    Raises RuntimeError if the server is unreachable.
    """
    req = urlrequest.Request(
        f"{host.rstrip('/')}/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        resp = urlrequest.urlopen(req, timeout=None)
    except (urlerror.URLError, OSError) as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {host} to pull '{model}'. Is it running?"
        ) from exc

    with resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = event.get("status", "")
            if event.get("total") and event.get("completed") is not None:
                pct = 100.0 * event["completed"] / max(1, event["total"])
                status = f"{status} {pct:.0f}%"
            if status and on_progress:
                on_progress(status)
            if status:
                yield status


def bundled_binary() -> str | None:
    """Path to the Ollama binary shipped inside the app bundle, if present.

    Build scripts drop it at vendor/ollama/<platform>/ollama[.exe]; the spec
    copies that into the frozen bundle. Returns None in dev / when not bundled.
    """
    exe = "ollama.exe" if sys.platform == "win32" else "ollama"
    plat = {"win32": "windows", "darwin": "darwin"}.get(sys.platform, "linux")
    try:
        from cvti.utils import resource_path
        candidate = resource_path(f"vendor/ollama/{plat}/{exe}")
    except Exception as exc:  # noqa: BLE001 - resource resolution is best-effort
        log.warning("local VLM call failed", exc_info=True)
        return None
    return str(candidate) if candidate.exists() else None


def ollama_binary() -> str | None:
    """Path to an Ollama binary: bundled one first, then one on PATH, else None."""
    return bundled_binary() or shutil.which("ollama")


def start_server() -> bool:
    """Best-effort launch of `ollama serve` in the background. True if spawned."""
    binary = ollama_binary()
    if not binary:
        return False
    # A bundled binary may lose its executable bit through packaging — restore it.
    try:
        current = os.stat(binary).st_mode
        os.chmod(binary, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass
    try:
        subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except OSError:
        return False
