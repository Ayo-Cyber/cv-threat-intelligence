"""Logging for Argus — one place to configure it, one way to get a logger.

The problem this solves: a customer says "it stopped alerting last night" and
there is no way to answer them. `print()` output goes to a terminal nobody was
watching, and in the packaged build it goes nowhere retrievable at all. Every
epic after this one either emits logs or is undiagnosable without them.

Usage — in any module under cvti/:

    from cvti.logging_setup import get_logger
    log = get_logger(__name__)
    log.info("engine started with %d camera(s)", n)

Entrypoints call `setup_logging()` once, early, naming which component they are.
Library modules never call it: they just `get_logger(__name__)` and inherit
whatever the process configured.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

DEFAULT_LEVEL = "INFO"
LEVEL_ENV = "ARGUS_LOG_LEVEL"
DIR_ENV = "ARGUS_LOG_DIR"

# 10 MB x 5 keeps roughly a week of a busy site inside 50 MB, which is small
# enough to attach to a support email after `Download diagnostics` zips it.
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

# Console gets the short form (a human is watching it); the file gets the module
# and line, because that is what you read six hours after the fact.
_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d — %(message)s"
_CONSOLE_FORMAT = "%(levelname)-8s %(name)s — %(message)s"

_configured: dict[str, Path] = {}    # component -> log file, so setup is idempotent


def get_logger(name: str) -> logging.Logger:
    """The logger for a module. Pass `__name__` so records carry attribution.

    A module run as `python -m cvti.serving.pipeline` has `__name__ == "__main__"`,
    which would label every record from the engine's own module `__main__` and
    throw away the attribution this exists for. Python keeps the real dotted name
    on the module spec, so recover it.
    """
    if name == "__main__":
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        real = getattr(spec, "name", None)
        if real:
            name = real
    return logging.getLogger(name)


def resolve_log_dir(output_dir: str | Path | None = None) -> Path:
    """Where logs go, in order of preference.

    The frozen build is the case that matters: it may be launched from a
    read-only mount or with a working directory the user cannot write to, so a
    relative `runs/logs` silently fails to be created and the logs vanish —
    exactly the build where the user has no terminal to fall back on. There, the
    per-user application-support directory always wins.
    """
    override = os.environ.get(DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()

    if getattr(sys, "frozen", False):
        from cvti.utils import user_data_dir
        return user_data_dir() / "logs"

    if output_dir:
        return Path(output_dir) / "logs"

    from cvti.utils import user_data_dir
    return user_data_dir() / "logs"


def setup_logging(
    output_dir: str | Path | None = None,
    *,
    component: str = "argus",
    level: str | None = None,
    console: bool = True,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
) -> Path:
    """Configure root logging for this process. Idempotent. Returns the log file.

    `component` names the log file (`argus-engine.log`, `argus-app.log`). The app
    runs the engine as a subprocess pointed at the *same* output directory, so a
    single shared file would mean two processes rotating one handle — which
    silently loses records on POSIX and fails outright on Windows, where the open
    file cannot be renamed.
    """
    if component in _configured:
        return _configured[component]

    level_name = (level or os.environ.get(LEVEL_ENV) or DEFAULT_LEVEL).upper()
    resolved = getattr(logging, level_name, None)
    if not isinstance(resolved, int):
        resolved = logging.INFO

    log_dir = resolve_log_dir(output_dir)
    root = logging.getLogger()
    root.setLevel(resolved)

    path = log_dir / f"{component}.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        file_handler.setLevel(resolved)
        root.addHandler(file_handler)
    except OSError as exc:
        # An unwritable log directory must not stop the engine from detecting
        # threats. Fall back to console only and say so, loudly, once.
        path = Path(os.devnull)
        # EXEMPT from the no-print rule: this is the failure path of logging
        # itself, so it cannot use a logger to report it.
        print(f"[logging] cannot write to {log_dir} ({exc}); console only", file=sys.stderr)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        stream.setLevel(resolved)
        root.addHandler(stream)

    # These two are chatty at DEBUG and neither tells us anything about Argus.
    logging.getLogger("urllib3").setLevel(max(resolved, logging.WARNING))
    logging.getLogger("PIL").setLevel(max(resolved, logging.WARNING))

    _configured[component] = path
    get_logger(__name__).info(
        "logging started — component=%s level=%s file=%s", component, level_name, path)
    return path


def configured_log_files() -> list[Path]:
    """Every log file this process knows about, newest rotation included."""
    out: list[Path] = []
    for path in _configured.values():
        if path == Path(os.devnull):
            continue
        out.extend(sorted(path.parent.glob(f"{path.name}*")))
    return out


def reset_for_tests() -> None:
    """Drop handlers and configuration state. Tests only."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # noqa: BLE001 - SILENT-OK: this is logging tearing
            pass                # itself down; there is no logger left to report to.
    _configured.clear()
