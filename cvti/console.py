"""Deliberate user-facing output, as distinct from logging.

`print()` was doing two unrelated jobs: telling an operator what the system is
doing (which belongs in a log — timestamped, attributed, retrievable after the
fact), and rendering output a human asked for at a terminal (a report table, a
progress line, a prompt). The first is now `get_logger(__name__)`.

This is the second. It exists so that "this text is intentionally on stdout" is
a greppable, reviewable decision rather than an accident of history — and so
that a report table is not mangled with a timestamp and a module name.

Use it only for output a person is standing there reading. If you are describing
what the system did, log it.
"""

from __future__ import annotations

import sys


def emit(*parts: object, err: bool = False, end: str = "\n", flush: bool = False) -> None:
    """Write user-facing output to stdout (or stderr with `err=True`)."""
    print(*parts, file=sys.stderr if err else sys.stdout, end=end, flush=flush)
