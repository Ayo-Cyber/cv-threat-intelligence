"""Close what the tests open, so Windows can delete it afterwards.

Nearly every backend test builds its stores inside a TemporaryDirectory and
lets them fall out of scope. Those stores hold SQLite connections (auth.db,
audit.db, events.db) and logging holds open file handles, and Windows refuses
to delete a file a process still holds open -- so the directory cleanup raised
PermissionError and the test failed, on Windows only. POSIX unlinks open files
happily, which is why the suite was green everywhere else while 210 assertions
failed on Windows.

Two things had to be right:

  * WHAT is tracked. Tracking ConsoleBackend alone left 53 failures, because
    the API, cvti/serving/mobile.py and cvti/security/recovery.py each build
    their own stores. Tracking the STORE CLASSES covers every creator.
  * WHEN they are closed. An autouse fixture runs AFTER unittest's tearDown,
    and tearDown is where these tests call cleanup() -- so the removal had
    already failed (166 -> 165). Closing from cleanup itself is early enough
    however the directory goes away.

The tracking lives here, in the test tree, so the product keeps its ordinary
constructors.
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.app.console_backend import ConsoleBackend
from cvti.security.accounts import AccountStore
from cvti.security.audit import AuditLog

_live: list = []


def _track(cls):
    """Remember every instance so it can be closed before a directory goes."""
    original = cls.__init__

    def tracking_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _live.append(self)

    tracking_init.__wrapped__ = original       # so a second import is a no-op
    if not hasattr(cls.__init__, "__wrapped__"):
        cls.__init__ = tracking_init


for _cls in (ConsoleBackend, AccountStore, AuditLog):
    _track(_cls)

with contextlib.suppress(Exception):           # optional: not every tree has it
    from cvti.serving.alert_sink import AlertSink
    _track(AlertSink)


def _paths_of(owner) -> list:
    """Every file path this owner might be holding open."""
    out = []
    for attr in ("db_path", "_home_db", "events_dir", "site_path"):
        value = getattr(owner, attr, None)
        if isinstance(value, (str, Path)):
            with contextlib.suppress(Exception):
                out.append(Path(value).resolve())
    return out


def _close_under(directory: Path | None) -> None:
    """Close tracked owners, or only those living inside `directory`.

    Scoping matters: closing everything on any cleanup shut a store that a
    later test still held, which surfaced as
    "sqlite3.ProgrammingError: Cannot operate on a closed database".
    A store outside the directory being removed cannot be blocking it.
    """
    keep = []
    for owner in _live:
        inside = directory is None or any(
            directory == path or directory in path.parents for path in _paths_of(owner))
        if not inside:
            keep.append(owner)
            continue
        with contextlib.suppress(Exception):   # teardown must never fail a test
            owner.close()
    _live[:] = keep
    # NOT logging's file handlers: tearing those off the root logger on every
    # cleanup wedged the suite (past 600s instead of 170s). The handful of
    # argus-engine.log/t.log locks stay, and belong to the tests that
    # configure logging into a temp directory.


_original_cleanup = tempfile.TemporaryDirectory.cleanup


def _cleanup_closing_handles(self) -> None:
    with contextlib.suppress(Exception):
        _close_under(Path(self.name).resolve())
    _original_cleanup(self)


tempfile.TemporaryDirectory.cleanup = _cleanup_closing_handles

# NOTE: deliberately no per-test autouse backstop. Closing everything after
# each test tore down stores belonging to a setUpClass-scoped fixture that
# later tests in the same class still used
# ("sqlite3.ProgrammingError: Cannot operate on a closed database" in
# tests/test_mobile.py). The cleanup hook above fires exactly when a
# directory is actually being removed, which is the only moment a held
# handle can block anything.
