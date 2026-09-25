"""Close what the tests open, so Windows can delete it afterwards.

Nearly every backend test builds a ConsoleBackend inside a
TemporaryDirectory and lets it fall out of scope. That backend owns two
SQLite stores (auth.db, audit.db) whose connections nothing closed, and
Windows refuses to delete a file a process still holds open -- so the
directory cleanup raised PermissionError and the test failed, on Windows
only. POSIX unlinks open files happily, which is why the suite was green
everywhere else while 400+ assertions failed on Windows.

Rather than add a tearDown to twenty-six files, track the backends a test
creates and close them when it ends. The tracking lives here, in the test
tree, so the product keeps its ordinary constructor.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.app.console_backend import ConsoleBackend

_live: list = []
_original_init = ConsoleBackend.__init__


def _tracking_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    _live.append(self)


ConsoleBackend.__init__ = _tracking_init


@pytest.fixture(autouse=True)
def _close_backends_after_each_test():
    yield
    while _live:
        backend = _live.pop()
        with contextlib.suppress(Exception):   # teardown must never fail a test
            backend.close()
