"""Cross-platform file locks with in-process thread exclusion."""

from __future__ import annotations

import errno
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    key = os.path.normcase(str(path.resolve()))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def file_lock(path: str | Path, *, blocking: bool = True,
              timeout: float | None = None) -> Iterator[bool]:
    """Lock *path* across threads and processes, yielding acquisition status."""
    if timeout is not None and (timeout < 0 or not blocking):
        raise ValueError("timeout requires blocking mode and must be non-negative")
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout is None else time.monotonic() + timeout
    thread_lock = _thread_lock(lock_path)
    if deadline is None:
        acquired_thread = thread_lock.acquire(blocking=blocking)
    else:
        acquired_thread = thread_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
    if not acquired_thread:
        yield False
        return
    handle = None
    acquired_process = False
    try:
        handle = lock_path.open("a+b")
        acquired_process = (_lock_windows(handle, blocking, deadline) if os.name == "nt"
                            else _lock_posix(handle, blocking, deadline))
        if not acquired_process:
            yield False
            return
        yield True
    finally:
        if acquired_process and handle is not None:
            _unlock(handle)
        if handle is not None:
            handle.close()
        thread_lock.release()


def _lock_posix(handle, blocking: bool, deadline: float | None) -> bool:
    import fcntl
    if blocking and deadline is None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return True
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            if not blocking or (deadline is not None and time.monotonic() >= deadline):
                return False
            assert deadline is not None
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def _lock_windows(handle, blocking: bool, deadline: float | None) -> bool:
    import msvcrt
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    mode = getattr(msvcrt, "LK_LOCK") if blocking and deadline is None else getattr(msvcrt, "LK_NBLCK")
    while True:
        try:
            handle.seek(0)
            getattr(msvcrt, "locking")(handle.fileno(), mode, 1)
            return True
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            if not blocking or (deadline is not None and time.monotonic() >= deadline):
                return False
            assert deadline is not None
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def _unlock(handle) -> None:
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        getattr(msvcrt, "locking")(handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
