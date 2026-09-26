"""The audit log keeps no file handle open, and its chain survives a race.

Both properties were broken by the same line: one sqlite3 connection, opened
in __init__, kept for the life of the process, shared across threads with
check_same_thread=False.

  * Windows refuses to move or delete a file a process holds open. audit.db
    stayed locked for as long as Argus ran -- which an in-place upgrade or an
    uninstall has to do -- and made 167 tests fail on Windows and nowhere
    else, because POSIX unlinks open files happily and nobody noticed.
  * record() read the previous hash and appended the next entry with no
    transaction around the pair, so two threads could read the same previous
    hash and fork the chain -- in the one structure whose whole job is to
    prove nothing was tampered with.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.security.audit import AuditLog


class TheFileIsNotHeldOpen(unittest.TestCase):
    def test_the_database_can_be_deleted_while_the_log_is_alive(self):
        """What Windows enforces and POSIX does not: no lingering handle."""
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "audit.db"
            log = AuditLog(db)
            log.record("ayo", "login")
            os.remove(db)                      # PermissionError on Windows if held
            self.assertFalse(db.exists())

    def test_a_temporary_directory_cleans_up_afterwards(self):
        """The exact shape of the 167 Windows failures."""
        d = tempfile.TemporaryDirectory()
        log = AuditLog(Path(d.name) / "audit.db")
        log.record("ayo", "config_change", "camera:cam1")
        d.cleanup()                            # must not raise
        self.assertFalse(Path(d.name).exists())

    def test_close_is_still_callable_for_existing_callers(self):
        with tempfile.TemporaryDirectory() as d:
            log = AuditLog(Path(d) / "audit.db")
            log.close()
            log.record("ayo", "login")         # and the log still works after
            self.assertEqual(len(log.entries()), 1)


class TheChainSurvivesConcurrentWriters(unittest.TestCase):
    def test_parallel_records_produce_one_unbroken_chain(self):
        with tempfile.TemporaryDirectory() as d:
            log = AuditLog(Path(d) / "audit.db")
            errors: list = []

            def writer(n: int) -> None:
                try:
                    for i in range(10):
                        log.record(f"user{n}", "login", f"t{i}")
                except Exception as exc:       # noqa: BLE001 - surfaced below
                    errors.append(exc)

            threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"concurrent writers raised: {errors}")
            self.assertEqual(len(log.entries(limit=10 ** 6)), 40)
            verdict = log.verify()
            self.assertTrue(verdict["ok"], verdict)
            self.assertEqual(verdict["checked"], 40)


if __name__ == "__main__":
    unittest.main()
