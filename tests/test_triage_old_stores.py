"""Triage must survive every vintage of events store (2 Sep UI-pass find).

The Now screen polled needs_attention() every 3 seconds and crashed every
time — 'no such column: retracted' — whenever the store predated the
retraction feature: the bundled demo db, any pre-1.4 pilot db. Two causes,
both pinned here:

- the console's migration list (triage.ensure_columns) had drifted behind the
  engine's — it stopped at `note` while the queries had grown `retracted`;
- a store the migration cannot touch at all (the bundled demo ships
  read-only) still crashed, because the queries assumed the columns.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti import triage

# An events table as a pre-retraction store had it: no state machine, no
# provisional/retracted, no latency — review-era columns only.
_OLD_SCHEMA = """
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, iso TEXT, camera_id TEXT, rule TEXT, priority TEXT,
    confidence REAL, reason TEXT, track_id INTEGER, zone TEXT,
    object_label TEXT, evidence_dir TEXT, review TEXT, reviewed_at TEXT
);
"""


def _old_store(path: str) -> None:
    con = sqlite3.connect(path)
    con.executescript(_OLD_SCHEMA)
    con.execute("INSERT INTO events (ts, camera_id, rule, priority, review) "
                "VALUES (1000, 'front', 'theft_attempt', 'high', NULL)")
    con.execute("INSERT INTO events (ts, camera_id, rule, priority, review) "
                "VALUES (1001, 'back', 'fire_smoke', 'critical', 'true')")
    con.commit()
    con.close()


class OldStoreMigrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "events.db")
        _old_store(self.db)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ensure_columns_brings_an_old_store_current(self):
        con = sqlite3.connect(self.db)
        triage.ensure_columns(con)
        cols = {row[1] for row in con.execute("PRAGMA table_info(events)")}
        for col, _decl in triage.EVENT_COLUMNS:
            self.assertIn(col, cols)
        con.close()

    def test_needs_attention_works_after_migration(self):
        con = sqlite3.connect(self.db)
        triage.ensure_columns(con)
        out = triage.needs_attention(con)
        con.close()
        # The unreviewed event is the queue; the one reviewed 'true' projects
        # to resolved and stays out of it.
        self.assertEqual(out["now"]["rule"], "theft_attempt")
        self.assertEqual(out["waiting"], 1)


class ReadOnlyOldStoreTests(unittest.TestCase):
    """The bundled demo ships read-only: the migration CANNOT fix it, so the
    queries themselves must tolerate the missing columns."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "events.db")
        _old_store(self.db)
        os.chmod(self.db, 0o444)

    def tearDown(self):
        os.chmod(self.db, 0o644)
        self._tmp.cleanup()

    def _connect_ro(self):
        return sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)

    def test_ensure_columns_is_a_safe_noop(self):
        con = self._connect_ro()
        triage.ensure_columns(con)          # must not raise
        cols = {row[1] for row in con.execute("PRAGMA table_info(events)")}
        self.assertNotIn("retracted", cols)  # genuinely untouched
        con.close()

    def test_needs_attention_renders_the_queue_anyway(self):
        """THE regression: this exact call crashed every 3s on the Now screen."""
        con = self._connect_ro()
        out = triage.needs_attention(con)
        con.close()
        # The unreviewed event is the queue; the reviewed-'true' one is
        # resolved via the review projection. Nothing raises.
        self.assertIsNotNone(out["now"])
        self.assertEqual(out["now"]["rule"], "theft_attempt")
        self.assertEqual(out["waiting"], 1)
        self.assertEqual(out["held"], [])   # no state column -> nothing claimable


class FreshStoreTests(unittest.TestCase):
    def test_no_events_table_is_an_empty_queue_not_an_error(self):
        con = sqlite3.connect(":memory:")
        out = triage.needs_attention(con)
        con.close()
        self.assertEqual(out, {"now": None, "then": [], "waiting": 0, "held": []})


class MigrationListsCannotDriftTests(unittest.TestCase):
    def test_the_engine_sink_migrates_with_the_same_list(self):
        """The drift that caused this bug: two hand-maintained column lists.
        The sink must not keep a private one."""
        import inspect
        from cvti.serving import alert_sink
        src = inspect.getsource(alert_sink)
        self.assertIn("ensure_columns", src)
        self.assertNotIn('ADD COLUMN retracted', src)

    def test_every_schema_column_is_in_the_migration_list(self):
        """A column added to the live schema must be added to EVENT_COLUMNS in
        the same change, or every pre-existing store misses it forever."""
        from cvti.serving.alert_sink import _SCHEMA
        con = sqlite3.connect(":memory:")
        con.executescript(_SCHEMA)
        live = {row[1] for row in con.execute("PRAGMA table_info(events)")}
        con.close()
        migratable = {col for col, _ in triage.EVENT_COLUMNS}
        # The original core columns predate every store in the field; only
        # what came later must be migratable.
        core = {"id", "ts", "iso", "camera_id", "rule", "priority", "confidence",
                "reason", "track_id", "zone", "object_label", "evidence_dir"}
        missing = (live - core) - migratable
        self.assertFalse(missing,
                         f"live schema columns missing from EVENT_COLUMNS: {sorted(missing)}")


if __name__ == "__main__":
    unittest.main()
