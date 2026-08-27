"""Retention, purge, and legal hold (EP-02-T2).

Storage limitation is not optional under GDPR/NDPR, and an edge box with no
purge fills its disk and stops recording evidence exactly when it matters.

But blind time-based deletion would destroy the exact records a customer needs.
These tests exist mostly to pin down what must NOT be deleted.
"""
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from cvti.serving.retention import RetentionManager, RetentionPolicy, disk_status

DAY = 86400


class _Site:
    """An output directory shaped like a real one: rows plus evidence on disk."""

    def __init__(self, tmp: str, retention_days: int = 30):
        self.root = Path(tmp)
        self.events_dir = self.root / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "events.db"
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, "
                    "iso TEXT, camera_id TEXT, rule TEXT, evidence_dir TEXT, review TEXT, "
                    "legal_hold INTEGER DEFAULT 0, state TEXT, owner TEXT)")
        con.commit()
        con.close()
        self.mgr = RetentionManager(self.root, RetentionPolicy(days=retention_days))

    def add(self, *, age_days: float, review=None, legal_hold=0, with_files=True,
            state=None, owner=None) -> int:
        ts = time.time() - age_days * DAY
        name = f"{int(ts)}_{review}_{legal_hold}_{age_days}"
        ev = self.events_dir / name
        if with_files:
            ev.mkdir(parents=True, exist_ok=True)
            (ev / "frame_000.jpg").write_bytes(b"\xff\xd8fake-jpeg")
            (ev / "clip.mp4").write_bytes(b"fake-mp4")
        con = sqlite3.connect(self.db)
        cur = con.execute(
            "INSERT INTO events (ts, iso, camera_id, rule, evidence_dir, review, "
            "legal_hold, state, owner) VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, "iso", "cam1", "theft", str(ev), review, legal_hold, state, owner))
        con.commit()
        event_id = cur.lastrowid
        con.close()
        return event_id

    def ids(self) -> set:
        con = sqlite3.connect(self.db)
        out = {r[0] for r in con.execute("SELECT id FROM events")}
        con.close()
        return out

    def dirs(self) -> set:
        return {p.name for p in self.events_dir.iterdir() if p.is_dir()}


class PurgesWhatItShouldTest(unittest.TestCase):
    def test_expired_and_settled_events_are_deleted_with_their_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            old = site.add(age_days=40, review="true")
            result = site.mgr.purge()
            self.assertEqual(result["deleted"], 1)
            self.assertNotIn(old, site.ids())
            self.assertEqual(site.dirs(), set(), "evidence files survived the purge")

    def test_recent_events_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            recent = site.add(age_days=5, review="true")
            site.mgr.purge()
            self.assertIn(recent, site.ids())
            self.assertEqual(len(site.dirs()), 1)

    def test_the_boundary_is_the_configured_number_of_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp, retention_days=7)
            keep = site.add(age_days=6, review="false")
            drop = site.add(age_days=8, review="false")
            site.mgr.purge()
            self.assertEqual(site.ids(), {keep})
            self.assertNotIn(drop, site.ids())

    def test_dry_run_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            site.add(age_days=40, review="true")
            result = site.mgr.purge(dry_run=True)
            self.assertEqual(result["deleted"], 0)
            self.assertEqual(len(result["would_delete"]), 1)
            self.assertEqual(len(site.ids()), 1)

    def test_orphans_are_swept_even_when_nothing_has_expired(self):
        # An orphan exists independently of expiry. A young site can still be
        # holding evidence no record points to.
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            site.add(age_days=1, review=None)              # nothing expired
            stray = site.events_dir / "orphan_dir"
            stray.mkdir()
            (stray / "frame_000.jpg").write_bytes(b"unfindable personal data")
            result = site.mgr.purge()
            self.assertEqual(result["examined"], 0)
            self.assertEqual(result["orphans_removed"], 1)
            self.assertFalse(stray.exists())

    def test_purge_always_reports_the_same_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            keys = {"examined", "deleted", "failed", "held", "orphans_removed", "would_delete"}
            self.assertTrue(keys <= set(site.mgr.purge()))          # nothing to do
            self.assertTrue(keys <= set(site.mgr.purge(dry_run=True)))
            site.add(age_days=40, review="true")
            self.assertTrue(keys <= set(site.mgr.purge()))          # something to do

    def test_a_disabled_policy_purges_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            site.mgr.policy.enabled = False
            site.add(age_days=400, review="true")
            self.assertIn("skipped", site.mgr.purge())
            self.assertEqual(len(site.ids()), 1)


class NeverDeletesWhatMattersTest(unittest.TestCase):
    """The reason this is not `find -mtime +30 -delete`."""

    def test_an_explicit_legal_hold_survives_its_own_expiry(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            held = site.add(age_days=400, review="true", legal_hold=1)
            site.mgr.purge()
            self.assertIn(held, site.ids())
            self.assertEqual(len(site.dirs()), 1, "evidence on legal hold was deleted")

    def test_an_unreviewed_incident_survives_its_own_expiry(self):
        # Nobody has decided what this was yet. Deleting it destroys the record
        # while the question is still open.
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            open_case = site.add(age_days=400, review=None)
            site.mgr.purge()
            self.assertIn(open_case, site.ids())

    def test_held_items_are_reported_so_retention_is_not_a_surprise(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            site.add(age_days=400, review="true", legal_hold=1)
            site.add(age_days=400, review=None)
            site.add(age_days=400, review="true")
            result = site.mgr.purge()
            self.assertEqual(result["deleted"], 1)
            self.assertEqual(result["held"], {"legal_hold": 1, "unreviewed": 1})

    def test_a_claimed_but_unresolved_incident_survives_its_own_expiry(self):
        # Acknowledging is the start of the work, not the end of it. A guard
        # claimed this and never concluded — deleting it on a timer destroys
        # the record mid-investigation. (EP-06-T2: open incidents are held.)
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            claimed = site.add(age_days=400, review="ack",
                               state="acknowledged", owner="sam")
            done = site.add(age_days=400, review="true", state="resolved")
            site.mgr.purge()
            self.assertIn(claimed, site.ids(), "an open investigation was purged")
            self.assertNotIn(done, site.ids())

    def test_releasing_a_hold_makes_it_purgeable(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            held = site.add(age_days=400, review="true", legal_hold=1)
            site.mgr.purge()
            self.assertIn(held, site.ids())
            con = sqlite3.connect(site.db)
            con.execute("UPDATE events SET legal_hold = 0 WHERE id = ?", (held,))
            con.commit()
            con.close()
            site.mgr.purge()
            self.assertNotIn(held, site.ids())


class NoOrphansTest(unittest.TestCase):
    """Files without a row are personal data nobody can find, account for, or erase."""

    def test_purge_leaves_no_files_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            for _ in range(5):
                site.add(age_days=40, review="true")
            site.mgr.purge()
            self.assertEqual(site.ids(), set())
            self.assertEqual(site.dirs(), set())

    def test_a_directory_with_no_row_is_swept(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            stray = site.events_dir / "20260101_000000_cam9_theft"
            stray.mkdir(parents=True)
            (stray / "frame_000.jpg").write_bytes(b"orphaned personal data")
            self.assertEqual(site.mgr.sweep_orphans(), 1)
            self.assertFalse(stray.exists())

    def test_sweeping_never_touches_a_live_events_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            site.add(age_days=1, review=None)
            self.assertEqual(site.mgr.sweep_orphans(), 0)
            self.assertEqual(len(site.dirs()), 1)

    def test_a_row_whose_files_are_already_gone_still_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            gone = site.add(age_days=40, review="true", with_files=False)
            site.mgr.purge()
            self.assertNotIn(gone, site.ids())

    def test_it_refuses_to_delete_outside_its_own_events_directory(self):
        # A bad or hostile evidence_dir must not turn retention into rm -rf.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as elsewhere:
            site = _Site(tmp)
            victim = Path(elsewhere) / "not-ours"
            victim.mkdir()
            (victim / "important.txt").write_text("do not delete me")
            con = sqlite3.connect(site.db)
            con.execute("INSERT INTO events (ts, iso, camera_id, rule, evidence_dir, review) "
                        "VALUES (?,?,?,?,?,?)",
                        (time.time() - 400 * DAY, "iso", "cam1", "theft", str(victim), "true"))
            con.commit()
            con.close()
            result = site.mgr.purge()
            self.assertTrue(victim.exists(), "retention deleted a directory outside its own tree")
            self.assertEqual((victim / "important.txt").read_text(), "do not delete me")
            self.assertEqual(result["failed"], 1)


class DiskTest(unittest.TestCase):
    def test_disk_status_reports_a_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = disk_status(tmp)
            self.assertTrue(status["available"])
            self.assertIn(status["level"], ("ok", "warning", "critical"))

    def test_thresholds_drive_the_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            hair_trigger = RetentionPolicy(disk_warn_pct=0.0, disk_critical_pct=0.0)
            self.assertEqual(disk_status(tmp, hair_trigger)["level"], "critical")
            never = RetentionPolicy(disk_warn_pct=100.1, disk_critical_pct=100.2)
            self.assertEqual(disk_status(tmp, never)["level"], "ok")

    def test_emergency_purge_does_nothing_when_disk_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            site.mgr.policy.disk_critical_pct = 100.1
            self.assertFalse(site.mgr.emergency_purge()["triggered"])

    def test_emergency_purge_takes_the_oldest_first_and_still_respects_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            site.mgr.policy.disk_critical_pct = 0.0     # force the critical path
            held = site.add(age_days=500, review=None)          # open case, oldest
            site.add(age_days=400, review="true")
            site.add(age_days=1, review="true")
            result = site.mgr.emergency_purge()
            self.assertTrue(result["triggered"])
            self.assertIn(held, site.ids(), "an open case was destroyed to free space")
            self.assertEqual(len(site.ids()), 1)

    def test_a_disk_that_cannot_be_freed_says_so_rather_than_looping(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = _Site(tmp)
            site.mgr.policy.disk_critical_pct = 0.0
            site.add(age_days=500, review=None)         # everything is on hold
            result = site.mgr.emergency_purge()
            self.assertEqual(result["deleted"], 0)      # terminates instead of spinning


class PolicyTest(unittest.TestCase):
    def test_defaults_to_thirty_days(self):
        self.assertEqual(RetentionPolicy().days, 30)

    def test_reads_the_site_config(self):
        policy = RetentionPolicy.from_site({"retention_days": 7, "disk_warn_pct": 70})
        self.assertEqual(policy.days, 7)
        self.assertEqual(policy.disk_warn_pct, 70.0)

    def test_a_nonsense_setting_falls_back_rather_than_disabling_retention(self):
        policy = RetentionPolicy.from_site({"retention_days": "soon"})
        self.assertEqual(policy.days, 30)

    def test_retention_can_never_be_configured_to_zero_days(self):
        # 0 would delete everything on the next tick.
        self.assertGreaterEqual(RetentionPolicy.from_site({"retention_days": 0}).days, 1)


class MissingDatabaseTest(unittest.TestCase):
    def test_purge_on_a_bare_directory_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = RetentionManager(tmp, RetentionPolicy())
            self.assertEqual(mgr.purge()["examined"], 0)


if __name__ == "__main__":
    unittest.main()


class EvidencePathResolutionTest(unittest.TestCase):
    """Storage limitation silently did nothing for the life of an install
    (27 Aug). Evidence paths were stored relative to the WORKING DIRECTORY;
    retention resolved them against the OUTPUT ROOT; the containment guard
    then refused every one — so no evidence was ever purged and the disk
    filled to 94%. GDPR-relevant: the UI reported a retention period the
    purger could not enforce."""

    def _site(self, tmp):
        root = Path(tmp) / "site"
        (root / "events").mkdir(parents=True)
        con = sqlite3.connect(root / "events.db")
        con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts REAL, iso TEXT, "
                    "camera_id TEXT, rule TEXT, evidence_dir TEXT, review TEXT, "
                    "legal_hold INTEGER DEFAULT 0, state TEXT, outcome TEXT)")
        con.commit()
        return root, con

    def test_a_relative_row_is_purged_not_refused_forever(self):
        import os
        from cvti.serving.retention import RetentionManager, RetentionPolicy
        with tempfile.TemporaryDirectory() as tmp:
            root, con = self._site(tmp)
            ev = root / "events" / "20260101_000000_cam_rule"
            ev.mkdir()
            (ev / "frame.jpg").write_bytes(b"\xff\xd8x")
            cwd = os.getcwd()
            os.chdir(tmp)                       # rows were written relative to HERE
            try:
                rel = ev.relative_to(Path(tmp))
                con.execute("INSERT INTO events (id, ts, evidence_dir, state, review) "
                            "VALUES (1, 0, ?, 'resolved', 'true')", (str(rel),))
                con.commit()
                con.row_factory = sqlite3.Row
                row = con.execute("SELECT * FROM events WHERE id=1").fetchone()
                m = RetentionManager(root, RetentionPolicy(days=1))
                self.assertTrue(m._delete_event(con, row),
                                "retention refused a legitimate relative row — "
                                "nothing would ever be purged")
                con.commit()
                self.assertFalse(ev.exists(), "evidence survived its own purge")
            finally:
                os.chdir(cwd)
                con.close()

    def test_it_still_refuses_a_path_outside_its_events_tree(self):
        from cvti.serving.retention import RetentionManager, RetentionPolicy
        with tempfile.TemporaryDirectory() as tmp:
            root, con = self._site(tmp)
            outside = Path(tmp) / "precious"
            outside.mkdir()
            (outside / "source.py").write_text("keep me")
            con.execute("INSERT INTO events (id, ts, evidence_dir, state, review) "
                        "VALUES (1, 0, ?, 'resolved', 'true')", (str(outside),))
            con.commit()
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT * FROM events WHERE id=1").fetchone()
            m = RetentionManager(root, RetentionPolicy(days=1))
            self.assertFalse(m._delete_event(con, row))
            con.close()
            self.assertTrue((outside / "source.py").exists(),
                            "retention deleted outside its own tree")

    def test_the_sink_writes_absolute_evidence_paths(self):
        import inspect
        from cvti.serving.alert_sink import AlertSink
        src = inspect.getsource(AlertSink)
        self.assertIn('str(ev_dir.resolve())', src,
                      "a relative evidence path is unmatchable by retention")
