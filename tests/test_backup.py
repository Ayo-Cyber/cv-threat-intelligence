"""Config backup and one-click restore (EP-08-T1).

The acceptance line that matters: restore tested END TO END — back up a
configured site, destroy it, restore onto 'fresh', prove the zones and the
customer's English rule came back byte-identical.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _backend_helper import signed_in

from cvti import backup


class BackupRestoreEndToEndTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._cwd = os.getcwd()
        os.chdir(self.root)                    # zones/rules live under configs/
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(lambda: os.chdir(self._cwd))
        (self.root / "site.json").write_text(json.dumps({"cameras": [
            {"id": "till", "source": "0"}], "name": "Shop", "notify": "telegram"}))
        self.be = signed_in("owner", site_path="site.json",
                            db_path=str(self.root / "events.db"), enable_demo=False)
        # a real configured site: zone + English rule
        self.be.add_zone("till", "entrance", [[0, 0], [10, 0], [10, 10]], 5)
        self.be.set_custom_rule("till", "Is anyone wearing a black hoodie?")

    def test_backup_wipe_restore_round_trips_everything(self):
        out = backup.backup_config("site.json", "backups")
        self.assertTrue(out["ok"])
        before = {p: Path(p).read_bytes() for p in
                  ("site.json", "configs/zones/till.json", "configs/rules/till.json")}
        # disk dies; fresh install
        for p in before:
            Path(p).unlink()
        res = backup.restore_config(out["path"], "site.json")
        self.assertTrue(res["ok"], res)
        for p, want in before.items():
            self.assertEqual(Path(p).read_bytes(), want, f"{p} did not round-trip")
        rules = json.loads(Path("configs/rules/till.json").read_text())["rules"]
        self.assertTrue(any(r["name"].startswith("custom_english") for r in rules))

    def test_restore_backs_up_the_current_config_first(self):
        out = backup.backup_config("site.json", "backups")
        Path("site.json").write_text('{"cameras": [], "name": "Changed"}')
        res = backup.restore_config(out["path"], "site.json")
        self.assertTrue(res["ok"])
        self.assertTrue(res["previous_config_backup"],
                        "restore destroyed the only copy of the current config")

    def test_accounts_and_audit_never_ride_in_a_backup(self):
        import zipfile
        out = backup.backup_config("site.json", "backups")
        names = zipfile.ZipFile(out["path"]).namelist()
        self.assertFalse([n for n in names if "auth" in n or "audit" in n],
                         "security material leaked into a NAS-bound backup")

    def test_versions_are_pruned(self):
        for _ in range(backup.KEEP_VERSIONS + 4):
            backup.backup_config("site.json", "backups")
        import time as _t
        left = list(Path("backups").glob("argus-config-*.zip"))
        self.assertLessEqual(len(left), backup.KEEP_VERSIONS)

    def test_a_hostile_zip_cannot_escape_the_config_dirs(self):
        import zipfile
        evil = self.root / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr(backup.MANIFEST, json.dumps(
                {"entries": ["../../outside.txt"], "format": 1}))
            zf.writestr("../../outside.txt", "pwned")
        res = backup.restore_config(evil, "site.json")
        self.assertFalse(res["ok"])
        self.assertIn("unsafe path", res["error"])


class DbIntegrityTest(unittest.TestCase):
    def test_a_corrupt_db_is_quarantined_not_masqueraded(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "events.db"
            db.write_bytes(b"this was never a database" * 100)
            out = backup.check_events_db(db)
            self.assertEqual(out["state"], "quarantined")
            self.assertFalse(db.exists(), "corrupt file left in place")
            self.assertTrue(Path(out["quarantined_to"]).exists(),
                            "corrupt file destroyed — recovery evidence lost")

    def test_a_healthy_db_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "events.db"
            con = sqlite3.connect(db)
            con.execute("CREATE TABLE t (x)")
            con.commit(); con.close()
            self.assertEqual(backup.check_events_db(db)["state"], "ok")

    def test_a_fresh_site_is_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(backup.check_events_db(Path(tmp) / "none.db")["state"], "fresh")


class EvidenceBackupTest(unittest.TestCase):
    def test_incremental_copy_and_db_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "events" / "ev1").mkdir(parents=True)
            (root / "events" / "ev1" / "f.jpg").write_bytes(b"\xff\xd8jpeg")
            con = sqlite3.connect(root / "events.db")
            con.execute("CREATE TABLE events (id INTEGER)"); con.commit(); con.close()
            dest = root / "nas"
            out1 = backup.backup_evidence(root / "events.db", dest)
            self.assertEqual(out1["copied"], 1)
            (root / "events" / "ev2").mkdir()
            out2 = backup.backup_evidence(root / "events.db", dest)
            self.assertEqual((out2["copied"], out2["skipped"]), (1, 1))
            self.assertTrue((dest / "events.db").exists())


if __name__ == "__main__":
    unittest.main()
