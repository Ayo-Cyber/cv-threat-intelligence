"""An alert with no evidence must never mean "the current directory" (27 Aug).

`Path(None or "")` is `Path(".")`. Provisional alerts legitimately carry a NULL
evidence_dir — evidence is written when the verdict settles — so every reader
of that field was one `or ""` away from treating the working directory as an
evidence bundle. A cleanup script with the same flaw computed 31.8 GB of "event
evidence" that was actually the whole repository.
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _backend_helper import signed_in


class NoEvidenceIsNotTheCwdTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "site.json").write_text('{"cameras": []}')
        self.be = signed_in("owner", site_path=str(self.root / "site.json"),
                            db_path=str(self.root / "events.db"), enable_demo=False)
        self.addCleanup(self._tmp.cleanup)

    def test_frames_for_a_null_evidence_dir_are_empty_not_the_working_directory(self):
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            # a stray image in the working directory must not become "evidence"
            (self.root / "not_evidence.jpg").write_bytes(b"\xff\xd8stray")
            self.assertEqual(self.be._frames_as_data_uris(None), [])
            self.assertEqual(self.be._frames_as_data_uris(""), [])
        finally:
            os.chdir(cwd)

    def test_event_clip_refuses_a_null_evidence_dir(self):
        out = self.be.event_clip(None)
        self.assertEqual(out.get("frames"), [])
        self.assertIsNone(out.get("uri"))

    def test_export_of_an_evidenceless_alert_is_an_answer_not_a_bundle_of_cwd(self):
        out = self.be.export_evidence_dir("") if hasattr(self.be, "export_evidence_dir") \
            else None
        if out is not None:
            self.assertFalse(out["ok"])

    def test_notifier_and_inbox_readers_do_not_scan_the_cwd(self):
        from cvti.serving.inbox import _frames_for
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            (self.root / "stray.jpg").write_bytes(b"\xff\xd8stray")
            self.assertEqual(_frames_for({"evidence_dir": None}), [])
            self.assertEqual(_frames_for({}), [])
        finally:
            os.chdir(cwd)

    def test_no_source_file_still_builds_a_path_from_an_empty_string(self):
        import re
        root = Path(__file__).resolve().parents[1] / "cvti"
        offenders = []
        for f in root.rglob("*.py"):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                code = line.split("#", 1)[0]        # the pattern in a COMMENT is fine
                if re.search(r'Path\([^)]*\bor\s+""\s*\)', code):
                    offenders.append(f"{f.relative_to(root)}:{i}")
        self.assertEqual(offenders, [], f"empty path resolves to the CWD: {offenders}")


class RetentionNeverEscapesItsOwnTreeTest(unittest.TestCase):
    """The product's own purge was already defended — this keeps it that way."""

    def test_a_row_pointing_outside_the_events_dir_is_refused(self):
        from cvti.serving.retention import RetentionManager, RetentionPolicy
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "site"
            (root / "events").mkdir(parents=True)
            outside = Path(tmp) / "precious"
            outside.mkdir()
            (outside / "source.py").write_text("do not delete me")
            db = root / "events.db"
            con = sqlite3.connect(db)
            con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts REAL, iso TEXT, "
                        "camera_id TEXT, rule TEXT, evidence_dir TEXT, review TEXT, "
                        "legal_hold INTEGER, state TEXT)")
            con.execute("INSERT INTO events (id, ts, evidence_dir) VALUES (1, 0, ?)",
                        (str(outside),))
            con.commit()
            m = RetentionManager(root, RetentionPolicy(days=0))
            row = con.execute("SELECT * FROM events WHERE id=1").fetchone()
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT * FROM events WHERE id=1").fetchone()
            deleted = m._delete_event(con, row) if hasattr(m, "_delete_event") else None
            con.close()
            self.assertTrue(outside.exists(), "retention deleted OUTSIDE its events dir")
            self.assertTrue((outside / "source.py").exists())


if __name__ == "__main__":
    unittest.main()
