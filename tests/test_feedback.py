"""Tests for the feedback / reinforcement-training subsystem."""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.feedback.calibration import ACTION_DEMOTE, ACTION_TRUSTED, Calibration
from cvti.feedback.dataset import export_dataset
from cvti.feedback.manager import FeedbackManager
from cvti.feedback.registry import ModelRegistry
from cvti.feedback.store import FeedbackStore


def _make_db(d: Path, rows):
    """rows: list of (camera, rule, review). Creates events.db + evidence frames."""
    db = d / "events.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts REAL, iso TEXT, camera_id TEXT, "
                "rule TEXT, priority TEXT, confidence REAL, reason TEXT, track_id INT, zone TEXT, "
                "object_label TEXT, evidence_dir TEXT, review TEXT, reviewed_at TEXT, latency_s REAL)")
    for i, (cam, rule, review) in enumerate(rows, start=1):
        evd = d / "events" / str(i)
        evd.mkdir(parents=True)
        for f in range(3):
            ok, b = cv2.imencode(".jpg", np.full((40, 60, 3), 30, np.uint8))
            (evd / f"frame_{f:03d}.jpg").write_bytes(b.tobytes())
        con.execute("INSERT INTO events (id,ts,camera_id,rule,priority,confidence,reason,evidence_dir,review) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (i, float(i), cam, rule, "high", 0.8, f"reason {i}", str(evd), review))
    con.commit()
    con.close()
    return db


# 4 true + 1 false on Forecourt/theft (trusted); 5 false + 1 true on Live/crowd (demote)
ROWS = ([("Forecourt", "theft", "true")] * 4 + [("Forecourt", "theft", "false")]
        + [("Live", "crowd", "false")] * 5 + [("Live", "crowd", "true")])


class StoreCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.db = _make_db(self.d, ROWS)

    def test_store_counts(self):
        c = FeedbackStore(self.db).counts()
        self.assertEqual(c["reviewed"], 11)
        self.assertEqual(c["positive"], 5)
        self.assertEqual(c["negative"], 6)

    def test_calibration_demotes_noisy_and_trusts_good(self):
        cal = Calibration.from_store(FeedbackStore(self.db))
        self.assertTrue(cal.demoted("Live", "crowd"))
        self.assertFalse(cal.demoted("Forecourt", "theft"))
        self.assertEqual(cal.rules["Forecourt::theft"].action(), ACTION_TRUSTED)
        self.assertEqual(cal.rules["Live::crowd"].action(), ACTION_DEMOTE)
        self.assertEqual(cal.demoted_keys(), ["Live::crowd"])

    def test_calibration_save_load_roundtrip(self):
        cal = Calibration.from_store(FeedbackStore(self.db))
        p = self.d / "calibration.json"
        cal.save(p)
        loaded = Calibration.load(p)
        self.assertTrue(loaded.demoted("Live", "crowd"))
        self.assertFalse(loaded.demoted("Forecourt", "theft"))

    def test_missing_calibration_never_demotes(self):
        self.assertFalse(Calibration().demoted("any", "thing"))
        self.assertFalse(Calibration.load(self.d / "nope.json").demoted("any", "thing"))

    def test_examples_are_decisive_only(self):
        ex = FeedbackStore(self.db).examples("Live", "crowd", k=10)
        self.assertTrue(all(e.review in ("true", "false") for e in ex))
        self.assertTrue(len(ex) >= 1)


class DatasetRegistryManagerTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.db = _make_db(self.d, ROWS)

    def test_dataset_export_splits_by_label(self):
        res = export_dataset(FeedbackStore(self.db), self.d / "ds")
        self.assertEqual(res.per_class.get("threat"), 5)
        self.assertEqual(res.per_class.get("normal"), 6)
        self.assertTrue((self.d / "ds" / "manifest.json").exists())
        self.assertEqual(res.total, 11)

    def test_registry_register_and_rollback(self):
        reg = ModelRegistry(str(self.d / "registry.json"))
        reg.register("model_v1", created=1.0)
        reg.register("model_v2", created=2.0)
        self.assertEqual(reg.active_path(), "model_v2")
        reg.rollback()
        self.assertEqual(reg.active_path(), "model_v1")
        self.assertIsNone(ModelRegistry(str(self.d / "empty.json")).rollback())

    def test_manager_status_and_calibrate(self):
        mgr = FeedbackManager(str(self.db), dataset_dir=str(self.d / "ds"),
                              registry_path=str(self.d / "reg.json"))
        st = mgr.status()
        self.assertEqual(st["labels"]["reviewed"], 11)
        self.assertIn("Live::crowd", st["demoted"])
        cal = mgr.calibrate()
        self.assertTrue((self.d / "calibration.json").exists())
        self.assertIn("Live::crowd", cal["demoted"])

    def test_manager_retrain_dry_run(self):
        mgr = FeedbackManager(str(self.db), dataset_dir=str(self.d / "ds"))
        r = mgr.retrain(run=False)
        self.assertFalse(r["ran"])
        self.assertIn("video_finetune", r["command"])
        self.assertEqual(r["dataset"]["total"], 11)


if __name__ == "__main__":
    unittest.main()
