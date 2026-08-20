"""Incident record + shift handover (EP-06-T2).

The incident record is the product's deliverable made tangible — what a manager
reviews and what goes to an insurer or the police. The handover is the third
clause of the epic's definition of done: the incoming shift knows what happened.
"""
import sqlite3
import tempfile
import time
import unittest
import zlib
from pathlib import Path

import cv2
import numpy as np

from _backend_helper import OWNER_PASSWORD, signed_in

from cvti.incident_pdf import _jpeg_size, _wrap, build_incident_pdf
from cvti.serving.alert_sink import AlertSink


def _jpeg(w=64, h=48):
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    return buf.tobytes()


def _pdf_text(data: bytes) -> str:
    """Decompress every Flate content stream so text assertions can see it."""
    out = []
    i = 0
    while True:
        i = data.find(b"stream\n", i)
        if i < 0:
            break
        j = data.find(b"\nendstream", i)
        chunk = data[i + 7:j]
        try:
            out.append(zlib.decompress(chunk).decode("latin-1", "replace"))
        except zlib.error:
            pass                      # a JPEG stream, not Flate
        i = j
    return "\n".join(out)


class PdfWriterTest(unittest.TestCase):
    def _event(self, **kw):
        base = {"id": 7, "rule": "baseline_fire_smoke", "camera_id": "warehouse",
                "iso": "2026-08-20T21:04:12", "priority": "critical",
                "confidence": 0.94, "reason": "Visible flame at the shelving.",
                "state": "resolved", "owner": "ayo", "outcome": "real",
                "resolved_at": time.time(), "note": "Fire service called."}
        base.update(kw)
        return base

    def test_a_valid_pdf_with_the_frames_embedded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_incident_pdf(self._event(),
                                      [("frame_000.jpg", _jpeg()), ("frame_001.jpg", _jpeg())],
                                      Path(tmp) / "r.pdf")
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"%PDF-1.4"))
            self.assertTrue(data.rstrip().endswith(b"%%EOF"))
            self.assertEqual(data.count(b"/DCTDecode"), 2)

    def test_the_record_carries_reasoning_responder_and_conclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_incident_pdf(self._event(), [], Path(tmp) / "r.pdf")
            text = _pdf_text(path.read_bytes())
            for needed in ("Visible flame", "ayo", "REAL incident",
                           "Fire service called", "warehouse", "CRITICAL"):
                self.assertIn(needed, text, f"record is missing {needed!r}")

    def test_an_open_incident_says_so_instead_of_faking_a_conclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_incident_pdf(
                self._event(state="acknowledged", outcome=None, resolved_at=None,
                            note=None),
                [], Path(tmp) / "r.pdf")
            text = _pdf_text(path.read_bytes())
            self.assertIn("OPEN", text)
            self.assertIn("not yet concluded", text)

    def test_an_unverified_alert_is_marked_in_the_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_incident_pdf(self._event(unverified=1), [], Path(tmp) / "r.pdf")
            self.assertIn("UNVERIFIED", _pdf_text(path.read_bytes()))

    def test_missing_frames_are_stated_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_incident_pdf(self._event(), [], Path(tmp) / "r.pdf")
            self.assertIn("no evidence frames on disk", _pdf_text(path.read_bytes()))

    def test_many_frames_paginate_rather_than_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = [(f"frame_{i:03d}.jpg", _jpeg(320, 240)) for i in range(8)]
            path = build_incident_pdf(self._event(), frames, Path(tmp) / "r.pdf")
            data = path.read_bytes()
            self.assertGreaterEqual(data.count(b"/Type /Page>") +
                                    data.count(b"/Type /Page "), 2)

    def test_jpeg_dimensions_are_read_from_the_bytes(self):
        self.assertEqual(_jpeg_size(_jpeg(64, 48)), (64, 48))

    def test_wrap_never_loses_words(self):
        text = "word " * 200
        self.assertEqual(" ".join(_wrap(text)).split(), text.split())


class BackendExportTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "site.json").write_text('{"cameras": []}')
        AlertSink(root, save_evidence=False, routing_path=None).close()
        self.be = signed_in("owner", site_path=str(root / "site.json"),
                            db_path=str(root / "events.db"), enable_demo=False)
        ev_dir = root / "events" / "20260820_warehouse_fire"
        ev_dir.mkdir(parents=True)
        (ev_dir / "frame_000.jpg").write_bytes(_jpeg())
        con = sqlite3.connect(self.be.db_path)
        cur = con.execute(
            "INSERT INTO events (ts, iso, camera_id, rule, priority, confidence, "
            "reason, evidence_dir) VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), "2026-08-20T21:04:12", "warehouse", "baseline_fire_smoke",
             "critical", 0.94, "Visible flame.", str(ev_dir)))
        con.commit()
        self.event = cur.lastrowid
        con.close()

    def tearDown(self):
        self._tmp.cleanup()

    def test_export_produces_a_pdf_and_audits_it(self):
        result = self.be.export_incident_pdf(self.event)
        self.assertTrue(result["ok"], result)
        data = Path(result["path"]).read_bytes()
        self.assertTrue(data.startswith(b"%PDF"))
        self.assertEqual(data.count(b"/DCTDecode"), 1)     # the frame came along
        entry = self.be.audit_entries()[0]
        self.assertEqual(entry["action"], "evidence_export")
        self.assertEqual(entry["detail"]["format"], "pdf")

    def test_export_of_a_missing_incident_is_an_answer_not_a_crash(self):
        self.assertFalse(self.be.export_incident_pdf(99999)["ok"])


class HandoverTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "site.json").write_text('{"cameras": []}')
        AlertSink(root, save_evidence=False, routing_path=None).close()
        self.be = signed_in("owner", site_path=str(root / "site.json"),
                            db_path=str(root / "events.db"), enable_demo=False)

    def tearDown(self):
        self._tmp.cleanup()

    def _insert(self, *, hours_ago=1.0, priority="high", rule="theft",
                state=None, owner=None, outcome=None, note=None,
                resolved_hours_ago=None):
        con = sqlite3.connect(self.be.db_path)
        cur = con.execute(
            "INSERT INTO events (ts, iso, camera_id, rule, priority, state, owner, "
            "outcome, note, resolved_at, review) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time() - hours_ago * 3600, "iso", "cam1", rule, priority, state,
             owner, outcome, note,
             (time.time() - resolved_hours_ago * 3600) if resolved_hours_ago is not None else None,
             {"resolved": "true"}.get(state)))
        con.commit()
        event = cur.lastrowid
        con.close()
        return event

    def test_the_incoming_shift_sees_fired_resolved_and_open(self):
        self._insert(hours_ago=2, state="resolved", owner="sam", outcome="real",
                     note="police attended", resolved_hours_ago=1.5)
        self._insert(hours_ago=1, state="new")
        self._insert(hours_ago=3, state="acknowledged", owner="ayo")
        h = self.be.handover(8)
        self.assertEqual(h["counts"]["fired"], 3)
        self.assertEqual(h["counts"]["resolved"], 1)
        self.assertEqual(h["counts"]["open"], 2)
        self.assertEqual(h["resolved"][0]["owner"], "sam")
        self.assertEqual(h["resolved"][0]["note"], "police attended")

    def test_open_items_from_before_the_window_are_carried_over_and_flagged(self):
        # An incident from three shifts ago that nobody concluded is precisely
        # what must not be forgotten.
        old = self._insert(hours_ago=30, state="new")
        recent = self._insert(hours_ago=1, state="new")
        h = self.be.handover(8)
        by_id = {o["id"]: o for o in h["open"]}
        self.assertTrue(by_id[old]["carried_over"])
        self.assertFalse(by_id[recent]["carried_over"])
        self.assertGreater(by_id[old]["age_h"], 24)

    def test_resolved_outside_the_window_is_not_re_reported(self):
        self._insert(hours_ago=30, state="resolved", owner="sam", outcome="real",
                     resolved_hours_ago=29)
        h = self.be.handover(8)
        self.assertEqual(h["counts"]["resolved"], 0)
        self.assertEqual(h["counts"]["open"], 0)     # and it is not open either

    def test_unclaimed_open_items_are_distinguishable_from_claimed_ones(self):
        self._insert(hours_ago=1, state="acknowledged", owner="ayo")
        self._insert(hours_ago=1, state="new")
        h = self.be.handover(8)
        owners = sorted((o.get("owner") or "") for o in h["open"])
        self.assertEqual(owners, ["", "ayo"])

    def test_a_clean_board_is_an_answer(self):
        h = self.be.handover(8)
        self.assertEqual(h["counts"], {"fired": 0, "resolved": 0, "open": 0,
                                       "outcomes": {}})


if __name__ == "__main__":
    unittest.main()
