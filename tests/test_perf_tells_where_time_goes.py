"""The instrumentation build (4 Sep): "slow" becomes a named stage.

Every pilot performance conversation so far was adjectives, because nothing
on the customer's machine measured anything. These pins hold the fix: each
stage reports its unit of work to one board, the engine writes stage
percentiles to perf_report.json beside gate_health.json, and the diagnostics
bundle ships that file — so a support zip says whether the time went to
decode, detection, the verification queue, model inference, or the English
scanner, on the machine that actually hurts.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.serving.perf import BOARD, PerfBoard, write_report


class BoardTests(unittest.TestCase):
    def test_percentiles_come_from_the_window(self):
        board = PerfBoard()
        for ms in range(1, 101):                      # 1..100ms
            board.observe("verify_infer", "cam1", float(ms))
        snap = board.snapshot()["verify_infer"]["cam1"]
        self.assertEqual(snap["count"], 100)
        self.assertAlmostEqual(snap["p50_ms"], 51.0, delta=2.0)
        self.assertAlmostEqual(snap["p95_ms"], 95.0, delta=2.0)
        self.assertEqual(snap["max_ms"], 100.0)

    def test_the_window_is_bounded(self):
        board = PerfBoard(window=8)
        for ms in range(100):
            board.observe("decode", "cam1", float(ms))
        self.assertEqual(board.snapshot()["decode"]["cam1"]["count"], 8)

    def test_stages_and_cameras_stay_separate(self):
        board = PerfBoard()
        board.observe("decode", "cam1", 5.0)
        board.observe("decode", "cam2", 50.0)
        board.observe("detect_batch", "engine", 500.0)
        snap = board.snapshot()
        self.assertEqual(snap["decode"]["cam1"]["last_ms"], 5.0)
        self.assertEqual(snap["decode"]["cam2"]["last_ms"], 50.0)
        self.assertEqual(snap["detect_batch"]["engine"]["last_ms"], 500.0)


class ReportTests(unittest.TestCase):
    def test_the_report_lands_beside_gate_health(self):
        BOARD.observe("decode", "report-test-cam", 12.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(tmp)
            self.assertIsNotNone(path)
            doc = json.loads(Path(path).read_text())
            self.assertIn("report-test-cam", doc["stages"]["decode"])
            self.assertIn("system", doc)              # CPU/RAM context rides along
            self.assertLess(abs(doc["generated_at"] - time.time()), 10)

    def test_a_bad_directory_never_raises(self):
        self.assertIsNone(write_report("/dev/null/not-a-dir"))


class WiringTests(unittest.TestCase):
    def test_the_decoder_reports_decode_time(self):
        from cvti.serving.streams import StreamDecoder
        d = StreamDecoder.__new__(StreamDecoder)
        d.camera_id = "wire-decode-cam"
        d.target_fps = 4.0
        d._min_period = 0.25
        d._fps = 30.0
        d._busy_ms = None
        d.sustainable_fps = 30.0
        d._ingest_limited = False
        d.stream_width = d.stream_height = 0
        d._observe_ingest(0.020, stride=1)
        snap = BOARD.snapshot()
        self.assertIn("wire-decode-cam", snap.get("decode", {}))

    def test_the_gate_worker_reports_wait_and_inference(self):
        from cvti.serving.alert_queue import AlertQueue, QueuedAlert
        from cvti.serving.gate_pool import GatePool

        class SlowGate:
            def verify(self, frames, candidate, scene, examples=None):
                time.sleep(0.01)
                from cvti.contracts import VerificationResult
                return VerificationResult(confirmed=True, confidence=0.9,
                                          reason="r", alert_priority="high",
                                          timestamp=time.time(), raw_response="t")

        class Cand:
            detector = "not-bypassed"
            title = "t"

        queue = AlertQueue()
        queue.add(QueuedAlert(camera_id="wire-gate-cam", rule_name="r",
                              priority="high", title="t",
                              timestamp=time.time() - 0.5,     # sat 500ms queued
                              payload={"candidate": Cand(), "frames": [],
                                       "scene": {}}))
        done = []
        pool = GatePool(queue, gate_factory=SlowGate, workers=1,
                        on_verdict=lambda a, r: done.append(1)).start()
        deadline = time.time() + 5
        while not done and time.time() < deadline:
            time.sleep(0.02)
        pool.stop()
        self.assertTrue(done, "gate worker never produced a verdict")
        snap = BOARD.snapshot()
        self.assertIn("wire-gate-cam", snap.get("verify_wait", {}))
        self.assertIn("wire-gate-cam", snap.get("verify_infer", {}))
        self.assertGreaterEqual(snap["verify_wait"]["wire-gate-cam"]["last_ms"], 400)

    def test_the_scanner_reports_scan_time(self):
        import numpy as np
        from unittest import mock
        from cvti.serving.custom_rules import CustomRuleScanner
        cam = {"id": "wire-scan-cam", "source": "x",
               "custom_threats": [{"name": "r", "description": "a thing"}]}
        scanner = CustomRuleScanner([cam], sink=None, model="gemma3:4b",
                                    frame_source=lambda cid: np.zeros((32, 32, 3),
                                                                      dtype=np.uint8))
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible",
                        return_value='{"threats": []}'):
            scanner._timed_scan(cam, {}, {})
        self.assertIn("wire-scan-cam", BOARD.snapshot().get("english_scan", {}))

    def test_the_engine_writes_the_report_with_health(self):
        import inspect
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.run_site)
        self.assertIn("_write_perf(output_dir)", src)


class BundleShipsTheNumbersTests(unittest.TestCase):
    def test_perf_report_rides_the_diagnostics_zip(self):
        from cvti.diagnostics import build_bundle
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "perf_report.json").write_text('{"stages": {}}')
            (out / "gate_health.json").write_text('{"status": "ok"}')
            bundle = build_bundle(out)
            names = zipfile.ZipFile(bundle).namelist()
        self.assertIn("perf_report.json", names)
        self.assertIn("gate_health.json", names)

    def test_a_bundle_without_a_report_still_builds(self):
        from cvti.diagnostics import build_bundle
        with tempfile.TemporaryDirectory() as tmp:
            names = zipfile.ZipFile(build_bundle(tmp)).namelist()
        self.assertIn("health.json", names)
        self.assertNotIn("perf_report.json", names)


if __name__ == "__main__":
    unittest.main()
