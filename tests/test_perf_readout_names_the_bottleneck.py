"""The readout names the top-cost stage — and not the loudest one (7 Sep).

The 4 Sep instrumentation build made every stage report its latency. That is
necessary and not sufficient: the stages measure different units at different
rates, so the largest p95 in the file is routinely NOT the bottleneck. On a
real box (runs/feeds/stage) the English scanner's 4.8s p95 is forty times
detection's and would top any raw ranking — but it runs once every 12s, so it
costs 0.4 of a thread against detection's 0.96. Sorting by p95 would send a
week of engineering at the wrong stage.

These pins hold the fix: each series carries the wall-clock span it covers, so
a comparable share-of-a-thread can be computed; waiting is never counted as
work; a stage that never ran is named rather than omitted; and a report from a
build older than this one degrades to latency-only instead of inventing a rate.
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

from cvti.serving.perf import BOARD, PerfBoard, write_report  # noqa: E402
from tools.perf_readout import (  # noqa: E402
    load_report, rank_cost, render, saturation, ingest_warnings,
    missing_stages, verdict,
)


def _series(board: PerfBoard, stage: str, key: str, ms: float, n: int,
            period_s: float, units: int = 1) -> None:
    """n observations of `ms` spaced `period_s` apart, timestamps forced.

    Real spacing would make the test sleep for minutes; the board stores the
    clock reading alongside each sample, so the test writes that history
    directly and keeps the arithmetic under scrutiny instead of the sleep.
    """
    now = time.time()
    with board._lock:                       # noqa: SLF001 - fabricating history
        from collections import deque
        board._series[(stage, key)] = deque(   # noqa: SLF001
            [(now - (n - 1 - i) * period_s, ms, units) for i in range(n)],
            maxlen=board._window)             # noqa: SLF001


class SpanAndRateTests(unittest.TestCase):
    def test_a_series_reports_the_window_it_actually_covers(self):
        board = PerfBoard()
        _series(board, "decode", "cam1", 10.0, n=11, period_s=0.5)   # 5s of history
        doc = board.snapshot()["decode"]["cam1"]
        self.assertAlmostEqual(doc["span_s"], 5.0, delta=0.1)
        self.assertAlmostEqual(doc["rate_per_s"], 2.2, delta=0.1)    # 11 units / 5s
        # 10ms of work 2.2 times a second = 2.2% of a thread.
        self.assertAlmostEqual(doc["busy_fraction"], 0.022, delta=0.005)

    def test_a_batch_says_how_many_frames_it_bought(self):
        """35ms that decoded four frames is 8.75ms of detection, not 35."""
        board = PerfBoard()
        _series(board, "detect_batch", "engine", 35.0, n=10, period_s=0.1, units=4)
        doc = board.snapshot()["detect_batch"]["engine"]
        self.assertEqual(doc["count"], 10)
        self.assertEqual(doc["units"], 40)
        self.assertEqual(doc["mean_ms"], 35.0)          # per observation, unchanged
        self.assertAlmostEqual(doc["per_unit_ms"], 8.8, delta=0.1)   # per frame

    def test_one_sample_yields_no_rate_rather_than_a_wrong_one(self):
        board = PerfBoard()
        board.observe("verify_infer", "cam1", 12000.0)
        doc = board.snapshot()["verify_infer"]["cam1"]
        self.assertIsNone(doc["rate_per_s"])
        self.assertIsNone(doc["busy_fraction"])
        self.assertEqual(doc["p95_ms"], 12000.0)        # the latency still stands

    def test_notes_ride_along_with_the_series(self):
        board = PerfBoard()
        _series(board, "decode", "cam1", 10.0, n=5, period_s=0.5)
        board.note("decode", "cam1", sustainable_fps=5.9, target_fps=12.0, limited=True)
        doc = board.snapshot()["decode"]["cam1"]
        self.assertEqual(doc["sustainable_fps"], 5.9)
        self.assertTrue(doc["limited"])


class RankingTests(unittest.TestCase):
    """The regression guard: this must never become a sort by p95."""

    def _board_where_the_loudest_stage_is_cheap(self) -> PerfBoard:
        """Numbers taken from runs/feeds/stage/perf_report.json — a real box."""
        board = PerfBoard()
        # The English scanner: 4.8s per cycle, once every 12s. Its p95 is 40x
        # detection's and it would top a naive ranking — but it runs so rarely
        # that it costs 0.4 of a thread.
        _series(board, "english_scan", "Front Door", 4_800.0, n=20, period_s=12.0)
        # Detection: 120ms a batch of 4, eight batches a second. A p95 in the
        # low hundreds of ms, and very nearly a whole core. The real expense.
        _series(board, "detect_batch", "engine", 120.0, n=100, period_s=0.125, units=4)
        return board

    def test_the_loudest_stage_is_not_the_top_cost_stage(self):
        stages = self._board_where_the_loudest_stage_is_cheap().snapshot()
        by_p95 = sorted(
            ((k, d["p95_ms"]) for s in stages.values() for k, d in s.items()),
            key=lambda kv: kv[1], reverse=True)
        self.assertEqual(by_p95[0][0], "Front Door")     # p95 would blame the scanner

        ranked = rank_cost(stages)
        self.assertEqual(ranked[0]["stage"], "detect_batch")   # cost does not
        self.assertGreater(ranked[0]["busy_fraction"], ranked[-1]["busy_fraction"])

    def test_the_scanners_share_is_small_despite_its_latency(self):
        stages = self._board_where_the_loudest_stage_is_cheap().snapshot()
        scan = stages["english_scan"]["Front Door"]
        detect = stages["detect_batch"]["engine"]
        # Loud: the scanner's p95 is ~40x detection's.
        self.assertGreater(scan["p95_ms"], detect["p95_ms"] * 30)
        # Cheap: 4.8s of work every 12s is 0.4 of a thread ...
        self.assertAlmostEqual(scan["busy_fraction"], 0.40, delta=0.05)
        # ... against detection's 120ms eight times a second.
        self.assertAlmostEqual(detect["busy_fraction"], 0.96, delta=0.05)
        self.assertGreater(detect["busy_fraction"], scan["busy_fraction"])

    def test_waiting_is_never_ranked_as_work(self):
        board = PerfBoard()
        _series(board, "verify_wait", "cam1", 340_000.0, n=20, period_s=30.0)
        _series(board, "detect_batch", "engine", 40.0, n=100, period_s=0.125, units=4)
        stages = board.snapshot()
        self.assertNotIn("verify_wait", [r["stage"] for r in rank_cost(stages)])
        pressure = saturation(stages)
        self.assertEqual(pressure[0]["stage"], "verify_wait")
        self.assertGreater(pressure[0]["p95_ms"], 300_000)

    def test_an_unrankable_series_sorts_last_not_first(self):
        """Unknown is not cheap — a legacy series must not win the ranking."""
        board = PerfBoard()
        _series(board, "detect_batch", "engine", 40.0, n=100, period_s=0.125, units=4)
        stages = board.snapshot()
        stages["decode"] = {"legacy": {"count": 512, "mean_ms": 900.0, "p95_ms": 4000.0}}
        ranked = rank_cost(stages)
        self.assertEqual(ranked[0]["stage"], "detect_batch")
        self.assertIsNone(ranked[-1]["busy_fraction"])


class VerdictTests(unittest.TestCase):
    @staticmethod
    def _report(stages: dict, **system) -> dict:
        base = {"cpu_percent": 55.0, "cpu_count": 4, "memory_total_gb": 16.0,
                "memory_available_gb": 6.0, "memory_percent": 62.0}
        base.update(system)
        return {"generated_at": time.time(), "stages": stages, "system": base}

    def test_detection_bound_box_is_sent_to_W2(self):
        board = PerfBoard()
        _series(board, "detect_batch", "engine", 120.0, n=100, period_s=0.125, units=4)
        v = verdict(self._report(board.snapshot()))
        self.assertIn("detect_batch", v["headline"])
        self.assertIn("W2", v["remedy"])

    def test_a_starved_gate_is_told_to_fix_the_thief_first(self):
        """Martins's 360s verdicts: the queue, not the model."""
        board = PerfBoard()
        _series(board, "detect_batch", "engine", 120.0, n=100, period_s=0.125, units=4)
        _series(board, "verify_wait", "cam1", 340_000.0, n=20, period_s=30.0)
        v = verdict(self._report(board.snapshot()))
        self.assertIn("STARVED", v["detail"])
        self.assertIn("detect_batch", v["detail"])
        self.assertIn("W2", v["remedy"])          # not W5 — the queue is a symptom

    def test_a_genuinely_slow_verdict_model_is_sent_to_W5(self):
        board = PerfBoard()
        _series(board, "verify_infer", "cam1", 40_000.0, n=20, period_s=45.0)
        _series(board, "verify_wait", "cam1", 60_000.0, n=20, period_s=45.0)
        v = verdict(self._report(board.snapshot()))
        self.assertIn("verify_infer", v["headline"])
        self.assertIn("W5", v["remedy"])

    def test_decode_is_not_blamed_when_the_machine_keeps_up(self):
        """At the live edge decode's ms include waiting for the camera. A big
        number on a machine that is NOT ingest-limited is the source pacing
        us, and blaming it would buy go2rtc to fix a healthy box."""
        board = PerfBoard()
        _series(board, "decode", "Slow Camera", 400.0, n=50, period_s=0.5)
        board.note("decode", "Slow Camera", limited=False, sustainable_fps=30.0,
                   target_fps=4.0)
        _series(board, "detect_batch", "engine", 40.0, n=100, period_s=0.125, units=4)
        v = verdict(self._report(board.snapshot()))
        self.assertNotIn("decode", v["headline"])
        self.assertIn("detect_batch", v["headline"])
        self.assertTrue(any("not ingest-limited" in n.lower() or
                            "NOT ingest-limited" in n for n in v["notes"]))

    def test_decode_IS_blamed_when_the_machine_cannot_keep_up(self):
        board = PerfBoard()
        _series(board, "decode", "Front Door", 400.0, n=50, period_s=0.4)
        board.note("decode", "Front Door", limited=True, sustainable_fps=2.5,
                   target_fps=12.0)
        v = verdict(self._report(board.snapshot()))
        self.assertIn("decode", v["headline"])
        self.assertIn("W1", v["remedy"])
        self.assertTrue(ingest_warnings(
            {"decode": {"Front Door": {"limited": True, "sustainable_fps": 2.5,
                                       "target_fps": 12.0}}}))

    def test_a_paging_box_is_named_before_any_stage(self):
        board = PerfBoard()
        _series(board, "detect_batch", "engine", 120.0, n=100, period_s=0.125, units=4)
        v = verdict(self._report(board.snapshot(),
                                 memory_percent=93.0, memory_available_gb=1.1))
        self.assertIn("Memory", v["headline"])
        self.assertIn("provisional", v["detail"])

    def test_a_legacy_report_says_so_instead_of_claiming_no_data(self):
        legacy = {"generated_at": time.time(),
                  "stages": {"decode": {"cam1": {"count": 512, "mean_ms": 52.9,
                                                 "p95_ms": 420.9}}},
                  "system": {"memory_percent": 60.0}}
        v = verdict(legacy)
        self.assertIn("predates", v["headline"])
        self.assertNotIn("before doing measurable work", v["detail"])

    def test_an_empty_report_asks_for_a_longer_capture(self):
        v = verdict({"generated_at": time.time(), "stages": {}, "system": {}})
        self.assertIn("No stage", v["headline"])
        self.assertIn("few minutes", v["detail"])


class AbsenceTests(unittest.TestCase):
    def test_a_stage_that_never_ran_is_named(self):
        board = PerfBoard()
        _series(board, "decode", "cam1", 10.0, n=10, period_s=0.5)
        absent = missing_stages(board.snapshot())
        self.assertIn("verify_infer", absent)      # verification never happened
        self.assertIn("english_scan", absent)
        self.assertNotIn("decode", absent)

    def test_the_rendered_report_prints_the_absence(self):
        board = PerfBoard()
        _series(board, "decode", "cam1", 10.0, n=10, period_s=0.5)
        text = render({"generated_at": time.time(), "stages": board.snapshot(),
                       "system": {"cpu_count": 4}}, "test")
        self.assertIn("NOT MEASURED", text)
        self.assertIn("verify_infer", text)


class RenderingTests(unittest.TestCase):
    def test_the_total_says_whether_the_box_is_oversubscribed(self):
        """Six stages at 0.6 of a thread each is fine on 8 cores and fatal on
        2. The sum against the core count is the finding, so it is printed."""
        board = PerfBoard()
        _series(board, "detect_batch", "engine", 250.0, n=100, period_s=0.25, units=2)
        _series(board, "decode", "cam1", 240.0, n=100, period_s=0.25)
        report = {"generated_at": time.time(), "stages": board.snapshot(),
                  "system": {"cpu_count": 2}}
        text = render(report, "test")
        self.assertIn("TOTAL", text)
        self.assertIn("of 2 cores", text)
        self.assertIn("OVERSUBSCRIBED", text)

    def test_a_rare_stage_shows_its_period_not_a_rounded_zero(self):
        from tools.perf_readout import _fmt_rate
        self.assertEqual(_fmt_rate(1 / 12.0), "1/12s")   # not "0.1/s"
        self.assertEqual(_fmt_rate(8.0), "8.0/s")
        self.assertEqual(_fmt_rate(None), "—")


class BundleTests(unittest.TestCase):
    def test_the_readout_reads_a_diagnostics_zip(self):
        """The support path end to end: what the customer sends, in one command."""
        BOARD.observe("detect_batch", "engine", 40.0, units=4)
        BOARD.observe("detect_batch", "engine", 42.0, units=4)
        with tempfile.TemporaryDirectory() as tmp:
            report_path = write_report(tmp)
            self.assertIsNotNone(report_path)
            bundle = Path(tmp) / "argus-diagnostics-test.zip"
            with zipfile.ZipFile(bundle, "w") as zf:
                zf.write(report_path, "perf_report.json")
                zf.writestr("logs/monitor.log", "engine started\n")
            doc, origin = load_report(bundle)
            self.assertIn("detect_batch", doc["stages"])
            self.assertIn("perf_report.json", origin)
            self.assertIn("ARGUS PERF READOUT", render(doc, origin))

    def test_a_bare_json_and_a_directory_both_work(self):
        BOARD.observe("decode", "dir-test-cam", 9.0)
        with tempfile.TemporaryDirectory() as tmp:
            write_report(tmp)
            from_dir, _ = load_report(tmp)
            from_file, _ = load_report(Path(tmp) / "perf_report.json")
            self.assertEqual(from_dir["stages"].keys(), from_file["stages"].keys())

    def test_a_zip_without_a_report_says_which_build_to_ask_for(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "old.zip"
            with zipfile.ZipFile(bundle, "w") as zf:
                zf.writestr("logs/monitor.log", "engine started\n")
            with self.assertRaises(SystemExit) as caught:
                load_report(bundle)
            self.assertIn("older than 4 Sep", str(caught.exception))


class JsonOutputTests(unittest.TestCase):
    def test_the_json_mode_carries_the_verdict(self):
        from tools.perf_readout import main
        import io
        import contextlib
        BOARD.observe("detect_batch", "engine", 40.0, units=4)
        with tempfile.TemporaryDirectory() as tmp:
            write_report(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(main([tmp, "--json"]), 0)
            doc = json.loads(buf.getvalue())
            for key in ("cost", "saturation", "not_measured", "verdict"):
                self.assertIn(key, doc)


if __name__ == "__main__":
    unittest.main()
