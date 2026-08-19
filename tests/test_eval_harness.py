"""Tests for the evaluation harness — metrics, dataset labelling, and checkpointing.

No models, no Ollama: the scoring maths and the resume logic are what can silently
go wrong, so they're tested directly on synthetic clip results.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.eval.dataset import EvalClip, describe, load_dataset
from cvti.eval.harness import ClipResult, EvalHarness
from cvti.eval import metrics
from cvti.eval.metrics import compare_stages, render_report, score


def _rows():
    """3 threat clips, 3 normal. Detectors fire on everything (noisy);
    TrueSight keeps the real ones and kills 2 of the 3 false alarms."""
    return [
        {"is_threat": True,  "candidates": 4, "confirmed": 2},
        {"is_threat": True,  "candidates": 2, "confirmed": 1},
        {"is_threat": True,  "candidates": 3, "confirmed": 0},   # gate lost a real threat
        {"is_threat": False, "candidates": 3, "confirmed": 0},
        {"is_threat": False, "candidates": 2, "confirmed": 0},
        {"is_threat": False, "candidates": 1, "confirmed": 1},    # survived: still a false alarm
    ]


class MetricsTests(unittest.TestCase):
    def test_stage_scoring_counts_clips_not_alerts(self):
        m = score(_rows(), "raw", "candidates")
        self.assertEqual((m.tp, m.fn, m.fp, m.tn), (3, 0, 3, 0))
        self.assertEqual(m.alerts, 15)
        self.assertAlmostEqual(m.recall, 1.0)
        self.assertAlmostEqual(m.precision, 0.5)
        self.assertAlmostEqual(m.fpr, 1.0)

    def test_gate_stage_improves_precision(self):
        raw = score(_rows(), "raw", "candidates")
        gated = score(_rows(), "gated", "confirmed")
        self.assertEqual((gated.tp, gated.fn, gated.fp, gated.tn), (2, 1, 1, 2))
        self.assertGreater(gated.precision, raw.precision)   # 0.667 > 0.5
        self.assertLess(gated.fpr, raw.fpr)

    def test_compare_stages_reports_the_headline_delta(self):
        s = compare_stages(_rows())
        d = s["delta"]
        self.assertEqual(d["alerts_suppressed"], 15 - 4)
        self.assertEqual(d["false_alarm_clips_removed"], 2)
        self.assertEqual(d["threats_lost_to_gate"], 1)     # honest about the cost
        self.assertGreater(d["precision_gain"], 0)

    def test_empty_metrics_do_not_divide_by_zero(self):
        s = compare_stages([])
        self.assertIsNone(s["raw_detectors"]["precision"])
        self.assertIsNone(s["delta"]["alerts_suppressed_pct"])

    def test_report_mentions_the_key_numbers(self):
        s = compare_stages(_rows())
        md = render_report(s, {"total": 6, "threat": 3, "normal": 3, "sources": ["x"]},
                           {"gate": "ollama", "gate_model": "gemma3:4b"})
        self.assertIn("suppressed", md.lower())
        self.assertIn("Precision", md)
        self.assertIn("⚠️", md)          # warns it lost a real threat


class DatasetTests(unittest.TestCase):
    def test_local_filenames_are_labelled_correctly(self):
        clips = load_dataset("local")
        if not clips:
            self.skipTest("no local clips present")
        by = {c.name: c for c in clips}
        for n, c in by.items():
            if n.startswith(("theft_", "violence_", "weapons_", "stabbing")):
                self.assertTrue(c.is_threat, f"{n} should be a threat")
            if n.startswith(("normal_", "empty_")):
                self.assertFalse(c.is_threat, f"{n} should be normal")

    def test_limit_keeps_both_classes(self):
        clips = [EvalClip(f"t{i}.mp4", True) for i in range(5)] + \
                [EvalClip(f"n{i}.mp4", False) for i in range(5)]
        import cvti.eval.dataset as ds
        orig = ds._camnuvem_clips
        ds._camnuvem_clips = lambda limit_per_class=0: clips
        try:
            got = ds.load_dataset("camnuvem", limit=4)
        finally:
            ds._camnuvem_clips = orig
        self.assertEqual(len(got), 4)
        self.assertTrue(any(c.is_threat for c in got))
        self.assertTrue(any(not c.is_threat for c in got))

    def test_describe(self):
        d = describe([EvalClip("a", True, source="s"), EvalClip("b", False, source="s")])
        self.assertEqual((d["total"], d["threat"], d["normal"]), (2, 1, 1))


class ResumeTests(unittest.TestCase):
    def test_completed_clips_are_skipped_on_resume(self):
        d = Path(tempfile.mkdtemp())
        (d / "clip_results_k.jsonl").write_text(json.dumps(
            ClipResult("a.mp4", "/x/a.mp4", True, candidates=2, confirmed=1).to_dict()) + "\n")
        h = EvalHarness(out_dir=str(d), run_key="k")
        # would need models if it actually ran the clip; resume must avoid that
        h.run_clip = lambda clip: self.fail("should not re-run a checkpointed clip")
        out = h.run([EvalClip("/x/a.mp4", True)], progress=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].candidates, 2)
        self.assertEqual(out[0].confirmed, 1)


if __name__ == "__main__":
    unittest.main()


class SensitivityTests(unittest.TestCase):
    """The recall/precision trade-off is a measured setting, not a hidden constant."""

    def test_strict_overrides_the_theft_question(self):
        from cvti.verification.gate import _build_question
        bal = _build_question("video_theft_candidate", "retail", "video_action", "balanced")
        strict = _build_question("video_theft_candidate", "retail", "video_action", "strict")
        self.assertNotEqual(bal, strict)
        # the strict preset is what stopped 'taking an item off a shelf' counting as theft
        self.assertIn("NORMAL SHOPPING", strict)
        self.assertNotIn("NORMAL SHOPPING", bal)

    def test_unknown_sensitivity_falls_back_to_balanced(self):
        from cvti.verification.gate import VerificationGate
        self.assertEqual(VerificationGate(sensitivity="nonsense").sensitivity, "balanced")

    def test_measured_numbers_are_published_with_their_dataset(self):
        from cvti.verification.gate import SENSITIVITY_MEASURED
        for key in ("balanced", "strict"):
            m = SENSITIVITY_MEASURED[key]
            self.assertTrue(0 < m["recall"] <= 1 and 0 < m["precision"] <= 1)
        # strict trades recall for precision — that is the whole point
        self.assertLess(SENSITIVITY_MEASURED["strict"]["recall"],
                        SENSITIVITY_MEASURED["balanced"]["recall"])
        self.assertGreater(SENSITIVITY_MEASURED["strict"]["precision"],
                           SENSITIVITY_MEASURED["balanced"]["precision"])
        self.assertIn("held-out", SENSITIVITY_MEASURED["_dataset"])


class WilsonIntervalTest(unittest.TestCase):
    """Published numbers must never be point estimates alone — see docs/NUMBERS.md."""

    def test_perfect_score_on_a_small_sample_is_not_certainty(self):
        # The whole reason this exists: 9/9 is not "we never miss a fire".
        lo, hi = metrics.wilson_interval(9, 9)
        self.assertAlmostEqual(lo, 0.7009, places=3)
        self.assertEqual(hi, 1.0)

    def test_interval_brackets_the_estimate_and_stays_in_range(self):
        for k, n in ((0, 30), (2, 30), (15, 30), (30, 30), (1, 1)):
            lo, hi = metrics.wilson_interval(k, n)
            self.assertLessEqual(lo, k / n)
            self.assertGreaterEqual(hi, k / n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)

    def test_interval_narrows_as_the_sample_grows(self):
        narrow = metrics.wilson_interval(90, 100)
        wide = metrics.wilson_interval(9, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_no_sample_means_no_interval(self):
        self.assertIsNone(metrics.wilson_interval(0, 0))

    def test_every_published_rate_carries_n_and_an_interval(self):
        m = metrics.score(
            [{"is_threat": True, "confirmed": 1}] * 9 + [{"is_threat": False, "confirmed": 0}] * 30,
            "truesight_confirmed", "confirmed").to_dict()
        for key in ("precision", "recall", "fpr"):
            self.assertIn(f"{key}_n", m)
            self.assertIn(f"{key}_ci", m)
        self.assertEqual(m["recall_n"], 9)
        self.assertEqual(m["fpr_n"], 30)
