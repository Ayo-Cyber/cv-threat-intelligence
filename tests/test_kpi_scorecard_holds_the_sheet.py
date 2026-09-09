"""W6: the customer's sheet becomes measured numbers, or says why not yet.

The MINIMUM KPI sheet (9 Sep) is the primary acceptance frame. These pins hold
the machinery that answers it: the rows match the sheet, the n floors are
arithmetic rather than optimism, verdicts compare CONSERVATIVE Wilson bounds
to targets, the frozen manifest refuses to drift, and — the one that guards
the whole exercise — footage a model trained on can never score that model.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.eval import kpi  # noqa: E402


class SheetPins(unittest.TestCase):
    """The sheet's rows, targets and semantics — held equal to the document."""

    def test_the_measurable_rows_match_the_sheet(self):
        rows = {r.sn: r for r in kpi.KPI_ROWS}
        self.assertEqual(rows[2].target, 0.05)     # FP rate <= 5%
        self.assertEqual(rows[2].metric, "fpr")
        for sn in (5, 6, 7):                       # person/intrusion/loitering
            self.assertEqual(rows[sn].target, 0.95)
            self.assertEqual(rows[sn].metric, "recall")
        for sn in (8, 9):                          # theft/suspicious
            self.assertEqual(rows[sn].target, 0.90)

    def test_rows_one_and_three_are_deliberately_absent(self):
        """Row 1 (overall accuracy) is a roll-up, not a collection; row 3
        (latency) is instrumented live in health, not a clip metric."""
        self.assertNotIn(1, {r.sn for r in kpi.KPI_ROWS})
        self.assertNotIn(3, {r.sn for r in kpi.KPI_ROWS})


class FloorArithmetic(unittest.TestCase):
    def test_the_floors_follow_from_the_targets(self):
        self.assertEqual(kpi.n_floor(0.95, "recall"), 73)
        self.assertEqual(kpi.n_floor(0.90, "recall"), 35)
        self.assertEqual(kpi.n_floor(0.05, "fpr"), 73)

    def test_sixty_perfect_positives_cannot_prove_95(self):
        """The reason the floor exists: a perfect n=60 run's lower bound is
        93.98%. Anyone who signed >=95% on it would be signing noise."""
        from cvti.eval.metrics import wilson_interval
        lo, _ = wilson_interval(60, 60)
        self.assertLess(lo, 0.95)
        lo73, _ = wilson_interval(73, 73)
        self.assertGreaterEqual(lo73, 0.95)


class VerdictTests(unittest.TestCase):
    class _R:                                     # a minimal ClipResult stand-in
        def __init__(self, confirmed):
            self.confirmed = confirmed

    def _row(self, metric="recall", target=0.90, floor=35):
        return {"sn": 8, "name": "Theft", "metric": metric, "target": target,
                "n_floor": floor}

    def test_met_requires_the_conservative_bound_not_the_point_estimate(self):
        # 34/35 = 97% point estimate, but the lower bound is ~85% — NOT MET.
        results = [self._R(1)] * 34 + [self._R(0)]
        s = kpi.score_row(self._row(), results)
        self.assertGreater(s["rate"], 0.9)
        self.assertEqual(s["verdict"], "NOT MET")

    def test_a_perfect_run_at_the_floor_is_met(self):
        s = kpi.score_row(self._row(), [self._R(1)] * 35)
        self.assertEqual(s["verdict"], "MET")

    def test_below_the_floor_is_smoke_no_matter_how_perfect(self):
        s = kpi.score_row(self._row(), [self._R(1)] * 9)
        self.assertIn("SMOKE", s["verdict"])
        self.assertFalse(s["publishable"])

    def test_fpr_uses_the_upper_bound(self):
        row = self._row(metric="fpr", target=0.05, floor=73)
        # zero alerts on 73 normals: upper bound 5.0% -> exactly at target
        s = kpi.score_row(row, [self._R(0)] * 73)
        self.assertEqual(s["verdict"], "MET")
        # one alert on 73: upper bound ~7.3% -> NOT MET
        s = kpi.score_row(row, [self._R(1)] + [self._R(0)] * 72)
        self.assertEqual(s["verdict"], "NOT MET")


class ContaminationGuard(unittest.TestCase):
    def test_camnuvem_training_split_never_enters_the_manifest(self):
        """VideoMAE was fine-tuned on CamNuvem's training split. Scoring it on
        that footage is testing on the training set — the one mistake that
        would make every theft number on the sheet a lie."""
        for clip in kpi.collect_clips():
            self.assertNotIn("/training/", clip.path.replace("\\", "/"),
                             f"training-split clip leaked into the manifest: "
                             f"{clip.path}")

    def test_robbery_and_burglary_score_as_suspicious_not_theft(self):
        self.assertEqual(kpi.UCF_KPI_KINDS["Robbery"], "suspicious")
        self.assertEqual(kpi.UCF_KPI_KINDS["Burglary"], "suspicious")
        self.assertEqual(kpi.UCF_KPI_KINDS["Shoplifting"], "theft")


class FreezeTests(unittest.TestCase):
    def test_the_digest_is_deterministic(self):
        self.assertEqual(kpi._digest(kpi.collect_clips()),
                         kpi._digest(kpi.collect_clips()))

    def test_a_drifted_manifest_is_refused(self):
        import json
        import tempfile
        doc = kpi.build_manifest()
        doc["digest"] = "0000000000000000"        # footage changed underneath
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump(doc, fh)
        with self.assertRaises(RuntimeError) as caught:
            kpi.load_manifest(Path(fh.name))
        self.assertIn("DELIBERATELY", str(caught.exception))

    def test_the_frozen_manifest_loads_when_nothing_moved(self):
        if not kpi.MANIFEST_PATH.exists():
            self.skipTest("no frozen manifest on this machine")
        if not (ROOT / "data" / "ucf_crime").exists():
            # CI checks out the manifest but not the footage (gigabytes,
            # gitignored) — the digest guard is doing its job there, not
            # failing. The load pin is a dev-machine and replica assertion.
            self.skipTest("clip sets not on this machine (CI)")
        doc = kpi.load_manifest()
        self.assertIn("rows", doc)


if __name__ == "__main__":
    unittest.main()
