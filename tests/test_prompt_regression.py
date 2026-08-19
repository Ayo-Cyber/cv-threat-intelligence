"""Prompt regression guard (EP-07-T1).

The headline precision figure is a function of a string literal in `gate.py`.
Three prompt revisions moved theft precision 37.5% -> 53.3% -> 63.6% — a
26-point swing on wording alone, with nothing guarding it. Anyone could edit
`_QUESTIONS`, run the app, see it work, and ship that invisibly.
"""
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from cvti.contracts import VerificationResult
from cvti.eval import prompt_fingerprint as fp
from cvti.eval.golden import GoldenSet, GoldenSetWriter, score


def _candidate(rule="shoplifting"):
    from cvti.contracts import CandidateAlert
    return CandidateAlert(rule_name=rule, priority="high", detector="concealment",
                          title="POSSIBLE CONCEALMENT", person_id=1,
                          object_label=None, timestamp=0.0)


class FingerprintTest(unittest.TestCase):
    def test_it_covers_every_prompt_table(self):
        # A fingerprint with a hole in it is worse than none — it reads as
        # coverage. _DETECTOR_QUESTIONS holds the weapons question, on an
        # always-on critical detector.
        self.assertEqual(fp.describe()["missing"], [])
        for name in fp.PROMPT_NAMES:
            self.assertIn(name, fp.extract_prompts())

    def test_rewording_a_question_changes_it(self):
        before = fp.fingerprint()
        stub = types.SimpleNamespace(**fp.extract_prompts())
        stub._QUESTIONS = dict(stub._QUESTIONS)
        stub._QUESTIONS["shoplifting"] = "Is this person definitely stealing?"
        self.assertNotEqual(fp.fingerprint(stub), before)

    def test_changing_a_detector_question_changes_it(self):
        # The weapons wording lives here. It must not be invisible.
        before = fp.fingerprint()
        stub = types.SimpleNamespace(**fp.extract_prompts())
        stub._DETECTOR_QUESTIONS = dict(stub._DETECTOR_QUESTIONS)
        stub._DETECTOR_QUESTIONS["weapons"] = "Any weapon?"
        self.assertNotEqual(fp.fingerprint(stub), before)

    def test_unrelated_module_changes_do_not_move_it(self):
        before = fp.fingerprint()
        stub = types.SimpleNamespace(**fp.extract_prompts())
        stub.SOME_NEW_CONSTANT = "a refactor, a new provider, a renamed variable"
        self.assertEqual(fp.fingerprint(stub), before)

    def test_it_is_stable_across_calls(self):
        self.assertEqual(fp.fingerprint(), fp.fingerprint())


class GoldenSetTest(unittest.TestCase):
    def _write(self, tmp, cases):
        writer = GoldenSetWriter(tmp)
        for clip, is_threat in cases:
            writer.add(clip_name=clip, is_threat=is_threat, candidate=_candidate(),
                       frames=[np.zeros((8, 8, 3), np.uint8)], scene={"environment_type": "retail"})
        writer.write({"dataset": "test"})
        return GoldenSet(tmp)

    def test_round_trips_cases_and_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden = self._write(tmp, [("a.mp4", True), ("b.mp4", False)])
            self.assertEqual(len(golden), 2)
            self.assertEqual([c.is_threat for c in golden.cases], [True, False])
            self.assertEqual(len(golden.load_frames(golden.cases[0])), 1)

    def test_replay_scores_against_ground_truth_not_the_old_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden = self._write(tmp, [("threat.mp4", True), ("normal.mp4", False)])

            class ConfirmsEverything:
                def verify(self, frames, candidate, scene=None, examples=None):
                    return VerificationResult(True, 0.9, "yes", "high", "t")

            result = score(golden.replay(ConfirmsEverything()))
            self.assertEqual(result["tp"], 1)
            self.assertEqual(result["fp"], 1)
            self.assertEqual(result["recall"], 1.0)
            self.assertEqual(result["precision"], 0.5)

    def test_a_gate_error_is_excluded_rather_than_scored_as_a_rejection(self):
        # Scoring transport failures as "the model said no" is how a broken
        # gate reports excellent precision.
        with tempfile.TemporaryDirectory() as tmp:
            golden = self._write(tmp, [("threat.mp4", True), ("normal.mp4", False)])

            class Broken:
                def verify(self, *a, **k):
                    raise ConnectionRefusedError("ollama is down")

            result = score(golden.replay(Broken()))
            self.assertEqual(result["errors"], 2)
            self.assertEqual(result["scored"], 0)
            self.assertIsNone(result["precision"])
            self.assertIsNone(result["recall"])

    def test_an_unverified_verdict_is_an_error_not_a_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden = self._write(tmp, [("threat.mp4", True)])

            class Unverified:
                def verify(self, *a, **k):
                    return VerificationResult(True, 0.0, "UNVERIFIED", "high", "t",
                                              error="transport: OSError")

            self.assertEqual(score(golden.replay(Unverified()))["errors"], 1)

    def test_scores_carry_intervals_and_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden = self._write(tmp, [("t.mp4", True)] * 5 + [("n.mp4", False)] * 5)

            class Perfect:
                def verify(self, frames, candidate, scene=None, examples=None):
                    return VerificationResult(True, 0.9, "y", "high", "t")

            result = score(golden.replay(Perfect()))
            self.assertEqual(result["recall_n"], 5)
            self.assertEqual(len(result["recall_ci"]), 2)
            self.assertLess(result["recall_ci"][0], 1.0, "a perfect 5/5 is not certainty")


class BaselineTest(unittest.TestCase):
    BASELINE = Path("docs/prompt_baseline.json")

    def test_the_committed_baseline_matches_the_current_prompts(self):
        # This is the same assertion CI makes. If it fails here, the wording
        # changed and nobody re-measured.
        self.assertTrue(self.BASELINE.exists(),
                        "no prompt baseline — run tools/prompt_regression.py run "
                        "--update-baseline")
        base = json.loads(self.BASELINE.read_text())
        self.assertEqual(
            base.get("fingerprint"), fp.fingerprint(),
            "gate prompt text changed without re-measuring — see "
            "tools/prompt_regression.py")

    def test_the_baseline_records_what_it_measured(self):
        base = json.loads(self.BASELINE.read_text())
        for key in ("precision", "recall", "golden_cases", "gate_model", "tolerance",
                    "measured_at"):
            self.assertIn(key, base)
        self.assertGreater(base["golden_cases"], 0)

    def test_tolerance_is_tighter_on_recall_than_precision(self):
        # Losing precision costs an operator a review. Losing recall means a
        # threat is not reported, and there is no second chance at that.
        tol = json.loads(self.BASELINE.read_text())["tolerance"]
        self.assertLess(tol["recall"], tol["precision"])


if __name__ == "__main__":
    unittest.main()
