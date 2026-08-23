from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.contracts import RawEvent
from cvti.rules.customization import CustomizationEngine, _logic_satisfied, PRIORITY_ORDER

HIGH = PRIORITY_ORDER["high"]
MED = PRIORITY_ORDER["medium"]


def _ev(detector, level="medium", title="X", signal_type=None):
    extra = {"signal_type": signal_type} if signal_type else {}
    return RawEvent(detector=detector, active=True, title=title, level=level,
                    timestamp=1.0, extra=extra)


class LogicOpTests(unittest.TestCase):
    def test_one_high_or_two_medium(self):
        f = _logic_satisfied
        self.assertTrue(f("one_high_or_two_medium", [HIGH], 6))          # one high
        self.assertTrue(f("one_high_or_two_medium", [MED, MED], 6))      # two medium
        self.assertFalse(f("one_high_or_two_medium", [MED], 6))          # one medium only
    def test_at_least_and_all(self):
        self.assertTrue(_logic_satisfied("at_least_2", [MED, MED], 3))
        self.assertFalse(_logic_satisfied("at_least_2", [MED], 3))
        self.assertTrue(_logic_satisfied("all", [MED, MED, MED], 3))
        self.assertFalse(_logic_satisfied("all", [MED, MED], 3))


class CompoundRecipeTests(unittest.TestCase):
    def setUp(self):
        self.engine = CustomizationEngine("configs/compound_recipes_v1.json")

    def test_armed_robbery_fires_on_one_high_signal(self):
        alerts = self.engine.evaluate([_ev("weapons", level="critical", title="GUN")])
        ar = [a for a in alerts if a.rule_name == "armed_robbery"]
        self.assertEqual(len(ar), 1)
        self.assertEqual(ar[0].detector, "compound")
        self.assertEqual(ar[0].priority, "critical")
        self.assertIsNotNone(ar[0].question)          # recipe gate_question threaded through

    def test_violent_theft_needs_two_signals(self):
        one = self.engine.evaluate([_ev("concealment", level="high")])
        self.assertFalse(any(a.rule_name == "violent_theft" for a in one))
        two = self.engine.evaluate([_ev("concealment", level="high"), _ev("running", level="medium")])
        self.assertTrue(any(a.rule_name == "violent_theft" for a in two))

    def test_video_action_signal_type_matches(self):
        # a weak VideoMAE "violence_candidate" should count as the violence signal
        ev = _ev("video_action", level="medium", signal_type="violence_candidate")
        alerts = self.engine.evaluate([ev, _ev("running", level="medium")])
        # violence(1) + running(1) -> two medium -> armed_robbery fires
        self.assertTrue(any(a.rule_name == "armed_robbery" for a in alerts))


if __name__ == "__main__":
    unittest.main()


class SimpleRuleQuestionTest(unittest.TestCase):
    """A rule's plain-English gate_question must reach the VLM (demo-day bug).

    Compound recipes always carried it; simple trigger rules silently dropped
    it — the customer's sentence never reached the model, and the gate fell
    back to a generic question while the UI implied otherwise.
    """

    def test_gate_question_rides_the_candidate(self):
        import json
        import tempfile
        from cvti.contracts import RawEvent
        from cvti.rules.customization import CustomizationEngine
        rule = {"use_case_id": "t", "rules": [{
            "name": "stated", "trigger": {"detector": "presence"},
            "priority": "high",
            "gate_question": "Is someone waving both arms above their head?"}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(rule, f)
        eng = CustomizationEngine(f.name)
        got = eng.evaluate([RawEvent(detector="presence", active=True,
                                     title="p", level="low")], scene_context={})
        self.assertEqual(got[0].question,
                         "Is someone waving both arms above their head?")

    def test_rules_without_a_question_keep_the_default(self):
        import json
        import tempfile
        from cvti.contracts import RawEvent
        from cvti.rules.customization import CustomizationEngine
        rule = {"use_case_id": "t", "rules": [{
            "name": "plain", "trigger": {"detector": "presence"}}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(rule, f)
        eng = CustomizationEngine(f.name)
        got = eng.evaluate([RawEvent(detector="presence", active=True,
                                     title="p", level="low")], scene_context={})
        self.assertIsNone(got[0].question)
