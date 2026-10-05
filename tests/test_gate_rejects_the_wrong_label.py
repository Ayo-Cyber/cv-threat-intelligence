"""A real incident under the wrong label is a rejection, not a confirmation.

3 Oct 2026, demo wall: an ATM break-in clip raised fire, panic running, violence
AND theft. The verifier confirmed the fire alert while writing "there is no
visible fire or smoke" — the prompt rewarded "something bad is happening". One
incident reached the operator four times, three labels wrong.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.verification import gate as gate_mod


class WrongLabelIsARejectionTest(unittest.TestCase):
    def test_other_threat_overrides_a_confirmed_true(self):
        raw = ('Reason: a man with a chain is forcing the ATM; no fire or smoke anywhere.\n'
               '{"confirmed": true, "confidence": 0.95, "observed": "other_threat", '
               '"reason": "The individual is damaging the ATM with a chain.", '
               '"alert_priority": "critical"}')
        out = gate_mod._parse_response(raw, "critical")
        self.assertFalse(out.confirmed)
        self.assertFalse(out.errored)                       # a verdict, not a failure
        self.assertTrue(out.reason.startswith("different incident than claimed:"), out.reason)
        self.assertAlmostEqual(out.confidence, 0.95)

    def test_claimed_keeps_a_confirmation(self):
        raw = '{"confirmed": true, "confidence": 0.9, "observed": "claimed", "reason": "flames"}'
        self.assertTrue(gate_mod._parse_response(raw, "critical").confirmed)

    def test_an_answer_without_the_field_is_unchanged(self):
        # Older prompt revisions / other providers: no "observed" key, no change.
        raw = '{"confirmed": true, "confidence": 0.9, "reason": "flames"}'
        self.assertTrue(gate_mod._parse_response(raw, "critical").confirmed)

    def test_both_prompts_tell_the_model_a_different_threat_is_false(self):
        for tpl in (gate_mod._PROMPT_TEMPLATE, gate_mod._COT_PROMPT_TEMPLATE):
            self.assertIn("other_threat", tpl)
            self.assertIn('"observed"', tpl)
            self.assertIn("the claim is fire and you see a", tpl)   # the worked example


if __name__ == "__main__":
    unittest.main()
