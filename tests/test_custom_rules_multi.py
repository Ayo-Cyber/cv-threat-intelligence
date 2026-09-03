"""Several English rules true at once must ALL fire.

27 Aug, caught live: the operator wrote two glasses rules while a hoodie rule
was already firing. The scanner asked the VLM for THE threat — singular — so
one camera got one answer per cycle, whichever the model found most salient.
The hoodie won every scan; the glasses rules never fired once. The evidence
frame from the "hoodie" alert shows the operator plainly wearing glasses.

A rule that never fires because a NEIGHBOURING rule is also true is
indistinguishable, to the customer, from a product that does not work.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cvti.serving.custom_rules import CustomRuleScanner

CAM = {"id": "stage", "source": "0", "custom_rules": [
    {"question": "Is any person wearing a black hoodie?", "dwell": 4.0},
    {"question": "Detect anyone wearing eye glasses", "dwell": 4.0},
]}
FRAME = np.zeros((32, 32, 3), dtype=np.uint8)


def _check_with(raw: str):
    s = CustomRuleScanner([CAM], sink=None, model="test")
    with mock.patch("cvti.scene.agent_mapper.call_openai_compatible", return_value=raw):
        return s._check(CAM, FRAME)


class EveryTrueRuleFiresTest(unittest.TestCase):

    def test_two_true_rules_both_come_back(self):
        raw = json.dumps({"threats": [
            {"name": "is any person wearing a", "reason": "black hoodie visible"},
            {"name": "detect anyone wearing eye glasses", "reason": "glasses visible"},
        ]})
        hits = _check_with(raw)
        self.assertEqual(len(hits), 2, "a true rule was shadowed by its neighbour")

    def test_the_old_singular_shape_is_still_honoured(self):
        raw = json.dumps({"threat": "detect anyone wearing eye glasses", "reason": "glasses"})
        hits = _check_with(raw)
        self.assertEqual([h["name"] for h in hits], ["detect anyone wearing eye glasses"])

    def test_an_invented_threat_is_never_fired(self):
        raw = json.dumps({"threats": [{"name": "person holding a rifle", "reason": "x"}]})
        self.assertEqual(_check_with(raw), [],
                         "the model invented a threat the customer never defined")

    def test_duplicate_claims_fire_once(self):
        raw = json.dumps({"threats": [
            {"name": "detect anyone wearing eye glasses", "reason": "a"},
            {"name": "detect anyone wearing eye glasses", "reason": "b"},
        ]})
        self.assertEqual(len(_check_with(raw)), 1)

    def test_none_and_garbage_are_quiet(self):
        for raw in (json.dumps({"threats": []}),
                    json.dumps({"threat": "none", "reason": ""}),
                    "not json at all", ""):
            self.assertEqual(_check_with(raw), [], f"fired on {raw!r}")

    def test_cooldown_is_per_rule_not_per_camera(self):
        s = CustomRuleScanner([CAM], sink=None, model="test")
        self.assertFalse(s._cooling("stage", "hoodie"))
        self.assertTrue(s._cooling("stage", "hoodie"), "same rule should cool")
        self.assertFalse(s._cooling("stage", "glasses"),
                         "one rule's cooldown muted a different rule")

    def test_every_scan_path_emits_every_hit(self):
        """The loop used to take one hit; a regression here re-shadows rules.
        Both former loop paths now scan through _scan_camera (3 Sep), so the
        one emit loop there covers them — as long as nothing scans around it."""
        import inspect
        from cvti.serving.custom_rules import CustomRuleScanner
        scan = inspect.getsource(CustomRuleScanner._scan_camera)
        self.assertIn("for hit in hits:", scan,
                      "the scan path stopped iterating over all hits")
        src = Path("cvti/serving/custom_rules.py").read_text()
        self.assertNotIn('"threat": "<exact threat name', src,
                         "the singular-answer prompt is back")


if __name__ == "__main__":
    unittest.main()
