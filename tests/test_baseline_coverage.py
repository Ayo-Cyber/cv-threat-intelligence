"""Every safety detector must have a rule listening for it.

A detector can fire perfectly and still be invisible: if no rule triggers on its
name, CustomizationEngine drops the event and the operator never learns anything
happened. That is exactly how crowd_formation and running went unnoticed — the
detectors worked, the baseline had no rule for them, and an eval showed 0/8
"missed" as if detection were broken.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "configs" / "baseline_critical_v1.json"
CAMERA = ROOT / "cvti" / "serving" / "camera.py"

# Emitted through adapters/assessments rather than a literal detector="..." in
# camera.py, and covered by the retail presets rather than the safety baseline.
NON_BASELINE = {"presence", "concealment", "video_action", "theft"}


def _baseline_detectors() -> set[str]:
    rules = json.loads(BASELINE.read_text())["rules"]
    return {r["trigger"].get("detector") for r in rules}


def _emitted_detectors() -> set[str]:
    return set(re.findall(r'detector="([a-z_]+)"', CAMERA.read_text()))


class BaselineCoverageTests(unittest.TestCase):
    def test_every_emitted_safety_detector_has_a_baseline_rule(self):
        missing = _emitted_detectors() - _baseline_detectors() - NON_BASELINE
        self.assertEqual(missing, set(),
                         f"these detectors fire but nothing listens: {sorted(missing)}")

    def test_the_two_that_regressed_are_covered(self):
        listens = _baseline_detectors()
        for det in ("crowd_formation", "running", "fire", "person_fall"):
            self.assertIn(det, listens, f"no baseline rule for {det}")

    def test_baseline_rules_are_wellformed(self):
        for r in json.loads(BASELINE.read_text())["rules"]:
            self.assertTrue(r.get("name"))
            self.assertIn("detector", r.get("trigger", {}))
            self.assertIn(r.get("priority"), {"critical", "high", "medium", "low"})

    def test_a_crowd_event_now_produces_an_alert(self):
        """End-to-end through the real engine, not just config inspection."""
        from cvti.contracts import RawEvent
        from cvti.rules.customization import CustomizationEngine
        eng = CustomizationEngine(str(ROOT / "configs" / "all_threats_video_v1.json"),
                                  baseline_path=str(BASELINE))
        for det in ("crowd_formation", "running"):
            ev = RawEvent(detector=det, active=True, title="T", level="high", timestamp=1.0)
            alerts = eng.evaluate([ev])
            self.assertTrue(alerts, f"{det} event produced no alert")


if __name__ == "__main__":
    unittest.main()
