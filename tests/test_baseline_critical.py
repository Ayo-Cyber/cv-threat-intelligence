from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.contracts import RawEvent
from cvti.rules.customization import CustomizationEngine

BASELINE = "configs/baseline_critical_v1.json"


def _weapon_event():
    return RawEvent(detector="weapons", active=True, title="GUN", level="critical",
                    object_label="gun", timestamp=1.0)


class BaselineCriticalTests(unittest.TestCase):
    def test_baseline_fires_when_customer_config_omits_weapons(self):
        # Customer only cares about loitering; a weapon must STILL alert.
        engine = CustomizationEngine("configs/shelf_zones_demo.json", baseline_path=BASELINE)
        alerts = engine.evaluate([_weapon_event()])
        names = {a.rule_name for a in alerts}
        self.assertIn("baseline_weapon", names)

    def test_baseline_only_without_customer_config(self):
        engine = CustomizationEngine(None, baseline_path=BASELINE)
        self.assertTrue(engine.has_rules())
        alerts = engine.evaluate([_weapon_event()])
        self.assertTrue(any(a.rule_name == "baseline_weapon" for a in alerts))
        self.assertEqual(alerts[0].priority, "critical")

    def test_no_baseline_flag_leaves_only_customer_rules(self):
        engine = CustomizationEngine("configs/shelf_zones_demo.json")  # no baseline
        self.assertEqual(engine.baseline_rules, [])
        alerts = engine.evaluate([_weapon_event()])
        self.assertFalse(any(a.rule_name.startswith("baseline_") for a in alerts))


if __name__ == "__main__":
    unittest.main()
