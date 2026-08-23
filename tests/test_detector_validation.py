"""Detector validation status at the point of configuration (EP-07-T4).

The product markets ten capabilities and can evidence three-ish. The table
the UI reads must cover every detector flag, say EXPERIMENTAL in so many
words for the unmeasured ones, and agree with docs/NUMBERS.md — the
committed claims sheet — so the interface cannot outrun the evidence.
"""
import re
import tempfile
import unittest
from pathlib import Path

from _backend_helper import signed_in

ROOT = Path(__file__).resolve().parents[1]


def _backend(tmp):
    root = Path(tmp)
    (root / "site.json").write_text('{"cameras": []}')
    return signed_in("owner", site_path=str(root / "site.json"),
                     db_path=str(root / "events.db"), enable_demo=False)


class ValidationTableTest(unittest.TestCase):
    def test_every_detector_flag_has_a_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            be = _backend(tmp)
            v = be.detector_validation()
            self.assertEqual(sorted(v), sorted(be.RULE_FLAGS))

    def test_unmeasured_detectors_say_experimental_in_so_many_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            be = _backend(tmp)
            for flag, row in be.detector_validation().items():
                if not row["measured"]:
                    self.assertIn("EXPERIMENTAL", row["summary"], flag)
                else:
                    self.assertRegex(row["summary"], r"n=\d+",
                                     f"{flag} claims measurement without a sample size")

    def test_the_table_agrees_with_the_published_numbers_sheet(self):
        # NUMBERS.md is the claims sheet. Anything it lists as measured must be
        # measured here, and anything 'not yet validated' must not be.
        text = (ROOT / "docs" / "NUMBERS.md").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            be = _backend(tmp)
            v = be.detector_validation()
        sheet = {"fire_smoke": "Fire / smoke", "crowd_formation": "Crowd forming",
                 "concealment": "Theft / concealment", "weapons": "Weapons",
                 "violence": "Violence / assault", "running": "Panic running",
                 "fall": "Person collapsed", "tamper": "Camera tampering"}
        for flag, label in sheet.items():
            row = re.search(rf"\|\s*{re.escape(label)}\s*\|([^\n]*)\|", text)
            self.assertIsNotNone(row, f"NUMBERS.md lost its row for {label}")
            published_measured = "✅ measured" in row.group(1)
            self.assertEqual(v[flag]["measured"], published_measured,
                             f"{flag}: UI says measured={v[flag]['measured']}, "
                             f"NUMBERS.md says {row.group(1).strip()!r}")

    def test_the_weapons_and_violence_gap_is_still_declared(self):
        # EP-07-T3 is blocked on data; until it lands, the two critical
        # always-on detectors MUST read as unmeasured. If this fails because
        # they were measured — delete it with pleasure.
        with tempfile.TemporaryDirectory() as tmp:
            be = _backend(tmp)
            v = be.detector_validation()
        self.assertFalse(v["weapons"]["measured"])
        self.assertFalse(v["violence"]["measured"])


if __name__ == "__main__":
    unittest.main()
