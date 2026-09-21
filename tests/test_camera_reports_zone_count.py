"""Every camera the API returns says how many zones it has.

Loitering, intrusion and restricted-area alerts only exist inside a zone:
PerCameraState.process gates every presence event on zone_monitor being set,
and the shipped use-case templates define no zone rules at all. So a fresh
site has no loitering until someone draws a polygon -- and until 21 Sep
nothing on screen said so. A pilot's "loitering isn't working" was a camera
with no zone. The UI now warns from this number.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cvti.api.sources import _zone_count


class ZoneCount(unittest.TestCase):
    def test_counts_polygons_in_the_zones_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "cam.json"
            f.write_text(json.dumps({"zones": [
                {"name": "a", "polygon": [[0, 0], [1, 0], [1, 1]]},
                {"name": "b", "polygon": [[0, 0], [2, 0], [2, 2]]},
                {"name": "broken", "polygon": []},          # not a zone
            ]}))
            self.assertEqual(_zone_count(str(f)), 2)

    def test_no_zones_file_means_zero(self):
        self.assertEqual(_zone_count(None), 0)
        self.assertEqual(_zone_count(""), 0)
        self.assertEqual(_zone_count("/nowhere/at/all.json"), 0)

    def test_an_unreadable_file_is_zero_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "cam.json"
            f.write_text("{not json")
            self.assertEqual(_zone_count(str(f)), 0)


if __name__ == "__main__":
    unittest.main()
