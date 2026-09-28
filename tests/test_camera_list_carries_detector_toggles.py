"""The camera list carries every detector toggle's state.

Reported 25 Sep: "the toggles for the detections are working but it's not
turning green when clicked". The switch reads camera[key] for its position,
and the API's camera list carried none of the thirteen flags — so the switch
was always grey however the site file was configured.

The cosmetic half is the lesser one. The tab also sends `not camera[key]`
when a switch is clicked, so with the key absent every click computed
`not None` and sent True: a detector could be switched ON and never OFF.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from cvti.api.app import create_app
from cvti.api.sources import DETECTOR_FLAGS, _detectors


class TheFlagsSurviveTheRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        site = self.tmp / "site.json"
        site.write_text('{"name": "t", "notify": "console", "cameras": []}')
        self.client = TestClient(create_app(db_path=str(self.tmp / "events.db"),
                                            site_path=str(site)))
        self.client.post("/api/v1/auth/first-owner",
                         json={"username": "ayo", "password": "pw-123456"})
        r = self.client.post("/api/v1/auth/session",
                             json={"username": "ayo", "password": "pw-123456"})
        self.owner = {"Authorization": f"Bearer {r.json()['token']}"}
        self.client.post("/api/v1/cameras", headers=self.owner,
                         json={"camera": {"id": "gate", "source": "demo"}})

    def tearDown(self):
        os.chdir(self._cwd)

    def _camera(self) -> dict:
        r = self.client.get("/api/v1/cameras", headers=self.owner)
        self.assertEqual(r.status_code, 200, r.text)
        return {c["id"]: c for c in r.json()}["gate"]

    def test_every_toggle_is_present_and_explicitly_false(self):
        cam = self._camera()
        for flag in DETECTOR_FLAGS:
            self.assertIn(flag, cam, f"{flag} missing — its switch cannot draw")
            self.assertIs(cam[flag], False)

    def test_a_detector_switched_on_reads_back_on(self):
        r = self.client.put("/api/v1/cameras/gate/rules", headers=self.owner,
                            json={"rules": {"fire_smoke": True}})
        self.assertIn(r.status_code, (200, 201), r.text)
        self.assertIs(self._camera()["fire_smoke"], True)

    def test_a_detector_can_be_switched_off_again(self):
        """The serious half: with the key absent the tab sent `not None` =
        True on every click, so nothing could ever be turned off."""
        self.client.put("/api/v1/cameras/gate/rules", headers=self.owner,
                        json={"rules": {"fire_smoke": True}})
        self.assertIs(self._camera()["fire_smoke"], True)
        self.client.put("/api/v1/cameras/gate/rules", headers=self.owner,
                        json={"rules": {"fire_smoke": False}})
        self.assertIs(self._camera()["fire_smoke"], False)


class TheHelperIsExplicit(unittest.TestCase):
    def test_absent_means_false_not_missing(self):
        self.assertEqual(_detectors({}), {f: False for f in DETECTOR_FLAGS})

    def test_truthy_site_values_become_real_booleans(self):
        got = _detectors({"fire_smoke": 1, "violence": "yes", "theft": 0})
        self.assertIs(got["fire_smoke"], True)
        self.assertIs(got["violence"], True)
        self.assertIs(got["theft"], False)

    def test_it_matches_what_the_interface_offers(self):
        """DETECTOR_FLAGS must track the UI's DETECTORS list."""
        ui = Path(__file__).resolve().parents[1] / "Frontend/src/lib/types.ts"
        if not ui.exists():
            self.skipTest("frontend not present")
        import re
        keys = set(re.findall(r'key: "([a-z_]+)"', ui.read_text(encoding="utf-8")))
        self.assertTrue(keys, "no detector keys found in the UI")
        self.assertEqual(keys - set(DETECTOR_FLAGS), set(),
                         "the interface offers a toggle the API never returns")


if __name__ == "__main__":
    unittest.main()
