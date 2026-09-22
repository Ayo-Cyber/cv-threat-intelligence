"""A saved English rule is visible in the camera list the app renders.

The Rules tab draws `camera.custom_rules`; the API's camera list never
carried that field, so every sentence looked unsaved the moment the panel
refreshed -- while the engine was in fact scanning for it. Pilot report,
Windows, 22 Sep: "describe in English doesn't save". Held here as an HTTP
round trip through the real backend: write, list, remove, list.
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
from cvti.api.sources import _custom_rules


class CameraListShowsEnglishRules(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # the backend writes configs/zones and configs/rules relative to the
        # CWD when not frozen -- keep that out of the repo
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        site = self.tmp / "site.json"
        site.write_text('{"name": "rules-test", "notify": "console", "cameras": []}')
        self.client = TestClient(create_app(db_path=str(self.tmp / "events.db"),
                                            site_path=str(site)))
        self.client.post("/api/v1/auth/first-owner",
                         json={"username": "ayo", "password": "pw-123456"})
        r = self.client.post("/api/v1/auth/session",
                             json={"username": "ayo", "password": "pw-123456"})
        self.owner = {"Authorization": f"Bearer {r.json()['token']}"}
        made = self.client.post("/api/v1/cameras", headers=self.owner,
                                json={"camera": {"id": "gate", "source": "demo"}})
        self.assertIn(made.status_code, (200, 201), made.text)

    def tearDown(self):
        os.chdir(self._cwd)

    def _cameras(self) -> dict:
        r = self.client.get("/api/v1/cameras", headers=self.owner)
        self.assertEqual(r.status_code, 200, r.text)
        return {c["id"]: c for c in r.json()}

    def test_write_list_remove_round_trip(self):
        self.assertEqual(self._cameras()["gate"]["custom_rules"], [])

        added = self.client.post("/api/v1/cameras/gate/rules/custom", headers=self.owner,
                                 json={"question": "Is anyone climbing the gate?",
                                       "dwell": 4})
        self.assertEqual(added.status_code, 201, added.text)

        rules = self._cameras()["gate"]["custom_rules"]
        self.assertEqual(rules, [{"question": "Is anyone climbing the gate?",
                                  "dwell": 4.0}])

        gone = self.client.delete(
            "/api/v1/cameras/gate/rules/custom/Is%20anyone%20climbing%20the%20gate%3F",
            headers=self.owner)
        self.assertIn(gone.status_code, (200, 204), gone.text)
        self.assertEqual(self._cameras()["gate"]["custom_rules"], [])

    def test_two_sentences_accumulate(self):
        for q in ("Is anyone climbing the gate?", "Is a ladder against the fence?"):
            r = self.client.post("/api/v1/cameras/gate/rules/custom", headers=self.owner,
                                 json={"question": q, "dwell": 4})
            self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual([r["question"] for r in self._cameras()["gate"]["custom_rules"]],
                         ["Is anyone climbing the gate?", "Is a ladder against the fence?"])


class TheNormaliserMatchesTheBackend(unittest.TestCase):
    def test_legacy_single_rule_is_listed_first(self):
        cam = {"custom_rule": {"question": "old one?", "dwell": 3},
               "custom_rules": [{"question": "new one?", "dwell": 4}]}
        self.assertEqual(_custom_rules(cam),
                         [{"question": "old one?", "dwell": 3.0},
                          {"question": "new one?", "dwell": 4.0}])

    def test_blanks_and_junk_are_dropped(self):
        cam = {"custom_rules": [{"question": "  "}, "not a rule", {"dwell": 2},
                                {"question": " real? ", "dwell": None}]}
        self.assertEqual(_custom_rules(cam), [{"question": "real?", "dwell": 0.0}])

    def test_no_rules_is_an_empty_list(self):
        self.assertEqual(_custom_rules({}), [])


if __name__ == "__main__":
    unittest.main()
