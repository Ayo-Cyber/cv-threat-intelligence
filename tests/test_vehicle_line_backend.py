"""An installer can set the vehicle tripwire without editing JSON.

KPI 2 on the customer's sheet is "Vehicle entering/exiting", and until now
the only way to configure it was to hand-edit site.json on the machine:
no UI, no API route, no backend method. Three cameras were set up that way
on 28 Sep to get the alerts onto a phone, which is not something a site
installer can be asked to do.

The rules matter as much as the geometry. No shipped preset listens for
vehicle_entry, so a line without rules is a detector whose events the engine
discards in silence — the failure mode configs/baseline_critical_v1.json
documents for crowd and panic-running.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from cvti.api.app import create_app


class TheTripwireRoundTrips(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        site = self.tmp / "site.json"
        site.write_text('{"name": "t", "notify": "console", "cameras": []}')
        self.site = site
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

    def _set(self, **body):
        return self.client.post("/api/v1/cameras/gate/vehicle-line",
                                headers=self.owner, json=body)

    def _camera(self) -> dict:
        r = self.client.get("/api/v1/cameras", headers=self.owner)
        return {c["id"]: c for c in r.json()}["gate"]

    def _camera_rules(self) -> list:
        cam = [c for c in json.loads(self.site.read_text())["cameras"]
               if c["id"] == "gate"][0]
        return json.loads(Path(cam["config"]).read_text())["rules"]

    def test_a_camera_starts_with_no_line(self):
        r = self.client.get("/api/v1/cameras/gate/vehicle-line", headers=self.owner)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIsNone(r.json()["vehicle_line"])
        self.assertIsNone(self._camera()["vehicle_line"])

    def test_a_drawn_line_is_saved_and_read_back(self):
        r = self._set(start=[0.42, 0.05], end=[0.42, 0.95], flip=True)
        self.assertEqual(r.status_code, 201, r.text)
        line = self._camera()["vehicle_line"]
        self.assertEqual(line["start"], [0.42, 0.05])
        self.assertEqual(line["end"], [0.42, 0.95])
        self.assertIs(line["flip"], True)
        self.assertIs(line["normalized"], True)

    def test_fractions_and_pixels_are_told_apart_not_guessed(self):
        self._set(start=[0.42, 0.05], end=[0.42, 0.95])
        self.assertIs(self._camera()["vehicle_line"]["normalized"], True)
        self._set(start=[820, 40], end=[820, 1040])
        self.assertIs(self._camera()["vehicle_line"]["normalized"], False)

    def test_setting_a_line_wires_the_rules_that_listen_for_it(self):
        """No shipped preset carries vehicle rules; without these the engine
        emits the crossing and discards it in silence."""
        self._set(start=[0.5, 0.0], end=[0.5, 1.0])
        names = {r["name"] for r in self._camera_rules()}
        self.assertIn("vehicle_entered", names)
        self.assertIn("vehicle_exited", names)
        by_name = {r["name"]: r for r in self._camera_rules()}
        self.assertEqual(by_name["vehicle_entered"]["trigger"]["detector"], "vehicle_entry")
        self.assertEqual(by_name["vehicle_exited"]["trigger"]["detector"], "vehicle_exit")

    def test_removing_the_line_removes_its_rules_too(self):
        self._set(start=[0.5, 0.0], end=[0.5, 1.0])
        r = self.client.delete("/api/v1/cameras/gate/vehicle-line", headers=self.owner)
        self.assertIn(r.status_code, (200, 204), r.text)
        self.assertIsNone(self._camera()["vehicle_line"])
        names = {rule["name"] for rule in self._camera_rules()}
        self.assertNotIn("vehicle_entered", names)
        self.assertNotIn("vehicle_exited", names)

    def test_a_line_and_a_zone_coexist(self):
        """Drawing a gate line must not drop the loitering rules a zone wired."""
        self.client.post("/api/v1/cameras/gate/zones", headers=self.owner,
                         json={"name": "yard", "points": [[0, 0], [100, 0], [100, 100]],
                               "dwell_seconds": 5})
        self._set(start=[0.5, 0.0], end=[0.5, 1.0])
        names = {r["name"] for r in self._camera_rules()}
        self.assertIn("loitering_yard", names)
        self.assertIn("vehicle_entered", names)

    def test_a_degenerate_line_is_refused_with_a_reason(self):
        r = self._set(start=[0.5, 0.5], end=[0.5, 0.5])
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("same point", r.json()["error"]["message"])

    def test_a_malformed_line_is_refused(self):
        self.assertEqual(self._set(start=[0.5], end=[0.5, 1.0]).status_code, 400)

    def test_an_unknown_camera_is_a_named_error(self):
        r = self.client.post("/api/v1/cameras/nope/vehicle-line", headers=self.owner,
                             json={"start": [0, 0], "end": [1, 1]})
        self.assertEqual(r.status_code, 404, r.text)


if __name__ == "__main__":
    unittest.main()
