from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cvti.serving import onboarding


class LocationHierarchyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_site(self, data: dict) -> Path:
        site = Path(self._tmp.name) / "site.json"
        site.write_text(json.dumps(data))
        return site

    def test_legacy_site_gets_stable_virtual_organization_and_branch(self):
        site = self.write_site({
            "name": "Plant One",
            "cameras": [{"id": "cam1", "source": "0"}],
        })

        first = onboarding.normalized_hierarchy(site)
        second = onboarding.normalized_hierarchy(site)

        self.assertEqual(first, second)
        self.assertEqual(first["organization"]["id"], "organization--default")
        self.assertEqual(first["branches"][0]["id"], "branch--default")
        self.assertEqual(
            first["branches"][0]["areas"][0]["cameras"][0]["id"], "cam1"
        )
        self.assertNotIn("organization", json.loads(site.read_text()))

    def test_area_must_reference_a_known_branch(self):
        site = self.write_site({
            "branches": [{"id": "lagos", "name": "Lagos"}],
            "cameras": [],
        })

        with self.assertRaisesRegex(ValueError, "unknown branch: missing"):
            onboarding.upsert_area(site, {
                "id": "paint",
                "name": "Paint floor",
                "branch_id": "missing",
            })

    def test_upsert_branch_persists_legacy_defaults_atomically(self):
        site = self.write_site({
            "name": "Plant One",
            "notify": "console",
            "areas": [{"id": "paint", "name": "Paint floor"}],
            "cameras": [{"id": "cam1", "source": "0"}],
        })

        branches = onboarding.upsert_branch(
            site, {"id": "warehouse", "name": "Warehouse"}
        )

        saved = json.loads(site.read_text())
        self.assertEqual(saved["organization"], {
            "id": "organization--default",
            "name": "Plant One",
        })
        self.assertEqual(branches, [
            {"id": "branch--default", "name": "Main branch"},
            {"id": "warehouse", "name": "Warehouse"},
        ])
        self.assertEqual(saved["areas"][0]["branch_id"], "branch--default")
        self.assertEqual(saved["notify"], "console")
        self.assertFalse(site.with_suffix(".json.tmp").exists())

    def test_camera_derives_branch_from_area(self):
        site = self.write_site({
            "organization": {"id": "org", "name": "Customer"},
            "branches": [{"id": "lagos", "name": "Lagos"}],
            "areas": [{
                "id": "paint",
                "name": "Paint floor",
                "branch_id": "lagos",
            }],
            "cameras": [{
                "id": "cam1",
                "source": "0",
                "area_id": "paint",
                "branch_id": "wrong-camera-value",
            }],
        })

        hierarchy = onboarding.normalized_hierarchy(site)
        camera = hierarchy["branches"][0]["areas"][0]["cameras"][0]

        self.assertEqual(camera["area_id"], "paint")
        self.assertEqual(camera["branch_id"], "lagos")

    def test_remove_nonempty_branch_raises_hierarchy_conflict(self):
        site = self.write_site({
            "branches": [{"id": "lagos", "name": "Lagos"}],
            "areas": [{
                "id": "paint",
                "name": "Paint floor",
                "branch_id": "lagos",
            }],
            "cameras": [],
        })
        original = site.read_text()

        with self.assertRaisesRegex(
            onboarding.HierarchyConflict, "branch contains areas"
        ):
            onboarding.remove_branch(site, "lagos")

        self.assertEqual(site.read_text(), original)

    def test_remove_empty_branch_succeeds(self):
        site = self.write_site({
            "organization": {"id": "org", "name": "Customer"},
            "branches": [
                {"id": "lagos", "name": "Lagos"},
                {"id": "abuja", "name": "Abuja"},
            ],
            "areas": [],
            "cameras": [],
        })

        branches = onboarding.remove_branch(site, "abuja")

        self.assertEqual(branches, [{"id": "lagos", "name": "Lagos"}])
        self.assertEqual(json.loads(site.read_text())["branches"], branches)

    def test_unknown_area_camera_is_returned_unassigned(self):
        site = self.write_site({
            "branches": [{"id": "lagos", "name": "Lagos"}],
            "areas": [{
                "id": "paint",
                "name": "Paint floor",
                "branch_id": "lagos",
            }],
            "cameras": [{
                "id": "cam1",
                "source": "0",
                "area_id": "missing",
                "branch_id": "stale",
            }],
        })

        hierarchy = onboarding.normalized_hierarchy(site)

        self.assertEqual(
            [camera["id"] for camera in hierarchy["unassigned_cameras"]],
            ["cam1"],
        )
        self.assertNotIn("branch_id", hierarchy["unassigned_cameras"][0])
        self.assertEqual(
            [area["id"] for area in hierarchy["branches"][0]["areas"]],
            ["paint"],
        )

    def test_unknown_area_camera_keeps_legacy_implicit_area_listing(self):
        site = self.write_site({
            "cameras": [{
                "id": "cam1",
                "source": "0",
                "area_id": "missing",
                "branch_id": "stale",
            }],
        })

        areas = onboarding.normalized_areas(site)

        self.assertEqual(areas, [{
            "id": "missing", "name": "cam1", "implicit": True,
            "camera_ids": ["cam1"],
        }])

    def test_unresolved_area_id_does_not_hide_colliding_implicit_camera(self):
        site = self.write_site({
            "cameras": [
                {
                    "id": "bad",
                    "source": "0",
                    "area_id": "camera--good",
                    "branch_id": "stale",
                },
                {"id": "good", "source": "1"},
            ],
        })

        hierarchy = onboarding.normalized_hierarchy(site)

        self.assertEqual(
            [camera["id"] for camera in hierarchy["unassigned_cameras"]],
            ["bad"],
        )
        self.assertEqual(
            [area["id"] for area in hierarchy["branches"][0]["areas"]],
            ["camera--good"],
        )
        self.assertEqual(
            hierarchy["branches"][0]["areas"][0]["cameras"][0]["id"],
            "good",
        )

    def test_set_organization_persists_normalized_hierarchy(self):
        site = self.write_site({
            "name": "Plant One",
            "areas": [{"id": "paint", "name": "Paint floor"}],
            "cameras": [{"id": "cam1", "source": "0"}],
        })

        organization = onboarding.set_organization(
            site, {"id": "customer", "name": "Customer Ltd"}
        )

        saved = json.loads(site.read_text())
        self.assertEqual(organization, {"id": "customer", "name": "Customer Ltd"})
        self.assertEqual(saved["organization"], organization)
        self.assertEqual(saved["branches"], [
            {"id": "branch--default", "name": "Main branch"},
        ])
        self.assertEqual(saved["areas"][0]["branch_id"], "branch--default")

    def test_set_organization_rejects_empty_id(self):
        site = self.write_site({"name": "Plant One", "cameras": []})
        original = site.read_text()

        with self.assertRaisesRegex(ValueError, "organization id and name"):
            onboarding.set_organization(site, {"id": " ", "name": "Customer"})

        self.assertEqual(site.read_text(), original)

    def test_duplicate_branch_ids_are_rejected(self):
        site = self.write_site({
            "branches": [
                {"id": "lagos", "name": "Lagos"},
                {"id": "lagos", "name": "Duplicate"},
            ],
            "cameras": [],
        })

        with self.assertRaisesRegex(ValueError, "duplicate branch id: lagos"):
            onboarding.normalized_branches(site)

    def test_upsert_branch_rejects_empty_id(self):
        site = self.write_site({"name": "Plant One", "cameras": []})
        original = site.read_text()

        with self.assertRaisesRegex(ValueError, "branch id and name"):
            onboarding.upsert_branch(site, {"id": " ", "name": "Lagos"})

        self.assertEqual(site.read_text(), original)

if __name__ == "__main__":
    unittest.main()
