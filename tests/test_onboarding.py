from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving import onboarding


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.site = str(Path(self._tmp.name) / "site.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_add_list_upsert_remove(self):
        self.assertEqual(onboarding.list_cameras(self.site), [])          # missing file -> empty
        onboarding.add_camera(self.site, {"id": "front", "source": "rtsp://a/1", "concealment": True})
        onboarding.add_camera(self.site, {"id": "back", "source": "rtsp://b/1"})
        cams = onboarding.list_cameras(self.site)
        self.assertEqual([c["id"] for c in cams], ["front", "back"])
        # upsert by id (not a duplicate)
        onboarding.add_camera(self.site, {"id": "front", "source": "rtsp://a/CHANGED"})
        cams = onboarding.list_cameras(self.site)
        self.assertEqual(len(cams), 2)
        self.assertEqual([c["id"] for c in cams], ["front", "back"])
        self.assertEqual(next(c for c in cams if c["id"] == "front")["source"], "rtsp://a/CHANGED")
        # remove
        onboarding.remove_camera(self.site, "front")
        self.assertEqual([c["id"] for c in onboarding.list_cameras(self.site)], ["back"])
        # file is valid JSON the pipeline can read
        self.assertIn("cameras", json.loads(Path(self.site).read_text(encoding="utf-8")))

    def test_add_requires_source(self):
        with self.assertRaises(ValueError):
            onboarding.add_camera(self.site, {"id": "x"})

    def test_auto_id_when_missing(self):
        onboarding.add_camera(self.site, {"source": "rtsp://a/1"})
        self.assertTrue(onboarding.list_cameras(self.site)[0]["id"].startswith("cam"))

    def test_add_camera_rejects_unknown_explicit_area(self):
        onboarding.upsert_area(self.site, {"id": "floor", "name": "Factory floor"})

        with self.assertRaisesRegex(ValueError, "unknown area: warehouse"):
            onboarding.add_camera(self.site, {
                "id": "front",
                "source": "rtsp://a/1",
                "area_id": "warehouse",
            })

    def test_add_camera_to_derived_area_from_hierarchy(self):
        onboarding.add_camera(self.site, {"id": "kpi9_webcam", "source": "0"})
        area = onboarding.normalized_hierarchy(self.site)["branches"][0]["areas"][0]
        self.assertEqual(area["id"], "camera--kpi9_webcam")
        self.assertTrue(area["implicit"])
        onboarding.add_camera(self.site, {
            "id": "Test kpi9", "source": "0", "area_id": area["id"],
        })
        saved = onboarding.load_site(self.site)
        self.assertEqual(saved["areas"], [{
            "id": area["id"], "name": "kpi9_webcam",
            "branch_id": onboarding.DEFAULT_BRANCH_ID,
        }])
        group = onboarding.normalized_hierarchy(self.site)["branches"][0]["areas"][0]
        self.assertEqual({c["id"] for c in group["cameras"]}, {"kpi9_webcam", "Test kpi9"})
        onboarding.add_camera(self.site, saved["cameras"][-1])
        self.assertEqual(len(onboarding.load_site(self.site)["areas"]), 1)

    def test_assign_existing_camera_to_derived_area(self):
        onboarding.add_camera(self.site, {"id": "one", "source": "0"})
        onboarding.add_camera(self.site, {"id": "two", "source": "1"})
        camera = onboarding.assign_camera_area(self.site, "two", "camera--one")
        self.assertEqual(camera["area_id"], "camera--one")

    def test_deleted_derived_area_is_rejected_without_writing(self):
        onboarding.add_camera(self.site, {"id": "one", "source": "0"})
        onboarding.remove_camera(self.site, "one")
        before = Path(self.site).read_bytes()
        with self.assertRaisesRegex(ValueError, "unknown area"):
            onboarding.add_camera(self.site, {
                "id": "two", "source": "1", "area_id": "camera--one",
            })
        self.assertEqual(Path(self.site).read_bytes(), before)

    def test_dangling_explicit_area_is_not_materialized(self):
        Path(self.site).write_text(json.dumps({"cameras": [{
            "id": "one", "source": "0", "area_id": "missing",
        }]}))
        with self.assertRaisesRegex(ValueError, "unknown area"):
            onboarding.add_camera(self.site, {
                "id": "two", "source": "1", "area_id": "missing",
            })

    def test_derived_area_stays_unassigned_in_explicit_hierarchy(self):
        Path(self.site).write_text(json.dumps({"branches": [], "cameras": [
            {"id": "one", "source": "0"},
        ]}))
        onboarding.add_camera(self.site, {
            "id": "two", "source": "1", "area_id": "camera--one",
        })
        self.assertNotIn("branch_id", onboarding.load_site(self.site)["areas"][0])

    def test_complete_first_run_never_stamps_a_strict_scene_policy(self):
        """Repinned 1 Sep: stamping fresh sites 'require_reviewed' turned a
        pilot's first run into a two-day watchdog loop — mapping timed out on
        his hardware, the strict policy blocked every camera, the engine
        exited, repeat. New sites keep the 'auto' default; strict policies
        are an explicit operator choice, never a wizard side effect."""
        onboarding.add_camera(self.site, {"id": "front", "source": "rtsp://a/1"})

        result = onboarding.complete_first_run(self.site)

        saved = json.loads(Path(self.site).read_text(encoding="utf-8"))
        self.assertTrue(result["configured"])
        self.assertNotIn("scene_context_policy", saved)

    def test_reopened_wizard_does_not_rewrite_legacy_scene_context_policy(self):
        Path(self.site).write_text(json.dumps({
            "name": "Legacy Site",
            "configured": True,
            "cameras": [{"id": "front", "source": "rtsp://a/1"}],
        }))

        onboarding.complete_first_run(self.site)

        saved = json.loads(Path(self.site).read_text(encoding="utf-8"))
        self.assertNotIn("scene_context_policy", saved)


if __name__ == "__main__":
    unittest.main()
