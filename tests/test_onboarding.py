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
        self.assertEqual(next(c for c in cams if c["id"] == "front")["source"], "rtsp://a/CHANGED")
        # remove
        onboarding.remove_camera(self.site, "front")
        self.assertEqual([c["id"] for c in onboarding.list_cameras(self.site)], ["back"])
        # file is valid JSON the pipeline can read
        self.assertIn("cameras", json.loads(Path(self.site).read_text()))

    def test_add_requires_source(self):
        with self.assertRaises(ValueError):
            onboarding.add_camera(self.site, {"id": "x"})

    def test_auto_id_when_missing(self):
        onboarding.add_camera(self.site, {"source": "rtsp://a/1"})
        self.assertTrue(onboarding.list_cameras(self.site)[0]["id"].startswith("cam"))

    def test_complete_first_run_never_stamps_a_strict_scene_policy(self):
        """Repinned 1 Sep: stamping fresh sites 'require_reviewed' turned a
        pilot's first run into a two-day watchdog loop — mapping timed out on
        his hardware, the strict policy blocked every camera, the engine
        exited, repeat. New sites keep the 'auto' default; strict policies
        are an explicit operator choice, never a wizard side effect."""
        onboarding.add_camera(self.site, {"id": "front", "source": "rtsp://a/1"})

        result = onboarding.complete_first_run(self.site)

        saved = json.loads(Path(self.site).read_text())
        self.assertTrue(result["configured"])
        self.assertNotIn("scene_context_policy", saved)

    def test_reopened_wizard_does_not_rewrite_legacy_scene_context_policy(self):
        Path(self.site).write_text(json.dumps({
            "name": "Legacy Site",
            "configured": True,
            "cameras": [{"id": "front", "source": "rtsp://a/1"}],
        }))

        onboarding.complete_first_run(self.site)

        saved = json.loads(Path(self.site).read_text())
        self.assertNotIn("scene_context_policy", saved)


if __name__ == "__main__":
    unittest.main()
