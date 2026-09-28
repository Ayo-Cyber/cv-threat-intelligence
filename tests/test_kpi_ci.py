"""The Windows KPI job's two scripts: the API-driven site builder and the
delivery verdict. Synthetic clips, no engine, no network."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "e2e"))

import kpi_assert  # noqa: E402
import kpi_site  # noqa: E402


def _tiny_clip(path: Path, w: int = 96, h: int = 64, frames: int = 6) -> None:
    import cv2
    import numpy as np
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (w, h))
    for _ in range(frames):
        out.write(np.zeros((h, w, 3), dtype=np.uint8))
    out.release()


class SiteBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._cwd = os.getcwd()
        os.chdir(self.tmp)   # generated rules/zones land under configs/ of the CWD
        self.clips = self.tmp / "clips"
        self.clips.mkdir()
        for c in kpi_site.CAMERAS:
            _tiny_clip(self.clips / c["clip"])

    def tearDown(self):
        os.chdir(self._cwd)

    def test_every_camera_is_configured_through_the_api_with_its_rules(self):
        summary = kpi_site.build(self.clips, self.tmp / "out")
        by_id = {c["id"]: c for c in summary["cameras"]}
        self.assertEqual(set(by_id), {c["id"] for c in kpi_site.CAMERAS})
        person = by_id["kpi1_person"]
        self.assertTrue(person["zones"])
        self.assertIn("person_entered", person["rules"])
        self.assertIn("person_exited", person["rules"])
        for vid in ("kpi2_vehicle_a", "kpi2_vehicle_b"):
            v = by_id[vid]
            self.assertIsNotNone(v["vehicle_line"])
            self.assertTrue(v["vehicle_line"]["normalized"])
            self.assertIn("vehicle_entered", v["rules"])
            self.assertIn("vehicle_exited", v["rules"])
        self.assertEqual(by_id["kpi2_vehicle_a"]["vehicle_line"]["flip"], True)
        # the zone is in ORIGINAL pixels of the 96x64 clip: x from 0.54*96
        site = json.loads(Path(summary["site"]).read_text())
        cam = next(c for c in site["cameras"] if c["id"] == "kpi1_person")
        zones = json.loads(Path(cam["zones"]).read_text())["zones"]
        self.assertEqual(zones[0]["polygon"][0], [51, 0])

    def test_the_token_never_lands_in_the_site_file(self):
        summary = kpi_site.build(self.clips, self.tmp / "out")
        self.assertEqual(json.loads(Path(summary["site"]).read_text())["notify"], "console")

    def test_a_missing_clip_is_named(self):
        (self.clips / "intrusion_23.mp4").unlink()
        with self.assertRaises(SystemExit) as cm:
            kpi_site.build(self.clips, self.tmp / "out")
        self.assertIn("intrusion_23.mp4", str(cm.exception))


LOG = """
INFO cvti.serving.alert_sink — [NOTIFY] kpi1_person :: person_entered (HIGH) — x
INFO cvti.serving.alert_sink — [notify telegram] delivered person_entered on kpi1_person to chat 111 (3 photo(s), video)
INFO cvti.serving.alert_sink — [notify telegram] delivered person_entered on kpi1_person to chat -222 (3 photo(s), video)
INFO cvti.serving.alert_sink — [NOTIFY] kpi1_person :: person_exited (MEDIUM) — x
INFO cvti.serving.alert_sink — [notify telegram] delivered person_exited on kpi1_person to chat 111 (video)
INFO cvti.serving.alert_sink — [NOTIFY] kpi2_vehicle_a :: vehicle_entered (HIGH) — x
INFO cvti.serving.alert_sink — [notify telegram] delivered vehicle_entered on kpi2_vehicle_a to chat 111 (video)
INFO cvti.serving.alert_sink — [NOTIFY] kpi2_vehicle_b :: vehicle_exited (MEDIUM) — x
INFO cvti.serving.alert_sink — [notify telegram] delivered vehicle_exited on kpi2_vehicle_b to chat -222 (video)
WARNING cvti.serving.alert_sink — [notify telegram] 3 alert(s) still queued for chat 111 at shutdown
"""


class VerdictTest(unittest.TestCase):
    def test_passes_when_every_kind_and_every_chat_got_a_delivery(self):
        ok, report = kpi_assert.verdict(LOG, ["111", "-222"])
        self.assertTrue(ok, report)
        self.assertIn("VERDICT: PASS", report)
        self.assertIn("3 alert(s) were still queued", report)

    def test_a_raised_but_undelivered_kind_fails(self):
        text = LOG.replace("delivered vehicle_exited on kpi2_vehicle_b to chat -222", "delivered nothing")
        ok, report = kpi_assert.verdict(text, ["111"])
        self.assertFalse(ok)
        self.assertIn("vehicle_exited", report)
        self.assertIn("none delivered", report)

    def test_a_chat_that_received_nothing_fails(self):
        ok, report = kpi_assert.verdict(LOG, ["111", "-222", "999"])
        self.assertFalse(ok)
        self.assertIn("chat 999", report)
        self.assertIn("nothing reached this chat", report)

    def test_notify_alone_is_not_delivery(self):
        text = "\n".join(l for l in LOG.splitlines() if "delivered" not in l)
        ok, _ = kpi_assert.verdict(text, ["111"])
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
