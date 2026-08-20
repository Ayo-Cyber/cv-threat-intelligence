"""The weapons/violence measurement harness (EP-07-T3).

The datasets are not acquired yet — that is the epic's blocker — but the
machinery must be proven BEFORE clips land, so measuring becomes one command
instead of a week of glue work at pilot time. Stub clips stand in for footage;
nothing here loads torch or calls a VLM.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from cvti.eval import dataset as ds

ROOT = Path(__file__).resolve().parents[1]


def _mp4(path: Path, frames: int = 3):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    for _ in range(frames):
        w.write(np.zeros((24, 32, 3), np.uint8))
    w.release()


class CriticalIngestTest(unittest.TestCase):
    def test_curated_layout_maps_kind_label_and_expectations(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _mp4(base / "weapons" / "threat" / "gun1.mp4")
            _mp4(base / "weapons" / "normal" / "calm1.mp4")
            _mp4(base / "violence" / "threat" / "fight1.mp4")
            clips = ds._critical_clips(base)
            by = {c.name: c for c in clips}
            self.assertTrue(by["gun1.mp4"].is_threat)
            self.assertEqual(by["gun1.mp4"].kind, "weapons")
            self.assertEqual(by["gun1.mp4"].expects, ("weapons",))
            self.assertFalse(by["calm1.mp4"].is_threat)
            self.assertEqual(by["calm1.mp4"].kind, "", "a normal clip has no threat kind")
            self.assertEqual(by["fight1.mp4"].expects, ("violence",))

    def test_ucf_layout_maps_conservatively(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _mp4(base / "Fighting" / "Fighting001_x264.mp4")
            _mp4(base / "Assault" / "Assault001_x264.mp4")
            _mp4(base / "Shooting" / "Shooting001_x264.mp4")
            _mp4(base / "Robbery" / "Robbery001_x264.mp4")          # NOT weapons
            _mp4(base / "Arson" / "Arson001_x264.mp4")              # out of scope
            _mp4(base / "Normal_Videos_event" / "Normal001_x264.mp4")
            clips = ds._ucf_crime_clips(base)
            kinds = {c.name: (c.is_threat, c.kind) for c in clips}
            self.assertEqual(kinds["Fighting001_x264.mp4"], (True, "violence"))
            self.assertEqual(kinds["Assault001_x264.mp4"], (True, "violence"))
            self.assertEqual(kinds["Shooting001_x264.mp4"], (True, "weapons"))
            self.assertEqual(kinds["Normal001_x264.mp4"], (False, ""))
            self.assertNotIn("Robbery001_x264.mp4", kinds,
                             "Robbery does not certify a visible weapon — counting "
                             "it would punish the detector for the dataset's label")
            self.assertNotIn("Arson001_x264.mp4", kinds)

    def test_kind_filter_keeps_that_threat_plus_every_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _mp4(base / "weapons" / "threat" / "gun1.mp4")
            _mp4(base / "violence" / "threat" / "fight1.mp4")
            _mp4(base / "violence" / "normal" / "calm1.mp4")
            real = ds.CRITICAL_CLIPS
            ds.CRITICAL_CLIPS = base
            try:
                clips = ds.load_dataset("critical", kind="weapons")
            finally:
                ds.CRITICAL_CLIPS = real
            names = sorted(c.name for c in clips)
            self.assertEqual(names, ["calm1.mp4", "gun1.mp4"],
                             "a violence clip must not count against the weapons eval")


class ClipFloorTest(unittest.TestCase):
    """The >=50 guard: small-n rates are noise wearing a percent sign."""

    def _tool(self, *argv, critical_dir):
        env_patch = (
            "import sys; sys.path.insert(0, r'%s'); "
            "from cvti.eval import dataset as ds; from pathlib import Path; "
            "ds.CRITICAL_CLIPS = Path(r'%s'); ds.UCF_CRIME = Path(r'%s/none'); "
            "sys.argv = ['measure_critical'] + %r; "
            "import runpy; runpy.run_path(r'%s', run_name='__main__')"
        ) % (ROOT, critical_dir, critical_dir, list(argv),
             ROOT / "tools" / "measure_critical.py")
        return subprocess.run([sys.executable, "-c", env_patch],
                              capture_output=True, text=True)

    def test_a_run_below_the_floor_is_refused_with_the_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            _mp4(Path(tmp) / "weapons" / "threat" / "gun1.mp4")
            r = self._tool("run", "weapons", critical_dir=tmp)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("below the publishable floor", r.stdout)
            self.assertIn("--smoke", r.stdout)

    def test_status_names_what_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _mp4(Path(tmp) / "violence" / "threat" / "fight1.mp4")
            r = self._tool("status", critical_dir=tmp)
            self.assertEqual(r.returncode, 1)
            self.assertIn("need >=50 threat", r.stdout)


class HarnessWiringTest(unittest.TestCase):
    def test_the_weapons_detector_actually_receives_its_model(self):
        # The bug this pins: detectors=("weapons",) set the flag but passed no
        # weapon_model, so the detector never fired and an eval would have
        # reported 0% recall for a detector that was never running.
        from cvti.eval.harness import EvalHarness
        h = EvalHarness(detectors=("weapons",))
        sentinel = object()
        h._weapon = sentinel
        clip = ds.EvalClip("x.mp4", True, "weapons", "test", ("weapons",))
        state = h._state_for(clip)
        self.assertIs(state.weapon_model, sentinel)
        self.assertTrue(state.weapons)

    def test_loading_is_conditional_on_the_detector_set(self):
        src = (ROOT / "cvti" / "eval" / "harness.py").read_text()
        self.assertIn('if "weapons" in self.detectors', src)


if __name__ == "__main__":
    unittest.main()
