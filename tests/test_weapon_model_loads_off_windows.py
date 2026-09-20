"""The weapon checkpoint must load on the platforms we actually ship to.

models/weapon_best.pt was pickled on Windows, so it carries a WindowsPath.
Unpickling one anywhere else raises

    NotImplementedError: cannot instantiate 'WindowsPath' on your system

which cvti/serving/pipeline.py caught, logged as "weapon model unavailable
... weapons disabled", and carried on. Nothing surfaced it: the weapons KPI
simply never fired on any macOS or Linux install, including every demo run on
a developer Mac. Found 20 Sep by running each KPI in isolation and reading the
engine log rather than the alert count.
"""
from __future__ import annotations

import pathlib
import unittest


class TheLoaderSurvivesAWindowsPickledCheckpoint(unittest.TestCase):
    def test_it_aliases_windowspath_for_the_load(self):
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "cvti" / "detector" / "core.py").read_text()
        loader = source[source.index("def load_yolov5_model"):
                        source.index("def load_detection_model")]
        self.assertIn("pathlib.WindowsPath", loader,
                      "the yolov5 loader must handle a Windows-pickled checkpoint")
        self.assertIn("pathlib.PosixPath", loader)

    def test_it_puts_the_class_back(self):
        """A permanent alias would corrupt every later WindowsPath use."""
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "cvti" / "detector" / "core.py").read_text()
        loader = source[source.index("def load_yolov5_model"):
                        source.index("def load_detection_model")]
        self.assertIn("finally:", loader,
                      "the alias must be restored even when the load raises")
        self.assertIn("restore", loader)

    def test_the_real_checkpoint_loads_here(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        weights = root / "models" / "weapon_best.pt"
        repo = root / "external" / "yolov5"
        if not weights.is_file() or not repo.is_dir():
            self.skipTest("weapon weights or the yolov5 repo are not present")
        from cvti.detector.core import load_detection_model
        model = load_detection_model(str(weights), str(repo), preferred_kind="yolov5")
        self.assertTrue(set(model.names.values()) >= {"gun", "knife"},
                        f"expected gun/knife, got {sorted(model.names.values())}")
        self.assertIs(pathlib.WindowsPath, pathlib.WindowsPath,
                      "WindowsPath must be its own class again after loading")


if __name__ == "__main__":
    unittest.main()
