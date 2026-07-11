from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.contracts import CandidateAlert
from cvti.verification.gate import VerificationResult, _save_artifacts


def _alert() -> CandidateAlert:
    return CandidateAlert(
        rule_name="shoplifting",
        priority="high",
        detector="concealment",
        title="POSSIBLE CONCEALMENT",
        person_id=1,
        object_label=None,
        timestamp=0.0,
    )


def _result() -> VerificationResult:
    return VerificationResult(
        confirmed=False,
        confidence=0.0,
        reason="test",
        alert_priority="high",
        timestamp="2026-07-04T00:00:00Z",
    )


class VerificationGateArtifactTests(unittest.TestCase):
    def test_save_artifacts_removes_stale_frame_files(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            _save_artifacts(tmp_path, 1, [frame, frame, frame], _alert(), _result(), "{}")
            self.assertTrue((tmp_path / "gate_0001" / "frame_0.jpg").exists())
            self.assertTrue((tmp_path / "gate_0001" / "frame_1.jpg").exists())
            self.assertTrue((tmp_path / "gate_0001" / "frame_2.jpg").exists())

            _save_artifacts(tmp_path, 1, [frame], _alert(), _result(), "{}")

            self.assertTrue((tmp_path / "gate_0001" / "frame.jpg").exists())
            self.assertFalse((tmp_path / "gate_0001" / "frame_0.jpg").exists())
            self.assertFalse((tmp_path / "gate_0001" / "frame_1.jpg").exists())
            self.assertFalse((tmp_path / "gate_0001" / "frame_2.jpg").exists())


if __name__ == "__main__":
    unittest.main()
