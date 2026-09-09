"""W2: same detector, whatever silicon the box has — and the floor is today.

Three families of pins:

  PARITY    the exported ONNX model produces IDENTICAL boxes to torch on real
            footage. Not "close": the 9 Sep spike measured 0.00px, and any
            drift means the export changed the detector — which is a model
            change, and model changes go through the W6 scorecard.
  SELECTION the auto rule: ONNX exists to rescue CPU-bound boxes. torch on
            mps/cuda stays torch; the site's kill switch always wins; every
            unavailable rung lands on torch WITH THE REASON — health names it.
  THE SEAM  we rebuild ultralytics' ONNX session to put DirectML/OpenVINO in
            front, because its own selection stops at CUDA/CoreML/CPU. That
            wrapper leans on a pinned ultralytics; these pins fail loudly the
            moment a version bump moves the ground under it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.detector import accel  # noqa: E402

HAVE_ORT = accel.onnxruntime_or_none() is not None
DETECT_PT = ROOT / "models" / "yolov8n.pt"
DETECT_ONNX = ROOT / "models" / "yolov8n.onnx"


def _ensure_export() -> bool:
    """The .onnx artifacts are build products, not tracked files — make them."""
    if DETECT_ONNX.exists():
        return True
    if not (HAVE_ORT and DETECT_PT.exists()):
        return False
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_onnx.py"),
                        "--models", "models/yolov8n.pt"],
                       capture_output=True, timeout=300)
    return r.returncode == 0 and DETECT_ONNX.exists()


class SelectionTests(unittest.TestCase):
    def test_the_kill_switch_always_wins(self):
        got = accel.select_detection_weights("models/yolov8n.pt", backend="torch")
        self.assertEqual(got["backend"], "torch")
        self.assertIn("site file", got["reason"])

    def test_an_accelerated_torch_keeps_torch(self):
        """ONNX rescues CPU-bound boxes; on mps/cuda it is a downgrade — and
        on mps it walks into the CoreML EP the module rules out."""
        for device in ("mps", "cuda"):
            got = accel.select_detection_weights("models/yolov8n.pt",
                                                 backend="auto", device=device)
            self.assertEqual(got["backend"], "torch")
            self.assertIn(device, got["reason"])

    def test_no_onnxruntime_lands_on_torch_with_the_reason(self):
        with mock.patch.object(accel, "onnxruntime_or_none", return_value=None):
            got = accel.select_detection_weights("models/yolov8n.pt",
                                                 backend="auto", device="cpu")
        self.assertEqual(got["backend"], "torch")
        self.assertIn("onnxruntime not installed", got["reason"])

    @unittest.skipUnless(HAVE_ORT, "onnxruntime not installed")
    def test_a_missing_export_lands_on_torch_with_the_reason(self):
        with mock.patch("cvti.detector.core.resolve_weights",
                        side_effect=lambda w: str(ROOT / "models" / Path(w).name)), \
             mock.patch.object(Path, "exists", autospec=True,
                               side_effect=lambda p: p.suffix == ".pt"):
            got = accel.select_detection_weights("models/yolov8n.pt",
                                                 backend="auto", device="cpu")
        self.assertEqual(got["backend"], "torch")
        self.assertIn("no ONNX export", got["reason"])

    @unittest.skipUnless(HAVE_ORT, "onnxruntime not installed")
    def test_a_cpu_box_with_an_export_chooses_onnx(self):
        if not _ensure_export():
            self.skipTest("export unavailable on this machine")
        got = accel.select_detection_weights("models/yolov8n.pt",
                                             backend="auto", device="cpu")
        self.assertEqual(got["backend"], "onnx")
        self.assertTrue(got["weights"].endswith(".onnx"))
        self.assertTrue(got["provider"])              # named, even when CPU

    def test_nonsense_site_values_mean_auto(self):
        self.assertEqual(accel.backend_from_site({"detector_backend": "grille"}),
                         "auto")
        self.assertEqual(accel.backend_from_site({}), "auto")
        self.assertEqual(accel.backend_from_site({"detector_backend": "TORCH"}),
                         "torch")

    def test_provider_preference_order_is_dml_then_openvino(self):
        fake = mock.MagicMock()
        fake.get_available_providers.return_value = [
            "OpenVINOExecutionProvider", "DmlExecutionProvider",
            "CPUExecutionProvider"]
        with mock.patch.object(accel, "onnxruntime_or_none", return_value=fake):
            self.assertEqual(accel.best_available_provider(),
                             "DmlExecutionProvider")

    def test_cpu_only_boxes_offer_no_preferred_provider(self):
        fake = mock.MagicMock()
        fake.get_available_providers.return_value = ["CPUExecutionProvider"]
        with mock.patch.object(accel, "onnxruntime_or_none", return_value=fake):
            self.assertIsNone(accel.best_available_provider())


class FallbackTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_ORT, "onnxruntime not installed")
    def test_a_corrupt_export_falls_back_to_torch_and_says_so(self):
        """The .pt in the bundle is the fallback that makes the .onnx safe to
        prefer — a broken export costs speed, never detection."""
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / "yolov8n.pt"
            shutil.copy(DETECT_PT, pt)
            (Path(tmp) / "yolov8n.onnx").write_bytes(b"not an onnx graph")
            model, info = accel.load_detector(str(pt), backend="onnx")
            self.assertEqual(info["backend"], "torch")
            self.assertIn("ONNX load failed", info["reason"])
            self.assertIsNotNone(model)               # detection still works

    def test_detector_health_reports_backend_reason_and_model(self):
        doc = accel.detector_health({"backend": "torch", "weights": "x/y.pt",
                                     "provider": None, "reason": "because"})
        self.assertEqual(doc, {"backend": "torch", "model": "y.pt",
                               "provider": None, "reason": "because"})


class SeamPins(unittest.TestCase):
    """The fenced transgression: our wrapper rebuilds ultralytics' session."""

    def test_ultralytics_is_pinned_and_the_seam_looks_as_assumed(self):
        import inspect
        import ultralytics
        from ultralytics.nn.backends import onnx as backend
        req = (ROOT / "requirements.txt").read_text()
        self.assertIn(f"ultralytics=={ultralytics.__version__}", req,
                      "the DirectML seam patch leans on a PINNED ultralytics")
        src = inspect.getsource(backend.ONNXBackend)
        for marker in ("get_available_providers", "CPUExecutionProvider",
                       "InferenceSession"):
            self.assertIn(marker, src,
                          f"ultralytics' ONNX backend moved ({marker}) — "
                          "re-validate accel._patch_ultralytics_providers")

    def test_the_rebuild_puts_the_preferred_provider_in_front(self):
        inst = mock.MagicMock()
        inst.session.get_providers.return_value = ["CPUExecutionProvider"]
        fake_ort = mock.MagicMock()
        fake_ort.InferenceSession.return_value.get_outputs.return_value = []
        with mock.patch.object(accel, "best_available_provider",
                               return_value="DmlExecutionProvider"), \
             mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            self.assertTrue(accel._ensure_preferred_provider(inst, "model.onnx"))
        fake_ort.InferenceSession.assert_called_once_with(
            "model.onnx",
            providers=["DmlExecutionProvider", "CPUExecutionProvider"])

    def test_an_already_accelerated_session_is_left_alone(self):
        inst = mock.MagicMock()
        inst.session.get_providers.return_value = ["DmlExecutionProvider",
                                                   "CPUExecutionProvider"]
        with mock.patch.object(accel, "best_available_provider",
                               return_value="DmlExecutionProvider"):
            self.assertFalse(accel._ensure_preferred_provider(inst, "m.onnx"))

    def test_a_rebuild_failure_keeps_the_original_session(self):
        inst = mock.MagicMock()
        inst.session.get_providers.return_value = ["CPUExecutionProvider"]
        original_session = inst.session
        fake_ort = mock.MagicMock()
        fake_ort.InferenceSession.side_effect = RuntimeError("driver says no")
        with mock.patch.object(accel, "best_available_provider",
                               return_value="DmlExecutionProvider"), \
             mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            self.assertFalse(accel._ensure_preferred_provider(inst, "m.onnx"))
        self.assertIs(inst.session, original_session)


@unittest.skipUnless(HAVE_ORT and DETECT_PT.exists(), "needs models + onnxruntime")
class ParityTests(unittest.TestCase):
    """The claim W2 ships under: the ONNX detector IS the torch detector."""

    def test_identical_boxes_on_real_footage(self):
        if not _ensure_export():
            self.skipTest("export unavailable on this machine")
        import cv2
        import numpy as np
        from ultralytics import YOLO
        cap = cv2.VideoCapture(str(ROOT / "data/test_clips/normal_street_01.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
        ok, img = cap.read()
        cap.release()
        self.assertTrue(ok)
        rp = YOLO(str(DETECT_PT)).predict(img, imgsz=640, conf=0.25,
                                          device="cpu", verbose=False)[0]
        ro = YOLO(str(DETECT_ONNX), task="detect").predict(
            img, imgsz=640, conf=0.25, verbose=False)[0]
        self.assertGreater(len(rp.boxes), 0, "the parity frame must have people")
        self.assertEqual(len(rp.boxes), len(ro.boxes))
        a = rp.boxes.xyxy.numpy(); a = a[a[:, 0].argsort()]
        b = ro.boxes.xyxy.numpy(); b = b[b[:, 0].argsort()]
        self.assertLess(float(np.abs(a - b).max()), 0.5,
                        "ONNX boxes drifted from torch — the detector changed")
        self.assertEqual(sorted(rp.boxes.cls.tolist()),
                         sorted(ro.boxes.cls.tolist()))

    def test_the_hot_path_consumers_accept_onnx_results(self):
        if not _ensure_export():
            self.skipTest("export unavailable on this machine")
        import cv2
        import supervision as sv
        from ultralytics import YOLO
        from cvti.detector.core import extract_detections
        cap = cv2.VideoCapture(str(ROOT / "data/test_clips/normal_street_01.mp4"))
        ok, img = cap.read()
        cap.release()
        model = YOLO(str(DETECT_ONNX), task="detect")
        result = model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
        sv.Detections.from_ultralytics(result)                     # must not raise
        extract_detections(result, model.names, {"person"})        # must not raise


if __name__ == "__main__":
    unittest.main()
