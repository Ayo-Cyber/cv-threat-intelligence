"""Three reports from the pilot's machine, 30 Aug, one PR.

1. 'Anytime the software starts I get this alert of fire': cameras auto-adjust
   exposure/IR for the first seconds after a stream opens, and the fire
   detector's hot-area heuristic reads the white-out as flame — a critical
   alert at every engine start. Situational detectors now sit out a settle
   window.
2. detector.cam1: 480 errors at 100%, cause invisible: the health table showed
   a rate with no WHAT. The component's last error now rides along in the UI.
3. The triage timeline hard-coded a green 'Verified by TrueSight' dot — shown
   over an UNVERIFIED alert with confidence 0.00. It now tells the truth.
(The root of #2 — stdlib logging.config missing from the frozen bundle — is
pinned in the spec test below.)
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import supervision as sv


def _no_detections():
    return sv.Detections.empty()


class SettleWindowTest(unittest.TestCase):

    def _state(self, **kw):
        from cvti.rules.customization import CustomizationEngine
        from cvti.serving.camera import PerCameraState
        eng = CustomizationEngine("configs/all_threats_v1.json",
                                  baseline_path="configs/baseline_critical_v1.json")
        return PerCameraState("cam1", eng, fire_smoke=True, person_filter=False, **kw)

    def _hot_frame(self):
        # Matches the detector's own hot mask (warm hue, saturated, bright) —
        # what an IR/exposure bloom looks like after tone-mapping: BGR orange.
        f = np.zeros((120, 160, 3), dtype=np.uint8)
        f[:, :, 0] = 30    # B
        f[:, :, 1] = 140   # G
        f[:, :, 2] = 250   # R
        return f

    def test_no_fire_alert_during_the_settle_window(self):
        st = self._state()
        events = []
        for i in range(30):                    # 7.5s of white frames at 4fps
            out = st.process(_no_detections(), self._hot_frame(), timestamp=i * 0.25)
            events.extend(a.rule_name for a in (out or []))
        self.assertFalse([e for e in events if "fire" in e],
                         f"fire alert during exposure settle: {events}")

    def _flame_frame(self):
        # A LOCALIZED hot region (~15% of frame) — what an actual fire looks
        # like to a fixed camera, unlike the whole-frame bloom above, which
        # the detector now rejects outright (max_hot_area_ratio, 3 Sep).
        f = np.zeros((120, 160, 3), dtype=np.uint8)
        f[40:90, 50:110, 0] = 30
        f[40:90, 50:110, 1] = 140
        f[40:90, 50:110, 2] = 250
        return f

    def test_fire_can_still_fire_after_settling(self):
        """The window must delay the detector, not delete it: a real,
        localized flame that is ignored during settle DOES alert once the
        camera is past it — proving the guard is a delay, not a mute."""
        st = self._state()
        events = []
        for i in range(80):                    # 20s: settle ends at 8s
            out = st.process(_no_detections(), self._flame_frame(), timestamp=i * 0.25)
            events.extend(a.rule_name for a in (out or []))
        self.assertTrue([e for e in events if "fire" in e],
                        "the settle window muted the fire detector forever")

    def test_whole_frame_bloom_never_fires_even_after_settling(self):
        """The pilot's report, round two (3 Sep): 'when we start it shows
        baseline fire'. His camera's IR/exposure bloom outlives the 8s settle
        window — but bloom is WHOLE-FRAME hot, and a real fire is not. The
        ceiling makes the bloom class impossible regardless of timing."""
        st = self._state()
        events = []
        for i in range(200):                   # 50s — far past any settle
            out = st.process(_no_detections(), self._hot_frame(), timestamp=i * 0.25)
            events.extend(a.rule_name for a in (out or []))
        self.assertFalse([e for e in events if "fire" in e],
                         f"whole-frame bloom read as fire: {events}")

    def test_the_detector_wakes_after_settling(self):
        from cvti.serving.camera import PerCameraState
        import inspect
        src = inspect.getsource(PerCameraState)
        self.assertIn("settle_seconds", src)
        self.assertIn("self.fire_smoke and settled", src)
        self.assertIn("self.running and settled", src)
        self.assertIn("self.crowd_formation and settled", src)


class FrozenBundleCarriesLoggingConfigTest(unittest.TestCase):

    def test_the_spec_names_the_dynamic_stdlib_imports(self):
        """'No module named logging.config' killed the weapons model on every
        install — stdlib, but only imported dynamically by the yolov5 loader,
        so PyInstaller never saw it."""
        spec = Path("packaging/argus.spec").read_text()
        self.assertIn('"logging.config"', spec)


class UiTellsTheTruthTest(unittest.TestCase):

    def setUp(self):
        self.html = Path("cvti/app/web/index.html").read_text()

    def test_the_timeline_never_shows_verified_for_unverified(self):
        block = self.html.split("function timelineHTML")[1].split("function ")[0]
        self.assertIn("Verification unavailable", block)
        self.assertIn("a.unverified", block)

    def test_degraded_components_say_what_failed(self):
        block = self.html.split("function componentRows")[1].split("function ")[0]
        self.assertIn("c.last_error", block)
