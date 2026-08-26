"""First-run wizard backend: templates and the self-test (EP-05-T3).

The acceptance items these pin: three use-case templates with sensible
defaults, and a self-test that names exactly what is missing in plain English
— never a generic "something failed".
"""
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from _backend_helper import signed_in

from cvti.serving import onboarding, vlm


def _site(tmp: Path, *cameras) -> "ConsoleBackend":
    (tmp / "site.json").write_text(json.dumps({"cameras": list(cameras)}))
    return signed_in("owner", site_path=str(tmp / "site.json"),
                     db_path=str(tmp / "events.db"), enable_demo=False)


class TemplateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_three_templates_each_resolving_every_detector_flag(self):
        be = _site(self.root)
        tpls = be.use_case_templates()
        self.assertEqual(sorted(tpls), ["office", "retail", "warehouse"])
        for name, t in tpls.items():
            self.assertEqual(sorted(t["detectors"]), sorted(be.RULE_FLAGS),
                             f"{name} leaves a detector flag unresolved")

    def test_the_templates_differ_where_the_use_cases_differ(self):
        be = _site(self.root)
        tpls = be.use_case_templates()
        self.assertTrue(tpls["retail"]["detectors"]["concealment"])
        self.assertFalse(tpls["warehouse"]["detectors"]["concealment"],
                         "a warehouse has no shoppers to conceal merchandise")
        self.assertTrue(tpls["warehouse"]["detectors"]["fall"])
        self.assertFalse(tpls["office"]["detectors"]["theft"])

    def test_apply_sets_every_flag_on_every_camera_and_seeds_defaults(self):
        be = _site(self.root, {"id": "a", "source": "x.mp4"},
                   {"id": "b", "source": "y.mp4"})
        out = be.apply_template("warehouse")
        self.assertTrue(out["ok"])
        self.assertEqual(out["cameras"], 2)
        for cam in onboarding.list_cameras(be.site_path):
            self.assertTrue(cam["fall"] and cam["fire_smoke"])
            self.assertFalse(cam["concealment"], "flags must be set OFF explicitly "
                             "or switching templates couldn't remove detectors")
            # a newly enabled detector arrives with its tuning params
            self.assertIn("crowd_min_people", cam)

    def test_switching_templates_is_deterministic_not_additive(self):
        be = _site(self.root, {"id": "a", "source": "x.mp4"})
        be.apply_template("retail")
        be.apply_template("office")
        cam = onboarding.list_cameras(be.site_path)[0]
        self.assertFalse(cam["concealment"], "retail's detector survived the switch")
        self.assertTrue(cam["violence"])

    def test_an_unknown_template_is_an_answer_not_a_crash(self):
        be = _site(self.root)
        self.assertIn("error", be.apply_template("casino"))

    def test_applying_a_template_lands_in_the_audit_trail(self):
        be = _site(self.root, {"id": "a", "source": "x.mp4"})
        be.apply_template("retail")
        actions = [e["action"] for e in be.audit_entries()]
        self.assertIn("config_change", actions)


class SelfTestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _by_id(self, checks):
        return {c["id"]: c for c in checks}

    def test_no_cameras_is_named_with_the_fix(self):
        be = _site(self.root)
        c = self._by_id(be.setup_check())["cameras"]
        self.assertFalse(c["ok"])
        self.assertIn("Add a camera", c["fix"])

    def test_a_missing_video_file_is_named_precisely(self):
        be = _site(self.root, {"id": "till", "source": "/nope/gone.mp4"})
        c = self._by_id(be.setup_check())["stream:till"]
        self.assertFalse(c["ok"])
        self.assertIn("/nope/gone.mp4", c["detail"])

    def test_a_reachable_rtsp_host_passes_and_an_unreachable_one_fails(self):
        # a live local listener stands in for the camera
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0)); srv.listen(1)
        port = srv.getsockname()[1]
        accepts = threading.Thread(target=lambda: srv.accept(), daemon=True)
        accepts.start()
        try:
            be = _site(self.root,
                       {"id": "up", "source": f"rtsp://127.0.0.1:{port}/s1"},
                       {"id": "down", "source": "rtsp://127.0.0.1:1/s1"})
            by = self._by_id(be.setup_check())
            self.assertTrue(by["stream:up"]["ok"])
            self.assertFalse(by["stream:down"]["ok"])
            self.assertIn("127.0.0.1:1", by["stream:down"]["detail"])
        finally:
            srv.close()

    def test_verifier_advice_matches_what_is_actually_wrong(self):
        be = _site(self.root)
        real = vlm.gate_status
        try:
            vlm.gate_status = lambda *a, **k: {"mode": "no-model"}
            c = self._by_id(be.setup_check())["verifier"]
            self.assertIn("3.3 GB", c["fix"])
            vlm.gate_status = lambda *a, **k: {"mode": "offline"}
            from cvti.verification import ollama as vo
            real_bin = vo.ollama_binary
            vo.ollama_binary = lambda: "/fake/ollama"
            try:
                c = self._by_id(be.setup_check())["verifier"]
                self.assertIn("Start verifier", c["fix"])
            finally:
                vo.ollama_binary = real_bin
        finally:
            vlm.gate_status = real

    def test_console_only_notifications_are_a_warning_not_a_pass(self):
        be = _site(self.root)
        c = self._by_id(be.setup_check())["notify"]
        self.assertIsNone(c["ok"])
        self.assertIn("inside the app", c["detail"])

    def test_every_failing_check_carries_a_fix_in_plain_english(self):
        be = _site(self.root, {"id": "till", "source": "/nope/gone.mp4"})
        for c in be.setup_check():
            if c["ok"] is False:
                self.assertTrue(c.get("fix"), f"{c['id']} fails without telling "
                                "the user what to do about it")


if __name__ == "__main__":
    unittest.main()


class BackgroundPrefetchTest(unittest.TestCase):
    """The 3.3 GB verifier model downloads WHILE the user sets up (26 Aug),
    not as a wall at step 6. Pins the wiring; the visual flow was verified in
    a browser across every step."""

    UI = (Path(__file__).resolve().parents[1] / "cvti/app/web/index.html").read_text()

    def test_the_download_starts_when_the_wizard_opens(self):
        opener = self.UI.split("function openWizard")[1].split("\nfunction ")[0]
        self.assertIn("prepareVerifier()", opener,
                      "setup still waits until the Verification step to start 3.3 GB")

    def test_it_is_started_once_per_session(self):
        prep = self.UI.split("function prepareVerifier")[1].split("\nfunction ")[0]
        self.assertIn("if(wiz.prep) return;", prep)

    def test_an_already_installed_model_skips_straight_to_ready(self):
        prep = self.UI.split("function prepareVerifier")[1].split("\nfunction ")[0]
        self.assertIn('g.mode === "live"', prep)
        self.assertIn('state:"done"', prep)

    def test_progress_survives_every_step_change(self):
        render = self.UI.split("function renderWizard")[1].split("\nfunction ")[0]
        self.assertIn("renderPrepStrip()", render, "the strip vanishes on re-render")
        self.assertIn('id="wizPrep"', render)

    def test_a_failed_download_does_not_block_finishing_setup(self):
        strip = self.UI.split("function renderPrepStrip")[1].split("\nfunction ")[0]
        self.assertIn("You can finish setup", strip)
        self.assertIn("unverified", strip)

    def test_polling_stops_when_the_wizard_closes(self):
        poll = self.UI.split("function pollPrep")[1].split("\nfunction ")[0]
        self.assertIn('$("wizard").classList.contains("on")', poll,
                      "the poller outlives the wizard")
