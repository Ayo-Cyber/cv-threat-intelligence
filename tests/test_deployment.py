"""One installer that actually contains the product (EP-05-T1).

The audit's finding, verbatim: "you have shipped .dmg and .zip installers that
cannot detect anything on their own." These tests pin the three mechanisms
that close it — the engine ships inside the bundle and the app can find it,
the AI runtime is brought up (and its models stored) somewhere sane, and the
user's data directory says Argus without orphaning an existing CVTI install.

Nothing here launches Ollama or an engine: spawn points are monkeypatched.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

from _backend_helper import signed_in

from cvti.verification import ollama as vlm_ollama

ROOT = Path(__file__).resolve().parents[1]


class SpecBundlesTheProductTest(unittest.TestCase):
    """Static truth about packaging/argus.spec — cheap, but it is exactly the
    file whose silent regression re-creates the audit finding."""

    spec = (ROOT / "packaging" / "argus.spec").read_text()

    def test_both_executables_are_declared(self):
        self.assertIn('name="Argus"', self.spec)
        self.assertIn('name="argus-engine"', self.spec)

    def test_the_engine_analysis_carries_the_detection_stack(self):
        for needed in ("ultralytics", "transformers", "torch",
                       "cvti.serving.pipeline", "engine_entry.py"):
            self.assertIn(needed, self.spec, f"spec lost {needed}")

    def test_web_ui_and_weights_ride_along(self):
        self.assertIn('"cvti/app/web"', self.spec)
        self.assertIn('"models/*.pt"', self.spec)
        self.assertIn("vendor/ollama", self.spec)

    def test_the_old_name_is_gone_from_packaging_and_ci(self):
        for rel in ("packaging/argus.spec", "packaging/build.py",
                    "packaging/make_dmg.sh", ".github/workflows/build-app.yml"):
            text = (ROOT / rel).read_text()
            self.assertNotIn("CVTI Console", text, f"{rel} still says CVTI Console")

    def test_ci_still_gates_builds_on_the_test_suite(self):
        wf = (ROOT / ".github/workflows/build-app.yml").read_text()
        self.assertIn("name: Test suite", wf)      # branch protection matches on this
        self.assertIn("needs: test", wf)


class EngineCommandTest(unittest.TestCase):
    """The app finds the engine differently in dev and inside the bundle."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "site.json").write_text('{"cameras": []}')
        self.be = signed_in("owner", site_path=str(root / "site.json"),
                            db_path=str(root / "events.db"), enable_demo=False)
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self):
        for attr in ("frozen",):
            if hasattr(sys, attr):
                try:
                    delattr(sys, attr)
                except AttributeError:
                    pass

    def test_dev_runs_the_module_with_this_python(self):
        self.assertEqual(self.be._engine_command()[:3],
                         [sys.executable, "-m", "cvti.serving.pipeline"])

    def test_frozen_uses_the_engine_shipped_next_to_the_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = "argus-engine.exe" if sys.platform == "win32" else "argus-engine"
            engine = Path(tmp) / exe
            engine.write_bytes(b"#!/bin/sh\n")
            real_exec = sys.executable
            sys.frozen = True
            sys.executable = str(Path(tmp) / "Argus")
            try:
                self.assertEqual(self.be._engine_command(), [str(engine)])
            finally:
                sys.executable = real_exec
                del sys.frozen

    def test_a_lean_bundle_without_an_engine_says_so_instead_of_crashing(self):
        # A frozen app whose bundle has no argus-engine (the old viewer-only
        # build) must keep the honest playback-demo answer.
        sys.frozen = True
        try:
            out = self.be.start_monitoring()
        finally:
            del sys.frozen
        self.assertFalse(out["running"])
        self.assertIn("no detection engine", out["note"])


class DataDirMigrationTest(unittest.TestCase):
    """Renaming CVTI -> Argus must carry the user's events and logs across."""

    def _paths(self, fake_home: Path):
        if sys.platform == "darwin":
            support = fake_home / "Library" / "Application Support"
            return support / "CVTI", support / "Argus"
        if sys.platform == "win32":
            appdata = Path(os.environ.get("APPDATA", str(fake_home)))
            return appdata / "CVTI", appdata / "Argus"
        return fake_home / ".cvti", fake_home / ".argus"

    def _with_home(self, fake_home: Path):
        from cvti import utils
        real = Path.home
        Path.home = staticmethod(lambda: fake_home)  # type: ignore[method-assign]
        self.addCleanup(lambda: setattr(Path, "home", real))
        return utils

    def test_an_existing_cvti_dir_is_moved_not_orphaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            old, new = self._paths(home)
            if sys.platform == "win32":       # APPDATA would escape the tmp home
                self.skipTest("windows APPDATA not sandboxed here")
            old.mkdir(parents=True)
            (old / "events.db").write_text("precious")
            utils = self._with_home(home)
            got = utils.user_data_dir()
            self.assertEqual(got, new)
            self.assertEqual((new / "events.db").read_text(), "precious")
            self.assertFalse(old.exists(), "legacy dir left behind as a decoy")

    def test_a_fresh_machine_just_gets_argus(self):
        with tempfile.TemporaryDirectory() as tmp:
            if sys.platform == "win32":
                self.skipTest("windows APPDATA not sandboxed here")
            utils = self._with_home(Path(tmp))
            got = utils.user_data_dir()
            self.assertTrue(got.name in ("Argus", ".argus") and got.is_dir())


class OllamaServerManagementTest(unittest.TestCase):
    """The bundled runtime: started with a writable model dir, never fought over."""

    def test_start_server_points_models_at_the_given_dir(self):
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"], captured["env"] = cmd, kw.get("env")
            class P:  # noqa: D401 - minimal handle
                pid = 4242
            return P()

        real_popen = vlm_ollama.subprocess.Popen
        real_binary = vlm_ollama.ollama_binary
        vlm_ollama.subprocess.Popen = fake_popen
        vlm_ollama.ollama_binary = lambda: "/fake/ollama"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                models = str(Path(tmp) / "ollama-models")
                self.assertTrue(vlm_ollama.start_server(models_dir=models))
                self.assertEqual(captured["cmd"][:2], ["/fake/ollama", "serve"])
                self.assertEqual(captured["env"]["OLLAMA_MODELS"], models)
                self.assertTrue(Path(models).is_dir(), "model dir not pre-created")
                # RAM policy: without these, a 3.3 GB model runs ~13 GB resident
                self.assertEqual(captured["env"]["OLLAMA_NUM_PARALLEL"], "2")
                self.assertEqual(captured["env"]["OLLAMA_CONTEXT_LENGTH"], "8192")
                self.assertEqual(captured["env"]["OLLAMA_KV_CACHE_TYPE"], "q8_0")
        finally:
            vlm_ollama.subprocess.Popen = real_popen
            vlm_ollama.ollama_binary = real_binary

    def test_an_already_running_server_is_used_as_is(self):
        real_up = vlm_ollama.server_up
        real_start = vlm_ollama.start_server
        started = []
        vlm_ollama.server_up = lambda *a, **k: True
        vlm_ollama.start_server = lambda **k: started.append(1) or True
        try:
            self.assertTrue(vlm_ollama.ensure_server())
            self.assertEqual(started, [], "spawned a second server over a live one")
        finally:
            vlm_ollama.server_up = real_up
            vlm_ollama.start_server = real_start

    def test_no_binary_is_a_fast_no_not_a_hang(self):
        real_up = vlm_ollama.server_up
        real_binary = vlm_ollama.ollama_binary
        vlm_ollama.server_up = lambda *a, **k: False
        vlm_ollama.ollama_binary = lambda: None
        try:
            self.assertFalse(vlm_ollama.ensure_server())
        finally:
            vlm_ollama.server_up = real_up
            vlm_ollama.ollama_binary = real_binary


class FrozenDefaultsTest(unittest.TestCase):
    """The shell's frozen path must not write into the filesystem root."""

    def test_shell_source_routes_frozen_site_into_user_data_dir(self):
        src = (ROOT / "cvti" / "app" / "shell.py").read_text()
        self.assertIn("user_data_dir() / \"site\"", src.replace("'", '"'))
        self.assertNotIn('default="configs/site_live.json"', src,
                         "frozen app would resolve this against cwd '/'")


if __name__ == "__main__":
    unittest.main()


class BundleWeightTest(unittest.TestCase):
    """A customer's download must not carry what the product never calls
    (bundle audit of the shipped v1.0.0 windows zip, 25 Aug)."""

    spec = (ROOT / "packaging" / "argus.spec").read_text()

    def test_polars_is_excluded_from_both_analyses(self):
        self.assertEqual(self.spec.count('"polars"'), 2,
                         "156 MB of a dependency the product never imports")

    def test_devtools_debug_resources_are_stripped(self):
        self.assertIn("_strip_dead_weight", self.spec)
        self.assertIn("qtwebengine_devtools_resources.debug.pak", self.spec)
        self.assertIn("app_a.datas = _strip_dead_weight(app_a.datas)", self.spec)

    def test_the_ai_runtime_is_still_bundled(self):
        # the thing that must NEVER be pruned by a slimming pass
        self.assertIn("vendor/ollama", self.spec)
