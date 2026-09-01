"""Everything the shipped app references must actually ship.

The recurring disease of this codebase, in one test file. Four times now a
feature worked on the dev laptop and failed on every other machine, always for
the same reason — a path that resolves here and nowhere else:

  - the E2E camera published a clip .gitignore excluded (27 Aug)
  - retention resolved evidence against the wrong root and never purged (27 Aug)
  - Live EarthCams demanded a yt-dlp EXECUTABLE the bundle never carried (28 Aug)
  - the demo feed's configs pointed at data/hse_demo/, untracked (28 Aug)

These assertions walk the feed registry the way the installed app does and
fail if any referenced file would be missing from a fresh install, plus pin
the frozen-mode rules: read-only resources resolve against the bundle root,
mutable configs are copied to the user's data dir before being written.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKED = set(subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.splitlines())


def _tracked(rel) -> bool:
    return str(rel) in TRACKED


class EveryReferencedFileShipsTest(unittest.TestCase):

    def setUp(self):
        self.feeds = json.loads((ROOT / "configs/feeds.json").read_text())

    def test_the_registry_and_every_feed_config_are_tracked(self):
        self.assertTrue(_tracked("configs/feeds.json"))
        for src in self.feeds["sources"]:
            if src.get("kind") == "live":
                continue                     # generated fresh by the resolver
            self.assertTrue(_tracked(src["config"]),
                            f"feed '{src['key']}' points at {src['config']}, "
                            f"which a fresh install will not have")

    def test_every_demo_camera_source_ships(self):
        for src in self.feeds["sources"]:
            if src.get("kind") != "demo" or not _tracked(src["config"]):
                continue
            site = json.loads((ROOT / src["config"]).read_text())
            for cam in site.get("cameras", []):
                s = str(cam.get("source", ""))
                if s.isdigit() or "://" in s:
                    continue                 # webcam / network camera
                self.assertTrue(_tracked(s),
                                f"feed '{src['key']}' camera '{cam['id']}' plays {s} — "
                                f"exists on this laptop, not in the product")

    def test_every_rules_and_zones_reference_ships(self):
        for src in self.feeds["sources"]:
            if not _tracked(src["config"]):
                continue
            site = json.loads((ROOT / src["config"]).read_text())
            for cam in site.get("cameras", []):
                for key in ("config", "zones", "_base_config"):
                    v = cam.get(key)
                    if v:
                        self.assertTrue(_tracked(v),
                                        f"camera '{cam['id']}' {key}={v} is not tracked")

    def test_live_feed_generated_config_dependencies_ship(self):
        """The resolver writes a config whose cameras reference these."""
        for rel in ("configs/rules/live_watch.json", "configs/zones/live_watch.json"):
            self.assertTrue(_tracked(rel), f"{rel} is referenced by every resolved live camera")


class FrozenModePathRulesTest(unittest.TestCase):
    """The installed app is launched with cwd anywhere and its bundle may be
    read-only (Program Files). Two rules keep it working: read through the
    bundle root, write through the user data dir."""

    def _backend_src(self):
        from cvti.app import console_backend
        return inspect.getsource(console_backend.ConsoleBackend)

    def test_the_feed_registry_reads_from_the_bundle_root(self):
        from cvti.app import console_backend
        src = inspect.getsource(console_backend.ConsoleBackend._feeds_registry)
        self.assertIn("resource_path", src,
                      'Path("configs/feeds.json") is cwd-relative — the installed '
                      "app would offer no feeds at all")

    def test_feed_switching_writes_only_user_owned_copies(self):
        src = self._backend_src()
        self.assertIn("_writable_config", src)
        from cvti.app import console_backend
        sw = inspect.getsource(console_backend.ConsoleBackend._switch_feed) \
            if hasattr(console_backend.ConsoleBackend, "_switch_feed") else src
        self.assertIn("_writable_config", sw,
                      "a feed switch would write into the install directory — "
                      "PermissionError under Program Files")

    def test_zone_edits_write_to_the_user_dir_when_frozen(self):
        from cvti.app import console_backend
        src = inspect.getsource(console_backend.ConsoleBackend._zones_file)
        self.assertIn("user_data_dir", src)


if __name__ == "__main__":
    unittest.main()


class EveryDefaultModelShipsTest(unittest.TestCase):
    """The engine could not start on the pilot's installed app (29 Aug):
    models/yolov8n-pose.pt was never tracked, so CI bundles shipped without
    it, and ultralytics' silent fallback download aimed at a read-only
    directory — Permission denied x3, dead engine, crash loop. CI never
    noticed because runner checkouts are writable, so the download quietly
    succeeded there. Every weights path the pipeline defaults to must ship."""

    def test_every_default_weights_path_is_tracked(self):
        import re
        src = (ROOT / "cvti/serving/pipeline.py").read_text()
        weights = set(re.findall(r'"(models/[\w.\-]+\.pt)"', src))
        self.assertTrue(weights, "no default weights found — did the signature move?")
        for w in sorted(weights):
            self.assertTrue(_tracked(w),
                            f"{w} is a default the engine loads at startup, and it "
                            f"is not in the repo — installs get a crash loop")

    def test_missing_weights_resolve_to_a_writable_download_target(self):
        import os
        from cvti.detector.core import resolve_weights
        cwd = os.getcwd()
        os.chdir("/tmp")                        # nowhere near the repo
        try:
            target = resolve_weights("models/not-shipped-model.pt")
        finally:
            os.chdir(cwd)
        self.assertFalse(target.startswith("models"),
                         "a missing model still downloads into the cwd — "
                         "read-only on installs")


class MacBundleTakesKeyboardFocusTest(unittest.TestCase):
    """v1.5.0 shipped with LSBackgroundOnly=true — PyInstaller stamps it when
    any bundled EXE is console=True (the engine is) — and a background-only
    app can NEVER become the key window: every keystroke fell through to the
    window behind Argus (field report, 31 Aug). The spec must pin it False."""

    def test_the_spec_pins_lsbackgroundonly_off(self):
        spec = (ROOT / "packaging/argus.spec").read_text()
        self.assertIn('"LSBackgroundOnly": False', spec,
                      "argus.spec must force LSBackgroundOnly off, or the "
                      "console=True engine EXE makes the whole .app untypeable")


class BuildIdentityTest(unittest.TestCase):
    """The build must say which build it is. The plist said 1.0.0 and the
    sidebar said v0.9 while v1.6.x shipped; the installer filed a 64-bit app
    under Program Files (x86) (pilot log, 1 Sep)."""

    def test_the_spec_takes_the_version_from_the_release_tag(self):
        spec = (ROOT / "packaging/argus.spec").read_text()
        self.assertNotIn('APP_VERSION = "1', spec,
                         "version must come from the tag, not a hardcode")
        self.assertIn("GITHUB_REF_NAME", spec)
        self.assertIn('datas.append((_version_file', spec,
                      "the bundle must carry a VERSION file for the app to read")

    def test_the_installer_installs_64bit(self):
        iss = (ROOT / "scripts/installer.iss").read_text()
        self.assertIn("ArchitecturesInstallIn64BitMode", iss)

    def test_the_sidebar_version_is_live_not_hardcoded(self):
        html = (ROOT / "cvti/app/web/index.html").read_text()
        self.assertNotIn("v0.9", html)
        self.assertIn('call("appVersion"', html)

    def test_backend_reports_a_version_string(self):
        from cvti.app.console_backend import ConsoleBackend
        v = ConsoleBackend.app_version(object.__new__(ConsoleBackend))
        self.assertIsInstance(v, str)
        self.assertTrue(v)
