"""The cold-boot smoke's pure parts: which binary per OS, and what 'connected' means."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "e2e"))

from packaged_app_boot import (
    app_binary,
    camera_connected,
    free_port,
    resolve_app_root,
)


class BinaryPerPlatform(unittest.TestCase):
    def test_macos_app_bundle(self):
        self.assertEqual(app_binary(Path("/r/mac/Argus.app"), "darwin"),
                         Path("/r/mac/Argus.app/Contents/MacOS/Argus"))

    def test_windows_unpacked(self):
        self.assertEqual(app_binary(Path("C:/r/win-unpacked"), "win32"),
                         Path("C:/r/win-unpacked/Argus.exe"))

    def test_linux_is_named_after_the_package_not_the_product(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "argus-desktop").write_text("")
            (Path(d) / "chrome-sandbox").write_text("")
            self.assertEqual(app_binary(Path(d), "linux"), Path(d) / "argus-desktop")

    def test_linux_falls_back_to_the_one_big_executable(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            for name, size in (("chrome-sandbox", 10), ("chrome_crashpad_handler", 10),
                               ("some-renamed-app", 5000), ("libffmpeg.so", 9000)):
                f = Path(d) / name
                f.write_bytes(b"x" * size)
                os.chmod(f, 0o755)
            (Path(d) / "resources").mkdir()
            self.assertEqual(app_binary(Path(d), "linux"), Path(d) / "some-renamed-app")
            self.assertEqual(app_binary(Path("/nowhere"), "linux"), Path("/nowhere/argus-desktop"))


class Connected(unittest.TestCase):
    def test_uses_the_engines_own_state_word(self):
        self.assertTrue(camera_connected({"cameras": [{"state": "starting"}, {"state": "connected"}]}))
        self.assertFalse(camera_connected({"cameras": [{"state": "starting"}]}))
        self.assertFalse(camera_connected({"cameras": []}))
        self.assertFalse(camera_connected({}))
        self.assertFalse(camera_connected(None))


class Port(unittest.TestCase):
    def test_free_port_is_a_real_port(self):
        p = free_port()
        self.assertTrue(1024 <= p <= 65535)


if __name__ == "__main__":
    unittest.main()


class ResolveAppRoot(unittest.TestCase):
    """Boot either electron-builder's release dir or an INSTALLED copy.

    The pilot double-clicks the NSIS installer, which writes to
    %LOCALAPPDATA%\\Programs\\Argus — a directory nothing in CI had launched
    until 24 Sep.
    """

    def test_an_installed_directory_is_its_own_app_root(self):
        with tempfile.TemporaryDirectory() as d:
            installed = Path(d) / "Argus"
            installed.mkdir()
            (installed / "Argus.exe").write_text("")
            self.assertEqual(resolve_app_root(installed, "win32"), installed)

    def test_a_release_directory_still_resolves_to_the_unpacked_tree(self):
        with tempfile.TemporaryDirectory() as d:
            release = Path(d) / "release"
            unpacked = release / "win-unpacked"
            unpacked.mkdir(parents=True)
            (unpacked / "Argus.exe").write_text("")
            self.assertEqual(resolve_app_root(release, "win32"), unpacked)

