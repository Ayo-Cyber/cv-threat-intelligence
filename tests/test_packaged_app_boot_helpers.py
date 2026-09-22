"""The cold-boot smoke's pure parts: which binary per OS, and what 'connected' means."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "e2e"))

from packaged_app_boot import app_binary, camera_connected, free_port


class BinaryPerPlatform(unittest.TestCase):
    def test_macos_app_bundle(self):
        self.assertEqual(app_binary(Path("/r/mac/Argus.app"), "darwin"),
                         Path("/r/mac/Argus.app/Contents/MacOS/Argus"))

    def test_windows_unpacked(self):
        self.assertEqual(app_binary(Path("C:/r/win-unpacked"), "win32"),
                         Path("C:/r/win-unpacked/Argus.exe"))

    def test_linux_prefers_whatever_exists(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "Argus").write_text("")
            found = app_binary(Path(d), "linux")
            self.assertTrue(found.exists(), found)              # case-insensitive FS: either spelling
            self.assertEqual(found.name.lower(), "argus")
            self.assertEqual(app_binary(Path("/nowhere"), "linux"), Path("/nowhere/argus"))


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
