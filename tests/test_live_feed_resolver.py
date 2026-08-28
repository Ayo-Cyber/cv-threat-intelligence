"""Live EarthCams must work on an INSTALL, not just on the dev laptop.

Field report, 28 Aug: 'it didn't come on when it was switched.' The resolver
shelled out to a yt-dlp EXECUTABLE and refused outright when the machine had
none — true of every customer machine, since the bundle never shipped it. The
button rendered everywhere; the feature worked only where the repo's venv was
on PATH. The yt_dlp LIBRARY is a declared dependency and travels inside the
bundle, so it does the resolving now; the executable is a dev fallback only.
"""
from __future__ import annotations

import inspect
import unittest

from cvti.app import console_backend


class LiveFeedResolverShipsTest(unittest.TestCase):

    def setUp(self):
        self.src = inspect.getsource(console_backend.ConsoleBackend._resolve_live_config)

    def test_the_library_not_the_executable_is_the_primary_path(self):
        self.assertIn("import yt_dlp", self.src,
                      "the resolver no longer uses the bundled library")

    def test_a_missing_executable_is_not_a_hard_refusal(self):
        """shutil.which('yt-dlp') gated the whole feature on software the
        bundle does not carry — and the error told a security guard to
        'pip install'."""
        self.assertNotIn('shutil.which("yt-dlp")', self.src)
        self.assertNotIn("pip install", self.src,
                         "developer instructions surfaced as a customer error")

    def test_the_bundle_declares_the_dynamic_import(self):
        """The resolver imports yt_dlp inside a function; PyInstaller cannot
        see that statically. Without the hiddenimport the fix works in dev and
        silently dies in the installed app — the exact bug, one layer down."""
        spec = open("packaging/argus.spec").read()
        self.assertIn('"yt_dlp"', spec)

    def test_one_dead_feed_does_not_sink_the_others(self):
        self.assertIn('return name, ""', self.src,
                      "a failed resolve must degrade to an empty URL, not raise")


if __name__ == "__main__":
    unittest.main()
