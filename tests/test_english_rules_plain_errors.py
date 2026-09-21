"""The Rules panel must say what is wrong in words an operator can act on.

When the local VLM was absent, each English-rule scan recorded the raw
exception -- "urlopen error [Errno 61] Connection refused" -- and the pilot
read that as his sentences being broken. They were saved and fine; the
3.3 GB model had never downloaded (Windows, 21 Sep).
"""
from __future__ import annotations

import unittest

from cvti.serving.custom_rules import plain_error


class PlainErrors(unittest.TestCase):
    def test_missing_model_points_at_the_download(self):
        out = plain_error("model 'gemma3:4b' not found, try pulling it first")
        self.assertIn("not installed", out)
        self.assertIn("Verification", out)
        self.assertIn("saved", out, "the operator must hear the rules are safe")

    def test_no_server_points_at_the_same_button(self):
        for raw in ("<urlopen error [Errno 61] Connection refused>",
                    "HTTPConnectionPool: Max retries exceeded",
                    "[WinError 10061] No connection could be made"):
            with self.subTest(raw=raw):
                out = plain_error(raw)
                self.assertIn("not running", out)
                self.assertIn("Download model", out)

    def test_a_timeout_is_not_blamed_on_the_rules(self):
        self.assertIn("did not answer in time", plain_error("Read timed out."))

    def test_unknown_errors_pass_through_untouched(self):
        self.assertEqual(plain_error("division by zero"), "division by zero")
        self.assertEqual(plain_error(""), "")
