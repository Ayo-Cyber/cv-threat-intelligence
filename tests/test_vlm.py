from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving import vlm


class VlmStatusTests(unittest.TestCase):
    def setUp(self):
        self._orig = vlm._http_json  # restore so patches don't leak across files

    def tearDown(self):
        vlm._http_json = self._orig

    def test_model_matches_tag_and_base(self):
        self.assertTrue(vlm._model_matches("gemma3:4b", ["gemma3:4b", "llama3:8b"]))
        self.assertTrue(vlm._model_matches("gemma3:4b", ["gemma3:latest"]))  # base match
        self.assertFalse(vlm._model_matches("gemma3:4b", ["llama3:8b", "qwen:7b"]))
        self.assertFalse(vlm._model_matches("gemma3:4b", []))

    def test_gate_status_live(self):
        vlm._http_json = lambda url, timeout=2.0: {"models": [{"name": "gemma3:4b"}]}
        s = vlm.gate_status("gemma3:4b")
        self.assertTrue(s["ollama"] and s["model_present"])
        self.assertEqual(s["mode"], "live")

    def test_gate_status_no_model(self):
        vlm._http_json = lambda url, timeout=2.0: {"models": [{"name": "llama3:8b"}]}
        s = vlm.gate_status("gemma3:4b")
        self.assertTrue(s["ollama"])
        self.assertFalse(s["model_present"])
        self.assertEqual(s["mode"], "no-model")

    def test_gate_status_offline(self):
        def boom(url, timeout=2.0):
            raise OSError("connection refused")
        vlm._http_json = boom
        s = vlm.gate_status("gemma3:4b")
        self.assertFalse(s["ollama"])
        self.assertEqual(s["mode"], "offline")


if __name__ == "__main__":
    unittest.main()
