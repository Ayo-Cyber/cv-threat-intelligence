"""The spawned AI runtime's own output is kept, and the Diagnose zip carries
the files that explain a silent verifier.

Until 22 Sep `ollama serve` ran with stdout/stderr on DEVNULL. On the pilot's
Windows box the model may refuse to load (memory, a missing DLL, a damaged
pull) and the only process that knows why was told to say nothing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.verification import ollama


class ServerOutputGoesToAFile(unittest.TestCase):
    def _spawn(self, **kw) -> dict:
        captured: dict = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["stdout"] = kwargs.get("stdout")
            captured["stderr"] = kwargs.get("stderr")
            # what the child would do: write a line through the handle
            out = kwargs.get("stdout")
            if hasattr(out, "write"):
                out.write(b"time=... msg=\"model requires more system memory\"\n")
                out.flush()
            return mock.Mock()

        with mock.patch.object(ollama, "ollama_binary", return_value="/fake/ollama"), \
                mock.patch.object(ollama.subprocess, "Popen", fake_popen), \
                mock.patch.dict(ollama.os.environ, {}, clear=False):
            self.assertTrue(ollama.start_server(**kw))
        return captured

    def test_the_log_path_receives_the_servers_output(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "logs" / "ollama.log"
            got = self._spawn(log_path=log)
            self.assertEqual(got["stderr"], subprocess.STDOUT)
            self.assertIn("more system memory", log.read_text(encoding="utf-8"))

    def test_an_oversized_log_is_truncated_first(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "ollama.log"
            log.write_bytes(b"x" * (ollama.OLLAMA_LOG_MAX_BYTES + 1))
            self._spawn(log_path=log)
            text = log.read_text(encoding="utf-8")
            self.assertNotIn("xxxx", text)
            self.assertIn("more system memory", text)

    def test_an_unwritable_log_falls_back_to_devnull_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            blocker = Path(d) / "file"
            blocker.write_text("i am a file, not a directory")
            got = self._spawn(log_path=blocker / "ollama.log")
            self.assertEqual(got["stdout"], subprocess.DEVNULL)
            self.assertEqual(got["stderr"], subprocess.DEVNULL)

    def test_default_log_lives_in_the_collected_log_dir(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(ollama.os.environ, {"ARGUS_LOG_DIR": d}):
            self.assertEqual(ollama.default_log_path(), Path(d) / "ollama.log")


class TheDiagnoseZipCarriesTheScannerStatus(unittest.TestCase):
    def test_english_rules_status_and_ollama_log_are_included(self):
        from cvti.diagnostics import build_bundle
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(ollama.os.environ, {"ARGUS_LOG_DIR": str(Path(d) / "logs")}):
            out = Path(d)
            (out / "logs").mkdir()
            (out / "logs" / "ollama.log").write_text("runner exited: no AVX\n")
            (out / "english_rules_status.json").write_text(json.dumps(
                {"generated_at": 1, "cameras": {"gate": {"scans": 3, "errors": 3,
                                                          "last_outcome": "call failed"}}}))
            (out / "monitor.log").write_text("hello\n")
            path = build_bundle(out)
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
            self.assertIn("english_rules_status.json", names)
            self.assertIn("logs/ollama.log", names)
            self.assertIn("logs/monitor.log", names)


if __name__ == "__main__":
    unittest.main()
