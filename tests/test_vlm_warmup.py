"""The model stays hot (latency audit 1 Sep, V3).

Ollama's default unloads a model after 5 idle minutes, so on a quiet site the
first alert after a lull paid a full ~3 GB cold load inside its own verify
call. Two promises here:

- the server WE spawn is told to never unload (OLLAMA_KEEP_ALIVE=-1), with an
  operator's explicit env override still winning;
- the engine starts loading the model the moment it starts, in a background
  thread, so the first verdict finds the weights resident — and a failed
  warmup costs exactly nothing (the old behaviour: the first call loads).
"""
from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib import error as urlerror

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.verification import ollama


class SpawnedServerEnvTests(unittest.TestCase):
    def _spawn_env(self, preset_env: dict | None = None) -> dict:
        captured = {}

        def fake_popen(cmd, stdout=None, stderr=None, env=None):
            captured["env"] = env
            return mock.Mock()

        with mock.patch.object(ollama, "ollama_binary", return_value="/fake/ollama"), \
             mock.patch.object(ollama.os, "stat", side_effect=OSError), \
             mock.patch.object(ollama.subprocess, "Popen", fake_popen), \
             mock.patch.dict(ollama.os.environ, preset_env or {}, clear=False):
            if not preset_env:
                # A developer's own OLLAMA_KEEP_ALIVE must not leak into the
                # no-override case; patch.dict restores it afterwards.
                ollama.os.environ.pop("OLLAMA_KEEP_ALIVE", None)
            self.assertTrue(ollama.start_server())
        return captured["env"]

    def test_spawned_server_never_unloads_the_model(self):
        self.assertEqual(self._spawn_env()["OLLAMA_KEEP_ALIVE"], "-1")

    def test_an_operator_override_still_wins(self):
        env = self._spawn_env({"OLLAMA_KEEP_ALIVE": "10m"})
        self.assertEqual(env["OLLAMA_KEEP_ALIVE"], "10m")


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class WarmupTests(unittest.TestCase):
    def test_warmup_is_ollamas_documented_preload(self):
        """POST /api/generate, a model, keep_alive -1, and NO prompt — the
        server loads weights and returns without generating a single token."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["payload"] = json.loads(req.data.decode())
            return _Resp(b"{}")

        with mock.patch.object(ollama.urlrequest, "urlopen", fake_urlopen):
            self.assertTrue(ollama.warmup_model("gemma3:4b", "http://localhost:11434"))
        self.assertTrue(captured["url"].endswith("/api/generate"))
        self.assertEqual(captured["payload"]["model"], "gemma3:4b")
        self.assertEqual(captured["payload"]["keep_alive"], -1)
        self.assertNotIn("prompt", captured["payload"])

    def test_unreachable_server_is_a_false_not_a_crash(self):
        with mock.patch.object(ollama.urlrequest, "urlopen",
                               side_effect=urlerror.URLError("down")):
            self.assertFalse(ollama.warmup_model("gemma3:4b"))

    def test_native_host_is_derived_from_the_gate_base_url(self):
        self.assertEqual(ollama.host_from_base_url("http://localhost:11434/v1"),
                         "http://localhost:11434")
        self.assertEqual(ollama.host_from_base_url("http://10.0.0.5:11434/v1/"),
                         "http://10.0.0.5:11434")
        self.assertEqual(ollama.host_from_base_url(""), ollama.DEFAULT_HOST)


class EngineWarmupThreadTests(unittest.TestCase):
    def test_local_providers_warm_in_the_background(self):
        from cvti.serving.pipeline import warm_gate_model_async
        with mock.patch.object(ollama, "ensure_server", return_value=True) as ensured, \
             mock.patch.object(ollama, "warmup_model", return_value=True) as warmed:
            thread = warm_gate_model_async("ollama", "gemma3:4b",
                                           "http://localhost:11434/v1")
            self.assertIsNotNone(thread)
            thread.join(timeout=5.0)
        ensured.assert_called_once_with("http://localhost:11434")
        warmed.assert_called_once_with("gemma3:4b", "http://localhost:11434")

    def test_an_empty_model_warms_the_shipped_default(self):
        from cvti.contracts import LOCAL_VLM_MODEL
        from cvti.serving.pipeline import warm_gate_model_async
        with mock.patch.object(ollama, "ensure_server", return_value=True), \
             mock.patch.object(ollama, "warmup_model", return_value=True) as warmed:
            warm_gate_model_async("local", "", "").join(timeout=5.0)
        self.assertEqual(warmed.call_args.args[0], LOCAL_VLM_MODEL)

    def test_cloud_gates_have_nothing_to_warm(self):
        from cvti.serving.pipeline import warm_gate_model_async
        with mock.patch.object(ollama, "ensure_server") as ensured:
            self.assertIsNone(warm_gate_model_async("anthropic", "claude", ""))
            self.assertIsNone(warm_gate_model_async("mock", "", ""))
        ensured.assert_not_called()

    def test_a_server_that_never_comes_up_skips_the_warmup(self):
        from cvti.serving.pipeline import warm_gate_model_async
        with mock.patch.object(ollama, "ensure_server", return_value=False), \
             mock.patch.object(ollama, "warmup_model") as warmed:
            warm_gate_model_async("ollama", "gemma3:4b", "").join(timeout=5.0)
        warmed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
