"""Download AI installs the object-recognition model (SigLIP) too.

The installer must: fetch the pinned files, resume an interrupted download,
refuse a corrupt file, refuse to start without disk space, and never report
ready until the model has loaded and produced an embedding.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.object_watch import model_install as mi


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with HTTP Range support, recording what clients asked for."""
    seen_ranges: list[str] = []
    corrupt: set[str] = set()

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        name = self.path.rsplit("/", 1)[-1]
        path = Path(self.directory) / name
        if not path.is_file():
            self.send_error(404); return
        data = path.read_bytes()
        if name in self.corrupt:
            data = b"X" * len(data)
        rng = self.headers.get("Range")
        start = 0
        if rng:
            self.seen_ranges.append(rng)
            start = int(rng.replace("bytes=", "").split("-")[0])
            if start >= len(data):
                self.send_error(416); return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(body)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class InstallerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = tempfile.TemporaryDirectory()
        files = {"config.json": b'{"model_type": "siglip"}\n',
                 "preprocessor_config.json": b'{"size": 224}\n',
                 "model.safetensors": os.urandom(3 * 1024 * 1024 + 123)}
        for n, b in files.items():
            (Path(cls.src.name) / n).write_bytes(b)
        cls.spec = mi.ModelSpec(
            repo="test/siglip", revision="deadbeef", license="apache-2.0", display_size="3 MB",
            files=tuple(mi.ModelFile(n, len(b), _sha(b)) for n, b in files.items()))
        import functools
        handler = functools.partial(_RangeHandler, directory=cls.src.name)
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        os.environ["ARGUS_SIGLIP_BASE_URL"] = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close()
        os.environ.pop("ARGUS_SIGLIP_BASE_URL", None)
        cls.src.cleanup()

    def setUp(self):
        _RangeHandler.seen_ranges.clear(); _RangeHandler.corrupt.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "siglip"
        self.events: list[dict] = []
        mi._state.update(state="idle", percent=0, detail="", bytes_done=0, bytes_total=0, error="")

    def tearDown(self):
        self.tmp.cleanup()

    def _progress(self, **kw):
        self.events.append(kw)

    def _good_smoke(self, path):
        self.assertTrue((path / "model.safetensors").is_file(), "smoke ran before the files landed")
        return {"ok": True, "fingerprint": "siglip-test", "dimensions": 768}

    def test_fresh_install_downloads_verifies_smokes_and_marks_ready(self):
        marker = mi.install(self.dest, spec=self.spec, smoke=self._good_smoke, progress=self._progress)
        self.assertTrue(marker["verified"])
        self.assertEqual(marker["dimensions"], 768)
        for f in self.spec.files:
            self.assertEqual((self.dest / f.name).stat().st_size, f.size)
        self.assertFalse(list(self.dest.glob("*.part")), "part files left behind")
        self.assertTrue(mi.is_installed(self.dest, self.spec))
        st = mi.status(self.dest, self.spec)
        self.assertEqual((st["state"], st["percent"], st["fingerprint"]), ("ready", 100, "siglip-test"))
        states = [e.get("state") for e in self.events if "state" in e]
        self.assertIn("verifying", states)
        self.assertEqual(states[-1], "ready")
        self.assertEqual(_RangeHandler.seen_ranges, [])

    def test_an_interrupted_download_resumes_where_it_stopped(self):
        self.dest.mkdir(parents=True)
        big = next(f for f in self.spec.files if f.name == "model.safetensors")
        whole = (Path(self.src.name) / big.name).read_bytes()
        (self.dest / (big.name + ".part")).write_bytes(whole[: len(whole) // 2])
        mi.install(self.dest, spec=self.spec, smoke=self._good_smoke, progress=self._progress)
        self.assertEqual(_RangeHandler.seen_ranges, [f"bytes={len(whole) // 2}-"])
        self.assertEqual((self.dest / big.name).read_bytes(), whole)

    def test_a_corrupt_download_is_refused_and_the_part_file_removed(self):
        _RangeHandler.corrupt.add("model.safetensors")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            mi.install(self.dest, spec=self.spec, smoke=self._good_smoke, progress=self._progress)
        self.assertFalse((self.dest / "model.safetensors").exists())
        self.assertFalse((self.dest / "model.safetensors.part").exists(),
                         "a corrupt part file would be resumed forever")
        self.assertFalse(mi.is_installed(self.dest, self.spec))

    def test_a_failed_load_test_means_not_ready(self):
        def bad_smoke(path):
            return {"ok": False, "error": "SigLIP embedding backend unavailable: boom"}
        with self.assertRaisesRegex(RuntimeError, "failed its check"):
            mi.install(self.dest, spec=self.spec, smoke=bad_smoke, progress=self._progress)
        self.assertFalse((self.dest / mi.MARKER).exists())
        self.assertFalse(mi.is_installed(self.dest, self.spec))

    def test_no_disk_space_is_refused_before_any_download(self):
        usage = shutil.disk_usage(self.tmp.name)._replace(free=1024)
        with patch.object(mi.shutil, "disk_usage", return_value=usage):
            with self.assertRaisesRegex(RuntimeError, "not enough disk space"):
                mi.install(self.dest, spec=self.spec, smoke=self._good_smoke)
        self.assertFalse(list(self.dest.glob("*")), "started downloading with no room")

    def test_a_file_that_changes_after_install_is_no_longer_ready(self):
        mi.install(self.dest, spec=self.spec, smoke=self._good_smoke)
        (self.dest / "config.json").write_text("{}")
        self.assertFalse(mi.is_installed(self.dest, self.spec))

    def test_background_install_reports_through_status(self):
        done = threading.Event()
        def smoke(path):
            done.set(); return {"ok": True, "fingerprint": "f", "dimensions": 768}
        first = mi.start_install(self.dest, spec=self.spec, smoke=smoke)
        self.assertIn(first["state"], ("downloading", "verifying", "ready"))
        self.assertTrue(done.wait(20))
        for _ in range(200):
            st = mi.status(self.dest, self.spec)
            if st["state"] == "ready":
                break
            time.sleep(0.05)
        self.assertEqual(st["state"], "ready")
        self.assertTrue(st["installed"])
        # a second start on an installed model is a no-op that says ready
        self.assertEqual(mi.start_install(self.dest, spec=self.spec, smoke=smoke)["state"], "ready")

    def test_background_failure_is_an_error_state_with_the_reason(self):
        _RangeHandler.corrupt.add("config.json")
        mi.start_install(self.dest, spec=self.spec, smoke=self._good_smoke)
        for _ in range(200):
            st = mi.status(self.dest, self.spec)
            if st["state"] == "error":
                break
            time.sleep(0.05)
        self.assertEqual(st["state"], "error")
        self.assertIn("checksum mismatch", st["detail"])


class SpecTest(unittest.TestCase):
    def test_the_pinned_checkpoint_is_the_one_demi_validated(self):
        self.assertEqual(mi.SIGLIP.repo, "google/siglip-base-patch16-224")
        self.assertEqual(mi.SIGLIP.revision, "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed")
        self.assertEqual({f.name for f in mi.SIGLIP.files},
                         {"config.json", "preprocessor_config.json", "model.safetensors"})
        self.assertEqual(mi.SIGLIP.license, "apache-2.0")
        self.assertIn("/resolve/7fd15f06", mi.file_url(mi.SIGLIP, "config.json"))


class SmokeModuleTest(unittest.TestCase):
    def test_a_missing_directory_is_a_clear_failure_not_a_traceback(self):
        from cvti.object_watch import smoke
        out = smoke.check(Path(tempfile.mkdtemp()) / "nope")
        self.assertFalse(out["ok"])
        self.assertIn("model directory is missing", out["error"])

    def test_the_dev_smoke_command_runs_the_module(self):
        cmd = mi.smoke_command(Path("/x/siglip"))
        self.assertEqual(cmd[1:], ["-m", "cvti.object_watch.smoke", "/x/siglip"])


class BackendAdoptionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["ARGUS_MODELS_DIR"] = str(self.root / "models")
        self.installed = self.root / "models" / "siglip"
        self.installed.mkdir(parents=True)
        for f in mi.SIGLIP.files:                       # right sizes, no hashing on poll
            with (self.installed / f.name).open("wb") as h:
                h.truncate(f.size)
        (self.installed / mi.MARKER).write_text(json.dumps(
            {"revision": mi.SIGLIP.revision, "verified": True, "fingerprint": "siglip-x", "dimensions": 768}))
        from cvti.app.console_backend import ConsoleBackend
        self.be = ConsoleBackend.__new__(ConsoleBackend)
        self.be.db_path = str(self.root / "site" / "events.db")
        (self.root / "site").mkdir()

    def tearDown(self):
        os.environ.pop("ARGUS_MODELS_DIR", None)
        self.tmp.cleanup()

    def test_a_site_without_a_model_adopts_the_installed_one(self):
        self.be._adopt_recognition_model()
        doc = json.loads((self.root / "site" / "object_library" / "runtime.json").read_text())
        self.assertEqual(Path(doc["model_path"]), self.installed.resolve())
        self.assertEqual(doc["backend"], "siglip")

    def test_a_site_with_its_own_model_directory_is_left_alone(self):
        own = self.root / "site" / "own-siglip"; own.mkdir()
        (own / "config.json").write_text(json.dumps({"model_type": "siglip", "vision_config": {"hidden_size": 768}}))
        (own / "preprocessor_config.json").write_text("{}")
        (own / "model.safetensors").write_bytes(b"w")
        from cvti.object_watch.runtime_config import ObjectWatchConfig, write_config
        write_config(self.root / "site", ObjectWatchConfig(model_path=own))
        self.be._adopt_recognition_model()
        doc = json.loads((self.root / "site" / "object_library" / "runtime.json").read_text())
        self.assertEqual(Path(doc["model_path"]).resolve(), own.resolve())

    def test_status_carries_object_watch_readiness(self):
        out = self.be.recognition_model_status()
        self.assertEqual(out["state"], "ready")
        self.assertIn("object_watch", out)
        self.assertIn(out["object_watch"]["status"], ("structurally_available", "degraded", "unavailable"))


class RoutesTest(unittest.TestCase):
    def test_the_api_exposes_the_recognition_model(self):
        from cvti.api import writes
        rows = [r for v in vars(writes).values() if isinstance(v, list) for r in v if hasattr(r, "bridge")]
        bridges = {r.bridge: (r.verb, r.path) for r in rows}
        self.assertEqual(bridges["recognition_model_status"], ("GET", "/engine/models/recognition"))
        self.assertEqual(bridges["pull_recognition_model"], ("POST", "/engine/models/recognition/pull"))


if __name__ == "__main__":
    unittest.main()
