"""Signed updates (EP-04-T3).

The property that matters most is the refusal: a site installs nothing the
vendor release key did not sign. There is no override, no "skip verification",
and a missing signature is treated exactly like a forged one.
"""
import hashlib
import io
import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    HAVE_CRYPTO = True
except ImportError:              # the module fails closed without it; so do we
    HAVE_CRYPTO = False

from cvti.updates import (
    SignatureError,
    UpdateError,
    UpdateInfo,
    UpdateManager,
    parse_version,
    verify_signature,
)

# A throwaway keypair for the tests — the real private key never touches CI.
if HAVE_CRYPTO:
    _KEY = Ed25519PrivateKey.generate()
    _PUB_HEX = _KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw).hex()


def _archive(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _release(version: str, files: dict) -> tuple:
    """(archive_bytes, manifest_dict) signed with the test key."""
    archive = _archive(files)
    return archive, {
        "version": version,
        "url": f"https://updates.test/argus-{version}.zip",
        "sha256": hashlib.sha256(archive).hexdigest(),
        "signature": _KEY.sign(archive).hex(),
        "notes": "test release",
    }


class _Wire:
    """Serves the manifest and archives like a static file host."""

    def __init__(self):
        self.files: dict = {}

    def fetch(self, url: str) -> bytes:
        return self.files[url]


def _manager(tmp, wire, version="0.9.0"):
    return UpdateManager(tmp, current_version=version, public_key_hex=_PUB_HEX,
                         fetch=wire.fetch)


@unittest.skipUnless(HAVE_CRYPTO, "cryptography not installed")
class SignatureTest(unittest.TestCase):
    def test_a_vendor_signed_archive_verifies(self):
        archive, manifest = _release("1.0.0", {"app.py": "print('hi')"})
        verify_signature(archive, manifest["signature"], _PUB_HEX)   # no raise

    def test_a_forged_signature_is_refused(self):
        archive, _ = _release("1.0.0", {"app.py": "x"})
        other = Ed25519PrivateKey.generate()
        forged = other.sign(archive).hex()
        with self.assertRaises(SignatureError):
            verify_signature(archive, forged, _PUB_HEX)

    def test_a_tampered_archive_is_refused_even_with_a_real_signature(self):
        archive, manifest = _release("1.0.0", {"app.py": "x"})
        tampered = archive[:-1] + bytes([archive[-1] ^ 0xFF])
        with self.assertRaises(SignatureError):
            verify_signature(tampered, manifest["signature"], _PUB_HEX)

    def test_an_unsigned_manifest_is_not_an_update(self):
        with self.assertRaises(SignatureError):
            UpdateInfo.from_manifest({"version": "1.0.0", "url": "x", "sha256": "y"})

    def test_garbage_signature_hex_is_refused_not_crashed(self):
        archive, _ = _release("1.0.0", {"a": "b"})
        with self.assertRaises(SignatureError):
            verify_signature(archive, "zz-not-hex", _PUB_HEX)


class VersionTest(unittest.TestCase):
    def test_ordering(self):
        self.assertGreater(parse_version("0.10.0"), parse_version("0.9.9"))
        self.assertGreater(parse_version("1.0.0"), parse_version("0.99.0"))
        self.assertEqual(parse_version("v1.2.3"), parse_version("1.2.3"))

    def test_nonsense_compares_low_rather_than_crashing(self):
        self.assertLess(parse_version("beta"), parse_version("0.0.1"))


@unittest.skipUnless(HAVE_CRYPTO, "cryptography not installed")
class CheckTest(unittest.TestCase):
    def test_a_newer_version_is_offered(self):
        wire = _Wire()
        _, manifest = _release("0.10.0", {"a": "b"})
        wire.files["https://updates.test/manifest.json"] = json.dumps(manifest).encode()
        with TemporaryDirectory() as tmp:
            info = _manager(tmp, wire).check("https://updates.test/manifest.json")
            self.assertEqual(info.version, "0.10.0")

    def test_the_same_or_older_version_is_not_offered(self):
        wire = _Wire()
        for offered in ("0.9.0", "0.8.5"):
            _, manifest = _release(offered, {"a": "b"})
            wire.files["https://updates.test/manifest.json"] = json.dumps(manifest).encode()
            with TemporaryDirectory() as tmp:
                self.assertIsNone(_manager(tmp, wire).check("https://updates.test/manifest.json"))

    def test_check_downloads_nothing_and_touches_nothing(self):
        # The customer controls timing: checking is reading, only reading.
        wire = _Wire()
        archive, manifest = _release("0.10.0", {"a": "b"})
        wire.files["https://updates.test/manifest.json"] = json.dumps(manifest).encode()
        # the archive is deliberately NOT on the wire — a download attempt would KeyError
        with TemporaryDirectory() as tmp:
            mgr = _manager(tmp, wire)
            mgr.check("https://updates.test/manifest.json")
            self.assertFalse((Path(tmp) / "versions").exists())
            self.assertEqual(mgr.pointer()["version"], "0.9.0")


@unittest.skipUnless(HAVE_CRYPTO, "cryptography not installed")
class ApplyAndRollbackTest(unittest.TestCase):
    """The acceptance cycle: a full upgrade, then a full rollback."""

    def _wire_release(self, wire, version, files):
        archive, manifest = _release(version, files)
        wire.files[manifest["url"]] = archive
        wire.files["https://updates.test/manifest.json"] = json.dumps(manifest).encode()
        return UpdateInfo.from_manifest(manifest)

    def test_full_upgrade_then_rollback_cycle(self):
        wire = _Wire()
        with TemporaryDirectory() as tmp:
            mgr = _manager(tmp, wire, version="0.9.0")

            # --- upgrade ---
            info = self._wire_release(wire, "0.10.0", {"app.py": "print('new')",
                                                       "web/index.html": "<html>"})
            result = mgr.apply(info)
            self.assertTrue(result["ok"])
            self.assertTrue(result["restart_required"])
            self.assertEqual(mgr.pointer(), {**mgr.pointer(), "version": "0.10.0",
                                             "previous": "0.9.0"})
            installed = Path(tmp) / "versions" / "0.10.0"
            self.assertEqual((installed / "app.py").read_text(), "print('new')")
            self.assertTrue((installed / "web" / "index.html").exists())
            self.assertEqual(mgr.current_dir(), installed)

            # --- rollback ---
            result = mgr.rollback()
            self.assertTrue(result["ok"])
            self.assertEqual(mgr.pointer()["version"], "0.9.0")
            self.assertEqual(result["rolled_back_from"], "0.10.0")
            # the bad version's files still exist; nothing was re-downloaded
            self.assertTrue(installed.exists())

    def test_a_tampered_download_is_refused_and_nothing_changes(self):
        wire = _Wire()
        with TemporaryDirectory() as tmp:
            mgr = _manager(tmp, wire)
            info = self._wire_release(wire, "0.10.0", {"app.py": "x"})
            original = wire.files[info.url]
            wire.files[info.url] = original[:10] + bytes([original[10] ^ 0xFF]) + original[11:]
            with self.assertRaises(UpdateError):
                mgr.apply(info)
            self.assertEqual(mgr.pointer()["version"], "0.9.0")
            self.assertFalse((Path(tmp) / "versions" / "0.10.0").exists())

    def test_a_forged_release_is_refused_at_apply(self):
        wire = _Wire()
        with TemporaryDirectory() as tmp:
            mgr = _manager(tmp, wire)
            archive = _archive({"evil.py": "x"})
            attacker = Ed25519PrivateKey.generate()
            manifest = {"version": "0.10.0", "url": "https://updates.test/evil.zip",
                        "sha256": hashlib.sha256(archive).hexdigest(),
                        "signature": attacker.sign(archive).hex()}
            wire.files[manifest["url"]] = archive
            with self.assertRaises(SignatureError):
                mgr.apply(UpdateInfo.from_manifest(manifest))
            self.assertFalse((Path(tmp) / "versions").exists())

    def test_an_archive_with_path_traversal_is_refused(self):
        wire = _Wire()
        with TemporaryDirectory() as tmp:
            mgr = _manager(tmp, wire)
            info = self._wire_release(wire, "0.10.0", {"../outside.py": "x"})
            with self.assertRaises(UpdateError):
                mgr.apply(info)

    def test_rollback_with_no_previous_says_so(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(UpdateError):
                _manager(tmp, _Wire()).rollback()

    def test_old_versions_are_pruned_but_the_rollback_target_survives(self):
        wire = _Wire()
        with TemporaryDirectory() as tmp:
            mgr = _manager(tmp, wire, version="0.9.0")
            mgr.apply(self._wire_release(wire, "0.10.0", {"a": "1"}))
            mgr.apply(self._wire_release(wire, "0.11.0", {"a": "2"}))
            names = {p.name for p in (Path(tmp) / "versions").iterdir()}
            self.assertEqual(names, {"0.10.0", "0.11.0"})
            self.assertEqual(mgr.pointer()["previous"], "0.10.0")


class FailClosedTest(unittest.TestCase):
    def test_without_the_crypto_library_updates_refuse_rather_than_skip(self):
        # Simulate the library being absent regardless of the environment.
        import builtins
        real_import = builtins.__import__

        def no_crypto(name, *a, **k):
            if name.startswith("cryptography"):
                raise ImportError("not installed")
            return real_import(name, *a, **k)

        builtins.__import__ = no_crypto
        try:
            with self.assertRaises(UpdateError) as ctx:
                verify_signature(b"bytes", "00" * 64)
            self.assertIn("refusing", str(ctx.exception))
        finally:
            builtins.__import__ = real_import


if __name__ == "__main__":
    unittest.main()
