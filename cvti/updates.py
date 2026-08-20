"""Signed updates: ship a fix without a site visit (EP-04-T3).

During a pilot you *will* need to ship fixes. Without this, every fix is a site
visit or a phone call walking someone through a manual reinstall — which does
not scale past one customer and burns the goodwill the pilot exists to build.

The trust model, stated plainly:

- Updates are **Ed25519-signed** with a private key that never leaves the
  vendor. The matching public key is embedded here. A site will install
  nothing that key did not sign — there is no "skip verification" path, and a
  missing signature is treated exactly like a forged one.
- The **signature covers the archive bytes** (not the manifest), so a manifest
  that lies about its hash changes nothing: the archive is hashed and verified
  again locally before anything is unpacked.
- **The customer controls timing.** `check()` only reads; `apply()` runs only
  when a person asks for it. Nothing auto-updates mid-shift, ever — there is no
  code path from "an update exists" to "an update is installed" without a call
  a human makes.
- **Rollback is a first-class operation**, not a re-download: the previous
  version's files are kept on disk and switching back is a pointer change.

Layout under the install root:

    versions/<version>/...      the unpacked releases, immutable once written
    current.json                {"version": ..., "previous": ...}

Whatever launches Argus resolves `current.json` first. A failed start on a new
version is recovered by `rollback()` — one call, no network.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from cvti.logging_setup import get_logger

log = get_logger(__name__)

# The vendor release key (Ed25519, hex). The PRIVATE half lives with the
# vendor, offline — never in this repository, never on a site. Rotating it
# means shipping one last update signed by the old key that installs the new.
VENDOR_PUBLIC_KEY_HEX = "de06d0b0fb167d5556e3db42d2f7bb046c384a8c00ac34b0f4abb00a2afa1753"

MANIFEST_NAME = "manifest.json"


class UpdateError(Exception):
    """Anything that stops an update. The message is operator-facing."""


class SignatureError(UpdateError):
    """The archive is not one the vendor signed. Never installed, no override."""


@dataclass
class UpdateInfo:
    version: str
    url: str
    sha256: str
    signature: str          # hex Ed25519 over the archive bytes
    notes: str = ""
    published_at: str = ""

    @classmethod
    def from_manifest(cls, data: dict) -> "UpdateInfo":
        missing = [k for k in ("version", "url", "sha256", "signature") if not data.get(k)]
        if missing:
            # An unsigned manifest is not a lesser update — it is not an update.
            raise SignatureError(f"manifest is missing {', '.join(missing)}; refusing")
        return cls(version=str(data["version"]), url=str(data["url"]),
                   sha256=str(data["sha256"]), signature=str(data["signature"]),
                   notes=str(data.get("notes", "")),
                   published_at=str(data.get("published_at", "")))


def parse_version(v: str) -> tuple:
    """Lenient numeric-tuple ordering: '0.10.1' > '0.9.0'. Non-numeric parts
    compare as zero rather than crashing an update check."""
    out = []
    for part in str(v).lstrip("v").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out or [0])


def verify_signature(archive_bytes: bytes, signature_hex: str,
                     public_key_hex: str = VENDOR_PUBLIC_KEY_HEX) -> None:
    """Raise SignatureError unless `signature_hex` is the vendor's Ed25519
    signature over exactly these bytes. Fail closed: no crypto library means no
    updates, not unsigned updates."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise UpdateError(
            "update verification needs the 'cryptography' package, which is not "
            "installed — refusing to install anything unverified") from exc
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        key.verify(bytes.fromhex(signature_hex), archive_bytes)
    except (ValueError, InvalidSignature) as exc:
        raise SignatureError(
            "signature verification FAILED — this archive was not signed by the "
            "vendor release key and will not be installed") from exc


class UpdateManager:
    def __init__(self, install_root: str | Path, *, current_version: str,
                 public_key_hex: str = VENDOR_PUBLIC_KEY_HEX,
                 fetch=None) -> None:
        self.root = Path(install_root)
        self.versions_dir = self.root / "versions"
        self.pointer_path = self.root / "current.json"
        self.current_version = current_version
        self.public_key_hex = public_key_hex
        self._fetch = fetch or self._http_get

    @staticmethod
    def _http_get(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()

    # --- pointer ------------------------------------------------------------
    def pointer(self) -> dict:
        try:
            return json.loads(self.pointer_path.read_text())
        except (OSError, ValueError):
            return {"version": self.current_version, "previous": None}

    def _write_pointer(self, version: str, previous: str | None) -> None:
        # Write-then-rename so a crash mid-write cannot leave a half pointer —
        # the pointer is the single thing the launcher trusts.
        tmp = self.pointer_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"version": version, "previous": previous, "updated_at": time.time()},
            indent=2))
        tmp.replace(self.pointer_path)

    # --- check ---------------------------------------------------------------
    def check(self, manifest_url: str) -> UpdateInfo | None:
        """Read the manifest; return the update if it is newer. Reads only —
        nothing here downloads the archive or touches the install."""
        data = json.loads(self._fetch(manifest_url).decode("utf-8"))
        info = UpdateInfo.from_manifest(data)
        if parse_version(info.version) <= parse_version(self.pointer()["version"]):
            log.info("update check: %s is current (offered %s)",
                     self.pointer()["version"], info.version)
            return None
        log.info("update check: %s available (running %s)",
                 info.version, self.pointer()["version"])
        return info

    # --- apply ---------------------------------------------------------------
    def apply(self, info: UpdateInfo) -> dict:
        """Download, verify, unpack, switch. Called only on a human's say-so.

        Order matters: every check happens before anything existing is touched,
        so a failure at any step leaves the running install exactly as it was.
        """
        archive = self._fetch(info.url)

        digest = hashlib.sha256(archive).hexdigest()
        if digest != info.sha256.lower():
            raise UpdateError(
                f"download does not match the manifest hash ({digest[:12]}… != "
                f"{info.sha256[:12]}…) — corrupted or tampered; not installed")
        verify_signature(archive, info.signature, self.public_key_hex)

        target = self.versions_dir / info.version
        if target.exists():
            shutil.rmtree(target)          # a half-unpacked earlier attempt
        target.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=self.root if self.root.exists() else None) as tmp:
            staging = Path(tmp) / "unpack"
            with zipfile.ZipFile(__import__("io").BytesIO(archive)) as zf:
                for name in zf.namelist():
                    # Refuse traversal — the archive is signed, but defence in
                    # depth costs three lines.
                    if name.startswith(("/", "..")) or ".." in Path(name).parts:
                        raise UpdateError(f"archive contains an unsafe path: {name!r}")
                zf.extractall(staging)
            shutil.move(str(staging), str(target))

        previous = self.pointer()["version"]
        self._write_pointer(info.version, previous)
        self._prune(keep={info.version, previous})
        log.info("update applied: %s -> %s (previous kept for rollback); "
                 "restart to run it", previous, info.version)
        return {"ok": True, "installed": info.version, "previous": previous,
                "restart_required": True}

    def rollback(self) -> dict:
        """Switch back to the previous version. No network, no re-download."""
        ptr = self.pointer()
        previous = ptr.get("previous")
        if not previous:
            raise UpdateError("nothing to roll back to — no previous version recorded")
        # The as-shipped version never lives under versions/ — its files ARE the
        # original install, so a pointer change alone restores it.
        if previous != self.current_version and not (self.versions_dir / previous).exists():
            raise UpdateError(f"previous version {previous} is no longer on disk")
        self._write_pointer(previous, None)
        log.warning("rolled back: %s -> %s; restart to run it", ptr["version"], previous)
        return {"ok": True, "installed": previous, "rolled_back_from": ptr["version"],
                "restart_required": True}

    def current_dir(self) -> Path | None:
        """Where the launcher should run from, or None for 'as shipped'."""
        version = self.pointer()["version"]
        path = self.versions_dir / version
        return path if path.exists() else None

    def _prune(self, keep: set) -> None:
        """Two versions live on disk: the new one and the rollback target."""
        if not self.versions_dir.exists():
            return
        for entry in self.versions_dir.iterdir():
            if entry.is_dir() and entry.name not in keep:
                try:
                    shutil.rmtree(entry)
                except OSError:
                    log.warning("could not prune old version %s", entry.name,
                                exc_info=True)
