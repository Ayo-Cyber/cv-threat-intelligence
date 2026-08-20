"""Build and sign a release update (vendor side of EP-04-T3).

    python tools/make_update.py keygen --out release.key
    python tools/make_update.py sign --key release.key --version 0.10.0 \
        --dir dist/CVTI-Console --url https://updates.example.com/argus-0.10.0.zip \
        --notes "fixes the frame publisher auth" --out-dir updates/

`sign` produces `argus-<version>.zip` and a `manifest.json` whose signature is
Ed25519 over the archive bytes. Host both anywhere (a GitHub release, any
static file host) and point sites' update URL at the manifest.

THE PRIVATE KEY IS THE PRODUCT'S ROOT OF TRUST. It signs what every site will
install and run. Keep it off the repository, off the sites, and ideally off any
machine that doesn't need it. Losing it means you cannot ship updates; leaking
it means someone else can.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _require_crypto():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        return Ed25519PrivateKey
    except ImportError:
        raise SystemExit("this tool needs the 'cryptography' package: pip install cryptography")


def cmd_keygen(args) -> int:
    Ed25519PrivateKey = _require_crypto()
    from cryptography.hazmat.primitives import serialization
    key = Ed25519PrivateKey.generate()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists — refusing to overwrite a release key "
                         f"(use --force if you truly mean it)")
    out.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()))
    os.chmod(out, 0o600)
    public_hex = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw).hex()
    print(f"private key : {out}  (mode 600 — keep this OFF the repo and OFF the sites)")
    print(f"public key  : {public_hex}")
    print("")
    print("Embed the public key as VENDOR_PUBLIC_KEY_HEX in cvti/updates.py.")
    return 0


def _zip_dir(directory: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(directory))
    return buf.getvalue()


def cmd_sign(args) -> int:
    Ed25519PrivateKey = _require_crypto()
    key = Ed25519PrivateKey.from_private_bytes(Path(args.key).read_bytes())

    source = Path(args.dir)
    if not source.is_dir():
        raise SystemExit(f"{source} is not a directory")
    archive = _zip_dir(source)
    digest = hashlib.sha256(archive).hexdigest()
    signature = key.sign(archive).hex()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"argus-{args.version}.zip"
    archive_path.write_bytes(archive)
    manifest = {
        "version": args.version,
        "url": args.url or archive_path.name,
        "sha256": digest,
        "signature": signature,
        "notes": args.notes,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Sanity: verify with the embedded public key before publishing, so a wrong
    # or rotated key is caught at build time rather than by every site refusing.
    from cvti.updates import SignatureError, verify_signature
    try:
        verify_signature(archive, signature)
        verdict = "verifies against the key embedded in cvti/updates.py"
    except SignatureError:
        verdict = ("!! does NOT verify against cvti/updates.py's embedded key — "
                   "sites will refuse this. Wrong key file?")
    print(f"archive  : {archive_path}  ({len(archive) / 1e6:.1f} MB)")
    print(f"sha256   : {digest}")
    print(f"manifest : {out_dir / 'manifest.json'}")
    print(f"signature: {verdict}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    kg = sub.add_parser("keygen", help="generate the vendor release keypair")
    kg.add_argument("--out", default="release.key")
    kg.add_argument("--force", action="store_true")
    kg.set_defaults(func=cmd_keygen)

    sg = sub.add_parser("sign", help="zip a build directory and sign it")
    sg.add_argument("--key", required=True, help="private key from keygen")
    sg.add_argument("--version", required=True)
    sg.add_argument("--dir", required=True, help="directory to package")
    sg.add_argument("--url", default="", help="where the archive will be hosted")
    sg.add_argument("--notes", default="")
    sg.add_argument("--out-dir", default="updates")
    sg.set_defaults(func=cmd_sign)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
