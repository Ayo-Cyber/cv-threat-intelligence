#!/usr/bin/env python3
"""Build the Argus desktop bundle (console + engine) for the CURRENT OS.

    python packaging/build.py            # build for this OS
    python packaging/build.py --clean    # wipe build/ and dist/ first

PyInstaller cannot cross-compile, so this only ever produces an artifact for the
OS it runs on:
    macOS   -> dist/Argus.app
    Windows -> dist/Argus/Argus.exe
    Linux   -> dist/Argus/Argus

To get all three from one commit, push and let the GitHub Actions matrix build
them (.github/workflows/build-app.yml).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "argus.spec"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", action="store_true", help="Remove build/ and dist/ first.")
    ap.add_argument("--dmg", action="store_true",
                    help="macOS only: also package the .app into dist/Argus.dmg.")
    args = ap.parse_args()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not installed. Run:  pip install pyinstaller")
        return 1

    if args.clean:
        for d in ("build", "dist"):
            shutil.rmtree(ROOT / d, ignore_errors=True)
        print("cleaned build/ and dist/")

    # Best-effort: assemble the self-contained playback demo so the app can
    # demo itself on any machine. Needs a prior run (runs/demo) + clips; if
    # those aren't present (e.g. CI), the app just builds without demo mode.
    if not (ROOT / "packaging" / "demo_data" / "events.db").exists():
        rc = subprocess.call([sys.executable, str(ROOT / "packaging" / "build_demo_data.py")],
                             cwd=str(ROOT))
        if rc != 0:
            print("(no demo_data assembled — bundling app without playback demo)")

    # The AI runtime rides inside the bundle only if it has been fetched.
    # Not fatal — a build without it still works, the app just directs the
    # user to install Ollama — but a RELEASE build should never skip this.
    import platform
    plat = {"win32": "windows", "darwin": "darwin"}.get(sys.platform, "linux")
    if not (ROOT / "vendor" / "ollama" / plat).is_dir():
        print("NOTE: vendor/ollama/%s missing — run scripts/fetch_ollama.%s first "
              "to ship the AI runtime inside the bundle." % (plat, "bat" if plat == "windows" else "sh"))

    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm",
           "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build")]
    print("running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        print(f"\nbuild FAILED (exit {rc})")
        return rc

    dist = ROOT / "dist"
    if sys.platform == "darwin":
        art = dist / "Argus.app"
    elif sys.platform.startswith("win"):
        art = dist / "Argus" / "Argus.exe"
    else:
        art = dist / "Argus" / "Argus"
    print(f"\nbuild OK -> {art}" if art.exists() else f"\nbuild finished but artifact missing: {art}")

    if args.dmg:
        if sys.platform != "darwin":
            print("--dmg is macOS-only; skipping.")
        else:
            print("packaging .dmg …")
            subprocess.call(["bash", str(ROOT / "packaging" / "make_dmg.sh")], cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
