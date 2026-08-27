"""Assert the AI runtime actually shipped inside the bundle, and can start.

'The ollama doesn't come in with it' (25 Aug) turned out to be about the 3.3 GB
MODEL, which downloads on first run by design — but nothing in CI could confirm
the RUNTIME was there at all. Now it can.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(bundle: str) -> int:
    root = Path(bundle).resolve()
    exe = "ollama.exe" if sys.platform == "win32" else "ollama"
    hits = list(root.rglob(f"vendor/ollama/*/{exe}"))
    if not hits:
        print(f"FAIL: no bundled AI runtime under {root} — the installer cannot "
              "verify anything on a machine without Ollama, which is the whole "
              "point of the one-installer promise")
        return 1
    binary = hits[0]
    size_mb = binary.stat().st_size / 1e6
    print(f"  bundled runtime: {binary.relative_to(root)} ({size_mb:.0f} MB)")

    runners = list(binary.parent.rglob("*ggml*"))
    if not runners:
        print("FAIL: the runtime has no ggml runner libraries — it cannot run a model")
        return 1
    print(f"  runner libraries: {len(runners)}")
    gpu = [r for r in runners if any(k in r.name.lower()
                                     for k in ("cuda", "rocm", "hip", "vulkan"))]
    if gpu:
        print(f"  NOTE: {len(gpu)} GPU runner(s) survived the prune — bundle is "
              "larger than it needs to be")

    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True,
                             text=True, timeout=120)
        print(f"  runtime answers --version: {(out.stdout or out.stderr).strip()[:60]}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: the bundled runtime would not execute: {exc}")
        return 1
    print("PASS — the AI runtime ships inside the bundle and runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "dist/Argus"))
