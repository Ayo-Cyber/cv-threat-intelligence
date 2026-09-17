"""Assert the PACKAGED desktop app contains both halves of the product.

This check exists because of what v1.8.12 shipped. Every gate in the build
tested the engine — decode video, publish frames, persist alerts, load the
models, run the AI runtime — and every one of them passed. None of them
looked at the interface, so five consecutive releases shipped the retired
PyQt shell while the React UI everyone demoed lived only in a developer's
checkout. Nothing failed, because nothing was asking.

So this asks. The app must carry the renderer, the Engine API binary the
shell spawns, the detection engine that API starts, and the model weights
they load.

    python tests/e2e/packaged_app_check.py Frontend/release
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _find_app(release: Path) -> tuple[Path, Path]:
    """(app root, resources dir) for whatever this OS just packaged."""
    for app in sorted(release.glob("mac*/*.app")):           # macOS
        return app, app / "Contents" / "Resources"
    for unpacked in sorted(release.glob("*-unpacked")):      # Windows, Linux
        return unpacked, unpacked / "resources"
    raise SystemExit(
        f"FAIL: no packaged app under {release} — electron-builder produced nothing to check")


def main(release_dir: str) -> int:
    release = Path(release_dir).resolve()
    app, resources = _find_app(release)
    print(f"  packaged app: {app.relative_to(release.parent.parent) if release.parent.parent in app.parents else app}")

    problems: list[str] = []
    exe = ".exe" if sys.platform == "win32" else ""
    engine_dir = resources / "engine"

    # 1. The renderer. Without the asar the window has nothing to show, which
    #    is precisely the failure this file was written for.
    asar = resources / "app.asar"
    if not asar.exists():
        problems.append("resources/app.asar is missing — the app has no UI to display")
    elif asar.stat().st_size < 100_000:
        problems.append(f"resources/app.asar is only {asar.stat().st_size} bytes — "
                        "the renderer build did not make it in")

    # 2. The two binaries the shell drives, and the weights they load.
    for rel in (f"argus-api{exe}", f"argus-engine{exe}"):
        if not (engine_dir / rel).exists():
            problems.append(f"resources/engine/{rel} is missing — "
                            "the UI would open onto an engine that cannot start")

    def _bundled(rel: Path) -> bool:
        """PyInstaller 6 keeps data under _internal/; older layouts keep it
        beside the binaries. Probe both, exactly as bundle_smoke.py does."""
        return (engine_dir / rel).exists() or (engine_dir / "_internal" / rel).exists()

    for rel in (Path("models") / "yolov8n.pt",
                Path("models") / "yolov8s-worldv2.pt",
                Path("vendor") / "clip" / "ViT-B-32.pt"):
        if not _bundled(rel):
            problems.append(f"resources/engine/{rel} is missing from the bundle")

    # 3. The API binary must actually RUN. A missing hidden import (uvicorn
    #    resolves its loop and protocols by string name) only shows up here,
    #    never in a file listing.
    api = engine_dir / f"argus-api{exe}"
    if api.exists():
        try:
            proc = subprocess.run([str(api), "--help"], capture_output=True,
                                  text=True, timeout=300)
            if proc.returncode != 0:
                problems.append(f"argus-api --help exited {proc.returncode}: "
                                f"{(proc.stderr or proc.stdout)[-400:]}")
        except (OSError, subprocess.SubprocessError) as exc:
            problems.append(f"argus-api would not execute: {exc}")

    if problems:
        print("\nFAIL — the packaged app is not shippable:")
        for p in problems:
            print("  -", p)
        return 1
    print("PASS — the packaged app carries the UI, both engine binaries and their weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Frontend/release"))
