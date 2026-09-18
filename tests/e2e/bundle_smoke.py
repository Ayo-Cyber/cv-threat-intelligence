"""Run the SHIPPED bundle against a real video and assert it produced the
artefacts a customer depends on (25 Aug).

`--help` proves the executable imports. It cannot tell you the decoder works,
that alerts persist, or that the frame publisher authenticates — and those are
exactly the things that broke in the field. This runs the actual bundled
binary, from a temp directory, with no repo venv on PATH.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(engine: str) -> int:
    engine = Path(engine).resolve()
    if not engine.exists():
        print(f"FAIL: no bundled engine at {engine}")
        return 1
    clips = sorted((ROOT / "data" / "test_clips").glob("*.mp4"))
    if not clips:
        print("SKIP: no demo clips in the checkout")
        return 0

    tmp = Path(tempfile.mkdtemp())
    site = tmp / "site.json"
    site.write_text(json.dumps({
        "name": "bundle-smoke", "notify": "console", "configured": True,
        "cameras": [{"id": "cam1", "source": str(clips[0]),
                     "config": "configs/all_threats_v1.json",
                     "crowd_formation": True, "crowd_min_people": 2,
                     "crowd_min_frames": 2}]}))

    cmd = [str(engine), "--site-config", str(site),
           "--gate-provider", "local", "--gate-base-url", "http://127.0.0.1:59999/v1",
           "--target-fps", "4", "--imgsz", "416", "--seconds", "70",
           "--gate-drain", "3", "--mobile-port", "0", "--output-dir", str(tmp),
           # What every installer runs: ONNX/CPU on Windows and Linux, and the
           # console spawns --device cpu on a Mac (#137: torch-MPS and the
           # local VLM fight for the GPU and the engine freezes mid-run —
           # seen again locally 16 Sep as a stall at frame 45).
           "--device", "cpu"]
    print("running:", " ".join(cmd[:3]), "...")
    # The frame publisher announces itself in <output_dir>/frames.json WHILE
    # the engine runs and removes the marker on a clean stop (#144: a stale
    # marker sent the app's tiles to a dead URL after Stop). So the marker
    # must be observed DURING the run — checking after exit is exactly the
    # check that failed the v1.8.12 tag build on all three OSes while the
    # bundle itself was fine.
    frames_snapshot: dict | None = None
    frames_path = tmp / "frames.json"
    # Output goes to FILES, never to PIPEs we are not reading: with pipes, the
    # engine blocks on its first log write once the pipe buffer fills (4 KB on
    # Windows), never reaches its --seconds exit, and the smoke kills it at
    # the deadline — the Windows v1.8.12 build died exactly like that while
    # macOS/Linux scraped by on their 64 KB buffers.
    out_path, err_path = tmp / "engine.stdout", tmp / "engine.stderr"
    with out_path.open("w", encoding="utf-8", errors="replace") as out_f, \
            err_path.open("w", encoding="utf-8", errors="replace") as err_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True)
        deadline = time.time() + 900
        while proc.poll() is None and time.time() < deadline:
            if frames_snapshot is None and frames_path.exists():
                try:
                    frames_snapshot = json.loads(frames_path.read_text())
                except (OSError, ValueError):
                    frames_snapshot = None      # half-written; try again next tick
            time.sleep(1.0)
        timed_out = proc.poll() is None
        if timed_out:
            proc.kill()
        proc.wait(timeout=60)
    out = out_path.read_text(encoding="utf-8", errors="replace")
    err = err_path.read_text(encoding="utf-8", errors="replace")
    proc.stdout, proc.stderr = out, err     # keep the old names for the checks below
    tail = out[-1500:] + err[-1500:]

    problems = []
    # W8: offline object rules need both halves IN the bundle — the world
    # weights and the CLIP checkpoint the engine pre-seeds into ~/.cache/clip.
    # Absence here is v1.8.8's silent-rot class again; fail loudly instead.
    bundle_dir = engine.parent
    for rel in (Path("models") / "yolov8s-worldv2.pt",
                Path("vendor") / "clip" / "ViT-B-32.pt"):
        # Every layout PyInstaller actually produces: plain onedir keeps datas
        # beside the binary or under _internal; a macOS .app puts the binary in
        # Contents/MacOS and the datas in Contents/Frameworks (mirrored into
        # Contents/Resources). The v1.8.10 tag build failed HERE on a green
        # bundle because only the first two were probed.
        cands = [bundle_dir / rel, bundle_dir / "_internal" / rel,
                 bundle_dir.parent / "Frameworks" / rel,
                 bundle_dir.parent / "Resources" / rel]
        if not any(c.exists() for c in cands):
            problems.append(f"open-vocab weights missing from the bundle: {rel}")
    # The two silent bundle-rot failures the 10 Sep field diagnostics exposed —
    # both were GREEN here while broken in every installer:
    # 1. a lazily-imported dep missing from the bundle sends ultralytics'
    #    AutoUpdate pip-installing at engine start (reads as "engine not
    #    starting" in the field);
    # 2. gitignored model weights silently absent -> a detector "configured
    #    but failed to load" -> theft detection off on every install.
    if "attempting AutoUpdate" in tail or "attempting AutoUpdate" in (proc.stdout or ""):
        problems.append("the bundle is missing a dependency badly enough that "
                        "ultralytics tried to PIP-INSTALL at runtime — a frozen "
                        "app must never do that (collect the dep in argus.spec)")
    health = tmp / "gate_health.json"
    if not health.exists():
        problems.append("the engine never wrote /health")
    else:
        doc = json.loads(health.read_text())
        cams = doc.get("cameras") or []
        if not cams:
            problems.append("health reports no cameras")
        elif not any(c.get("state") == "connected" for c in cams):
            problems.append(f"no camera ever connected: {[c.get('state') for c in cams]}")
        for reason in doc.get("reasons") or []:
            if "failed to load" in str(reason):
                problems.append(f"a configured detector could not load its "
                                f"model inside the bundle: {reason}")

    if frames_snapshot is None:
        problems.append("the frame publisher never announced itself while the engine ran")
    elif not frames_snapshot.get("token"):
        problems.append("frames.json carries no token — the UI could not authenticate")
    elif frames_path.exists():
        problems.append("frames.json outlived the engine — a stale marker sends the app's "
                        "tiles to a dead URL after Stop (#144)")

    db = tmp / "events.db"
    if db.exists():
        con = sqlite3.connect(str(db))
        try:
            n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            cols = {r[1] for r in con.execute("PRAGMA table_info(events)")}
        finally:
            con.close()
        missing = {"state", "unverified", "provisional"} - cols
        if missing:
            problems.append(f"events schema missing {sorted(missing)} — a migration did not run")
        print(f"  alerts persisted: {n}")
    else:
        problems.append("no events.db was created")

    if timed_out:
        problems.append("engine did not exit by itself within 900s (asked for --seconds 70)")
    elif proc.returncode != 0:
        problems.append(f"engine exited {proc.returncode}")
    if frames_snapshot is not None:
        print(f"  frame publisher announced on port {frames_snapshot.get('port')} during the run")

    if problems:
        print("\nFAIL — the shipped bundle did not work:")
        for p in problems:
            print("  -", p)
        print("\n--- engine output tail ---\n" + tail)
        return 1
    print("PASS — the shipped bundle decoded video, published authenticated frames, "
          "and persisted alerts.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
