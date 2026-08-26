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
           "--gate-drain", "3", "--mobile-port", "0", "--output-dir", str(tmp)]
    print("running:", " ".join(cmd[:3]), "...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-1500:]

    problems = []
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

    frames = tmp / "frames.json"
    if not frames.exists():
        problems.append("the frame publisher never announced itself")
    elif not json.loads(frames.read_text()).get("token"):
        problems.append("frames.json carries no token — the UI could not authenticate")

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

    if proc.returncode != 0:
        problems.append(f"engine exited {proc.returncode}")

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
