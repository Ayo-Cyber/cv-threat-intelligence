"""motion_gate_calibrate.py — pick the detector's motion threshold from footage.

    python tools/motion_gate_calibrate.py

Runs cvti/serving/motion_gate.MotionGate over the KPI manifest's clips at the
engine's own detection rate and reports, per threshold, how many detector runs
it would skip. Two populations matter and they pull in opposite directions:

  NORMALS  — empty/quiet CCTV. Skipping here is the entire point: it is the
             fixed compute floor we are trying to remove.
  ACTIVE   — clips with a real event (theft, person, intrusion). Skipping here
             is the risk. Measured with tracked=0 on EVERY frame, i.e. as if
             the camera never held a track, which is the worst case the gate
             can ever face; in the engine a live track disables it entirely.

The threshold to ship is the one that skips a lot of empty footage while
leaving active footage essentially untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGET_FPS = 4.0
MAX_SECONDS = 30.0


def frames(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src / TARGET_FPS)))
    i, taken = 0, 0
    while taken < int(MAX_SECONDS * TARGET_FPS):
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        taken += 1
        yield fr
    cap.release()


def measure(clips: list[str], threshold: float) -> tuple[int, int]:
    from cvti.serving.motion_gate import MotionGate
    gate = MotionGate(min_changed_fraction=threshold)
    for rel in clips:
        p = ROOT / rel
        if not p.exists():
            continue
        t = 0.0
        for fr in frames(p):
            gate.should_detect(rel, fr, tracked=0, now=t)
            t += 1.0 / TARGET_FPS
    return gate.ran, gate.skipped


def main() -> int:
    man = json.loads((ROOT / "configs/eval/kpi_manifest_v1.json").read_text())["rows"]
    normals = [c["path"] for c in man["false_positive"]["clips"]][:40]
    active = ([c["path"] for c in man["theft"]["clips"]][:20]
              + [c["path"] for c in man["person"]["clips"]][:20])
    print(f"normals: {len(normals)} clips   active: {len(active)} clips "
          f"(sampled; {TARGET_FPS:.0f} fps, {MAX_SECONDS:.0f}s each)\n")
    print(f"{'moved%':>10} {'skipped on NORMALS':>22} {'skipped on ACTIVE':>21}")
    for t in (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        nr, ns = measure(normals, t)
        ar, asx = measure(active, t)
        nf = ns / (nr + ns) if (nr + ns) else 0.0
        af = asx / (ar + asx) if (ar + asx) else 0.0
        print(f"{t*100:>9.1f}% {ns:>8}/{nr+ns:<7} = {nf*100:>4.1f}% "
              f"{asx:>7}/{ar+asx:<6} = {af*100:>4.1f}%")
    print("\nShip the largest NORMALS saving whose ACTIVE cost stays near zero.")
    print("In the engine the gate is also disabled whenever a track is live, so")
    print("the ACTIVE column above is a worst case, not the expected cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
