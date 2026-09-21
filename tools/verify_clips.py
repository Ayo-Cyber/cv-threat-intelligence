#!/usr/bin/env python3
"""Check that a clip contains what its filename claims.

tools/fetch_eval_clips.py downloads YouTube search results and prints "EYEBALL
THEM before trusting the labels". Nobody did, so data/test_clips ended up with
a WSJ segment about BMW cars named fall_03.mp4, a shoplifting clip named
weapons_yt_01.mp4, and two 0.23-second fragments inside the KPI manifest's
`suspicious` row (found 20 Sep). Every score computed over that set inherited
the error.

Verification differs by what the class actually is:

  COUNT classes (crowd, person)   -- how many people the detector finds. Person
                                     detection is the one thing here that is
                                     reliably good, so this is the strongest
                                     automatic check.
  GEOMETRY classes (fall)         -- a fallen person's box is wider than tall,
                                     SUSTAINED, the same test
                                     cvti/detector/fall.py makes.
  ACTION classes (violence, theft)-- an action has no object to point at, and
                                     using our own detector to label our own
                                     test set is circular.
  AMORPHOUS classes (fire, smoke) -- open-vocabulary detection cannot see them.
                                     Asked for "fire/flames/smoke" over a bus
                                     fully ablaze, YOLO-World answered 0/59
                                     frames. An earlier draft of this tool used
                                     it anyway and would have thrown the only
                                     genuine fire clip out of the set.
  WEAPON classes                  -- the weapon model calls cardboard boxes guns
                                     at 0.60, so it cannot police its own data.

The last three are reported REVIEW: the tool has no trustworthy opinion and a
person must watch them. That is a smaller claim than a verdict, and an honest
one -- a verifier that guesses is how the mislabels got in.

    python tools/verify_clips.py --class fall
    python tools/verify_clips.py --glob 'data/test_clips/weapons_*.mp4'
"""
from __future__ import annotations

import argparse
import glob as globlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# phrase list, and how many sampled frames must contain it to count
COUNT_CLASSES: dict[str, int] = {"crowd": 5, "person": 1, "intrusion": 1,
                                 "loitering": 1}
GEOMETRY_CLASSES = {"fall"}
# Nothing here can be judged automatically without lying about the confidence.
REVIEW_CLASSES = {"violence", "theft", "suspicious", "concealment",
                  "fire", "smoke", "weapons", "attr",
                  # A negative clip means NO THREAT, not no people. Counting
                  # heads rejected 33 perfectly good normal_kip clips for the
                  # crime of showing three colleagues in an office.
                  "normal", "empty"}

MIN_SECONDS = 2.0


def _class_of(path: Path) -> str:
    stem = path.stem
    for known in (list(COUNT_CLASSES) + sorted(GEOMETRY_CLASSES)
                  + sorted(REVIEW_CLASSES)):
        if stem.startswith(known):
            return known
    # An unknown class gets REVIEW, never a guess. synthetic_fire.mp4 fell
    # through to the person-count rule and was rejected for containing no
    # people, which is exactly what a synthetic fire clip should contain.
    return stem.split("_")[0]


def _duration(path: Path) -> float:
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except (ValueError, OSError, subprocess.SubprocessError):
        return 0.0


def _sample(path: Path, per_second: float = 2.0):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / per_second)))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield frame
        i += 1
    cap.release()


def verify(path: Path) -> dict:
    cls = _class_of(path)
    seconds = _duration(path)
    row = {"clip": path.name, "class": cls, "seconds": round(seconds, 2)}
    if seconds < MIN_SECONDS:
        row.update(verdict="REJECT", why=f"only {seconds:.2f}s of footage")
        return row

    if cls in REVIEW_CLASSES:
        row.update(verdict="REVIEW",
                   why="no trustworthy automatic check for this class -- watch it")
        return row

    frames = list(_sample(path))
    if not frames:
        row.update(verdict="REJECT", why="no decodable frames")
        return row

    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    counts, aspects = [], []
    # A fall is a SUSTAINED horizontal posture, so the verifier applies the same
    # test cvti/detector/fall.py does. Peak alone passed a WSJ segment about BMW
    # cars as fall_03.mp4: a presenter gesturing hits w/h 1.39 for one frame.
    run = best_run = 0
    for frame in frames:
        res = model.predict(frame, conf=0.25, verbose=False, device="cpu")[0]
        area = frame.shape[0] * frame.shape[1]
        people = 0
        widest_here = 0.0
        for box, cid in zip(res.boxes.xyxy, res.boxes.cls):
            if int(cid) != 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
            if (w * h) / area < 0.004:
                continue
            people += 1
            aspects.append(w / h)
            widest_here = max(widest_here, w / h)
        counts.append(people)
        run = run + 1 if widest_here >= 1.15 else 0
        best_run = max(best_run, run)

    if cls in GEOMETRY_CLASSES:
        widest = max(aspects) if aspects else 0.0
        need_run = 3
        row.update(widest_person_box=round(widest, 2), longest_run=best_run)
        ok = best_run >= need_run
        row.update(verdict="OK" if ok else "REJECT",
                   why=(f"a person stays horizontal for {best_run} sampled frames "
                        f"(w/h peaks at {widest:.2f})" if ok else
                        f"no sustained horizontal posture -- longest run {best_run} "
                        f"frame(s), need {need_run} (w/h peaks at {widest:.2f})"))
        return row

    need = COUNT_CLASSES.get(cls)
    if need is None:
        row.update(verdict="REVIEW",
                   why=f"no check defined for class {cls!r} -- watch it")
        return row
    peak = max(counts) if counts else 0
    row.update(peak_people=peak)
    row.update(verdict="OK" if peak >= need else "REJECT",
               why=f"peak {peak} people (need >= {need})")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", help="verify data/test_clips/<class>_*.mp4")
    ap.add_argument("--glob", help="explicit glob instead of --class")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.glob:
        paths = [Path(p) for p in sorted(globlib.glob(a.glob))]
    elif a.cls:
        paths = sorted((ROOT / "data" / "test_clips").glob(f"{a.cls}_*.mp4"))
    else:
        ap.error("pass --class or --glob")
    if a.limit:
        paths = paths[:a.limit]
    rows = []
    for path in paths:
        row = verify(path)
        rows.append(row)
        print(f"  {row['verdict']:<13} {row['clip']:<26} {row['why']}", flush=True)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"\n[verify] {len(rows)} clip(s): " +
          ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"[verify] report -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
