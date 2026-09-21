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

Before any of that, every clip must look like it came from a SURVEILLANCE
CAMERA, because the fetcher's YouTube queries return news coverage ABOUT
incidents at least as often as footage OF them, and a news anchor is a person
as far as a people-count is concerned. Two measurements separate them:

  cuts   -- hard scene changes. CCTV is one continuous shot; a news package
            cuts every few seconds; a vendor demo opens on a title card.
  motion -- global camera movement between frames. CCTV is bolted to a wall;
            a police body-cam and a panning news camera are not.

Calibrated 21 Sep on clips already judged by eye: five known fixed cameras
all measured 0 cuts and a median motion of 0.06-0.34 px; a news studio
(2 cuts), a picture-in-picture montage (3), a title-card demo (3), a body-cam
(1.0-1.4 px) and a panning advert (0.9-1.7 px) all fell outside. Thresholds
sit in the gap. The median is deliberate: the 90th percentile read 9.6 px on
a bolted-down camera watching a fire, because smoke moves.

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
# Fixed-camera gate. A single cut is flagged, not rejected: a barrier arm or a
# headlight sweep can produce one histogram spike on a real camera.
CUT_DISTANCE = 0.45          # Bhattacharyya distance between consecutive frames
REJECT_CUTS = 2
# MEDIAN global motion, not the 90th percentile: a fixed camera watching a
# fire reads a p90 of 9.6 px at 2 fps because the SMOKE moves, and phase
# correlation cannot tell content from camera in the tail. The median ignores
# a few frames of billowing. Fixed cameras measured 0.06-0.34 px; a body-cam,
# a panning advert and a montage measured 0.68-1.74. Threshold in the gap.
MAX_MOTION_MEDIAN_PX = 0.5


def camera_stability(frames) -> tuple[int, float]:
    """(hard cuts, MEDIAN global motion in px) over sampled frames."""
    import cv2
    import numpy as np
    prev_hist = prev_gray = None
    cuts = 0
    motions = []
    window = cv2.createHanningWindow((160, 90), cv2.CV_32F)
    for frame in frames:
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            if cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > CUT_DISTANCE:
                cuts += 1
            (dx, dy), _ = cv2.phaseCorrelate(prev_gray, gray, window)
            motions.append(float(np.hypot(dx, dy)))
        prev_hist, prev_gray = hist, gray
    median = float(np.median(motions)) if motions else 0.0
    return cuts, median


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

    frames = list(_sample(path))
    if not frames:
        row.update(verdict="REJECT", why="no decodable frames")
        return row

    cuts, motion = camera_stability(frames)
    row.update(cuts=cuts, motion_median_px=round(motion, 2))
    if cuts >= REJECT_CUTS:
        row.update(verdict="REJECT",
                   why=f"{cuts} hard cuts -- edited footage, not a surveillance camera")
        return row
    if motion > MAX_MOTION_MEDIAN_PX:
        row.update(verdict="REJECT",
                   why=f"camera moves {motion:.1f}px between frames -- handheld or "
                       f"body-worn, not a fixed camera")
        return row
    unsteady = f" (1 cut -- check it is not a title card)" if cuts == 1 else ""

    if cls in REVIEW_CLASSES:
        row.update(verdict="REVIEW",
                   why="no trustworthy automatic check for this class -- watch it" + unsteady)
        return row

    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    counts, aspects = [], []
    # frames were sampled once above and are reused here
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
                        f"(w/h peaks at {widest:.2f}){unsteady}" if ok else
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
               why=f"peak {peak} people (need >= {need}){unsteady}")
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
