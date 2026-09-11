"""fetch_virat.py — annotation-driven selective pull of VIRAT Ground 2.0.

    python tools/fetch_virat.py plan                 # which videos, which events, how many GB
    python tools/fetch_virat.py pull --budget-gb 8   # download the plan's videos (resumable)
    python tools/fetch_virat.py cut                  # cut labeled eval clips from downloads

The KPI manifest's rows 5-7 (person / intrusion / loitering) are SMOKE for lack
of footage. VIRAT is the canonical filler — real outdoor surveillance cameras
with per-event annotations — but the full set is 37.6 GB and this machine does
not have it. So: pull the 41 MB annotation bundle first, decide from the labels
which videos actually contain the events we need, and download only those.

Label mapping (VIRAT event types -> our KPI kinds):
  11 entering facility, 12 exiting facility  -> intrusion  (crossing a boundary)
  loitering is NOT a VIRAT event type        -> derived: a person track from
     the objects file that dwells >= LOITER_S seconds within a small radius
  person (row 5)                             -> any video's person tracks;
     normal-walking windows away from events become person_ positives

Hosting: Kitware Girder, anonymous HTTP, per-item downloads with Range support
(verified 9 Sep 2026). Videos land in data/staging/virat/videos/, cut clips in
data/eval_clips/virat/ with the label in the filename (the fetcher convention
collect_clips already understands).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "data" / "staging" / "virat"
VIDEOS = STAGING / "videos"
ANNOT = STAGING / "annotations"
OUT = ROOT / "data" / "eval_clips" / "virat"
API = "https://data.kitware.com/api/v1"

# VIRAT event types (docs/ in the annotation bundle)
EVENT_NAMES = {1: "person_loading_vehicle", 2: "person_unloading_vehicle",
               3: "person_opening_trunk", 4: "person_closing_trunk",
               5: "person_getting_into_vehicle", 6: "person_getting_out_of_vehicle",
               7: "person_gesturing", 8: "person_digging", 9: "person_carrying",
               10: "person_running", 11: "entering_facility", 12: "exiting_facility"}
INTRUSION_EVENTS = (11, 12)
LOITER_S = 60.0        # dwell threshold: a minute near one spot
LOITER_RADIUS = 60.0   # px — "one spot" at VIRAT's 1080/720p ground views
FPS_DEFAULT = 30.0     # VIRAT ground videos are 24-30fps; timing from docs uses frames
CLIP_PAD_S = 5.0       # seconds of context either side of an event window


def _events_files():
    # Girder folder zips wrap each item's file in a directory of the same
    # name — match only real files, wherever they nest.
    return sorted(p for p in ANNOT.rglob("*.viratdata.events.txt*")
                  if p.is_file())


def _objects_file(events_file: Path) -> Path:
    cand = events_file.with_name(
        events_file.name.replace(".events.txt", ".objects.txt"))
    if cand.is_file():
        return cand
    # the wrapped-directory layout: sibling dir named after the objects file
    d = events_file.parent.parent / events_file.parent.name.replace(
        ".events.txt", ".objects.txt")
    hits = [p for p in d.rglob("*")] if d.is_dir() else []
    return next((p for p in hits if p.is_file()), cand)


def _video_name(events_file: Path) -> str:
    stem = events_file.name.split(".viratdata")[0]
    if not stem.startswith("VIRAT"):    # wrapped layout: dir carries the name
        stem = events_file.parent.name.split(".viratdata")[0]
    return stem + ".mp4"


def _parse_events(path: Path) -> list[dict]:
    """events.txt columns: event_id type duration start_f end_f cur_f x y w h.

    One ROW PER FRAME of each event — dedupe by event_id or a 36s video
    "contains" 300 entering-facility events."""
    seen: dict[int, dict] = {}
    for ln in path.read_text().splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        eid = int(parts[0])
        if eid not in seen:
            seen[eid] = {"type": int(parts[1]),
                         "start_f": int(parts[3]), "end_f": int(parts[4])}
    return list(seen.values())


def _loiter_windows(objects_path: Path) -> list[tuple[int, int]]:
    """Person tracks (object type 1) that dwell within LOITER_RADIUS >= LOITER_S."""
    if not objects_path.exists():
        return []
    tracks: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for ln in objects_path.read_text().splitlines():
        p = ln.split()
        # object_id duration cur_f x y w h type  (type 1 = person)
        if len(p) < 8 or p[7] != "1":
            continue
        oid, f = int(p[0]), int(p[2])
        x, y, w, h = (float(v) for v in p[3:7])
        tracks[oid].append((f, x + w / 2, y + h / 2))
    windows = []
    need = int(LOITER_S * FPS_DEFAULT)
    for pts in tracks.values():
        pts.sort()
        i = 0
        for j in range(len(pts)):
            while (max(abs(pts[j][1] - pts[i][1]), abs(pts[j][2] - pts[i][2]))
                   > LOITER_RADIUS):
                i += 1
            if pts[j][0] - pts[i][0] >= need:
                windows.append((pts[i][0], pts[j][0]))
                break                      # one loiter window per track
    return windows


def _index() -> dict[str, dict]:
    items = json.loads((STAGING / "items.json").read_text())
    return {i["name"]: i for i in items}


def _person_window(objects_path: Path) -> list[tuple[int, int]]:
    """One clean person window per video: the first person track that lasts
    >= 10s. Fills KPI row 5 (person detection) — every VIRAT ground video is
    full of tracked people, but only labeled windows become clips."""
    if not objects_path.exists():
        return []
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for ln in objects_path.read_text().splitlines():
        parts = ln.split()
        if len(parts) < 8 or parts[7] != "1":
            continue
        oid, f = int(parts[0]), int(parts[2])
        first.setdefault(oid, f)
        last[oid] = f
    for oid in first:
        if last[oid] - first[oid] >= 300:          # >= 10s at 30fps
            return [(first[oid], min(first[oid] + 450, last[oid]))]
    return []


def build_plan() -> dict:
    """Per-video: which event windows exist, so `pull` can rank by yield/GB."""
    idx = _index()
    plan: dict[str, dict] = {}
    for ef in _events_files():
        name = _video_name(ef)
        if name not in idx:
            continue
        events = _parse_events(ef)
        intr = [(e["start_f"], e["end_f"]) for e in events
                if e["type"] in INTRUSION_EVENTS]
        loit = _loiter_windows(_objects_file(ef))
        person = _person_window(_objects_file(ef))
        if not intr and not loit and not person:
            continue
        plan[name] = {"item_id": idx[name]["_id"], "size": idx[name]["size"],
                      "intrusion": intr, "loitering": loit, "person": person}
    return plan


def cmd_plan() -> int:
    plan = build_plan()
    (STAGING / "plan.json").write_text(json.dumps(plan, indent=1))
    n_i = sum(len(v["intrusion"]) for v in plan.values())
    n_l = sum(len(v["loitering"]) for v in plan.values())
    gb = sum(v["size"] for v in plan.values()) / 1e9
    print(f"{len(plan)} videos carry events: {n_i} intrusion windows "
          f"(types 11/12), {n_l} derived loitering windows — {gb:.1f} GB if all pulled")
    ranked = sorted(plan.items(), key=lambda kv: -(
        (len(kv[1]["intrusion"]) + len(kv[1]["loitering"])) / max(kv[1]["size"], 1)))
    print(f"\n{'video':<28}{'MB':>6}{'intr':>6}{'loit':>6}")
    for name, v in ranked[:25]:
        print(f"{name:<28}{v['size']/1e6:>6.0f}{len(v['intrusion']):>6}"
              f"{len(v['loitering']):>6}")
    print("\nwritten data/staging/virat/plan.json — next: pull --budget-gb N")
    return 0


def cmd_pull(budget_gb: float) -> int:
    plan = json.loads((STAGING / "plan.json").read_text())
    VIDEOS.mkdir(parents=True, exist_ok=True)
    # Loitering-first (10 Sep): the yield/GB ranking filled intrusion fast
    # (146 cuttable events on disk vs 53 needed) because intrusion-dense
    # videos are small — but the dwell windows live in LONG recordings, and
    # loitering is the row still starving (73 floor, ~21 in hand). Weight
    # loitering windows heavily so the remaining budget buys the scarce row.
    ranked = sorted(plan.items(), key=lambda kv: -(
        (len(kv[1]["intrusion"]) + 40 * len(kv[1]["loitering"]))
        / max(kv[1]["size"], 1)))
    spent = sum((VIDEOS / n).stat().st_size for n, _ in ranked
                if (VIDEOS / n).exists())
    for name, v in ranked:
        dest = VIDEOS / name
        if dest.exists() and dest.stat().st_size >= v["size"]:
            continue
        if spent + v["size"] > budget_gb * 1e9:
            continue
        print(f"pulling {name} ({v['size']/1e6:.0f} MB)…", flush=True)
        url = f"{API}/item/{v['item_id']}/download"
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        # A short read is NOT a download: the server closing early does not
        # raise, and a truncated mp4 (index atom at the end) probes as 0s and
        # silently starves the cut. 15 files shipped that way on 10 Sep.
        if tmp.stat().st_size < v["size"]:
            print(f"  short read ({tmp.stat().st_size}/{v['size']}) — kept as .part")
            continue
        tmp.rename(dest)
        spent += v["size"]
    have = sorted(p.name for p in VIDEOS.glob("*.mp4"))
    print(f"\non disk: {len(have)} videos, "
          f"{sum((VIDEOS / n).stat().st_size for n in have)/1e9:.1f} GB")
    return 0


def cmd_cut() -> int:
    plan = json.loads((STAGING / "plan.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    made = 0
    for name, v in plan.items():
        src = VIDEOS / name
        if not src.exists():
            continue
        stem = name.rsplit(".", 1)[0]
        for kind, windows in (("intrusion", v["intrusion"]),
                              ("loitering", v["loitering"]),
                              ("person", v.get("person", []))):
            # Diversity beats volume — but the caps differ by scarcity:
            # intrusion has events to burn (3/video), loitering has only 68
            # windows in ALL of VIRAT against a 73 floor, so it keeps 6.
            cap = 6 if kind == "loitering" else 3
            for k, (f0, f1) in enumerate(windows[:cap]):
                t0 = max(0.0, f0 / FPS_DEFAULT - CLIP_PAD_S)
                dur = (f1 - f0) / FPS_DEFAULT + 2 * CLIP_PAD_S
                dest = OUT / f"{kind}_{stem}_{k:02d}.mp4"
                if dest.exists():
                    continue
                cmd = ["ffmpeg", "-nostdin", "-loglevel", "error",
                       "-ss", f"{t0:.1f}", "-i", str(src), "-t", f"{dur:.1f}",
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                       "-an", str(dest)]
                if subprocess.run(cmd).returncode == 0:
                    made += 1
    print(f"cut {made} new clips into {OUT.relative_to(ROOT)} "
          f"(filenames carry the kind — the manifest's collector reads them)")
    print("Next: python tools/kpi_manifest.py status, then build to refreeze "
          "DELIBERATELY.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("plan", "pull", "cut"))
    ap.add_argument("--budget-gb", type=float, default=8.0)
    args = ap.parse_args()
    if args.command == "plan":
        return cmd_plan()
    if args.command == "pull":
        return cmd_pull(args.budget_gb)
    return cmd_cut()


if __name__ == "__main__":
    raise SystemExit(main())
