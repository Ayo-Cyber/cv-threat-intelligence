"""Fetch short public clips to EVALUATE the rule-based detectors against.

The HSE detectors (fire/smoke, panic-running, crowd) are hand-written rules with
hand-picked thresholds — nothing is trained, so what they need isn't training
data, it's clips to MEASURE against. Positives are the scarce part: the repo
already has plenty of ordinary footage to serve as negatives.

    python tools/fetch_eval_clips.py --class fire --count 15
    python tools/fetch_eval_clips.py --class crowd --count 15 --seconds 20

Files land in data/test_clips/ named so the eval harness picks up the label from
the filename (fire_01.mp4 -> a fire positive). Clips are trimmed to keep them
small and are for internal evaluation only — not redistribution.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "test_clips"

# Curated queries beat one generic search: each angle surfaces different footage,
# and CCTV-style framing is what the detectors will actually see in production.
QUERIES = {
    # Aim at raw footage: generic "fire" searches return news anchors, brand
    # logos and product demos, none of which contain a flame.
    "fire": [
        "house fire spreading inside room footage no commentary",
        "flames engulf room raw footage",
        "fire burning building interior real footage",
        "car fire burning footage",
        "bonfire burning close up footage",
        "industrial fire flames raw footage",
    ],
    "crowd": [
        "crowd panic running cctv",
        "stampede surveillance footage",
        "crowd rushing security camera",
        "people running panic street camera",
    ],
    "fall": [
        "person collapses cctv",
        "man faints security camera",
        "slip and fall surveillance footage",
    ],
}


def fetch(cls: str, count: int, seconds: int, start: int) -> int:
    queries = QUERIES.get(cls)
    if not queries:
        print(f"no queries for '{cls}' (known: {', '.join(QUERIES)})")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUT.glob(f"{cls}_*.mp4")))
    got = 0
    per_query = max(1, count // len(queries) + 1)
    for q in queries:
        if got >= count:
            break
        idx = existing + got + 1
        cmd = [
            "yt-dlp",
            "--extractor-args", "youtube:player_client=android",
            "--match-filters", "duration < 600 & !is_live",
            "--download-sections", f"*{start}-{start + seconds}",
            "--force-keyframes-at-cuts",
            "-f", "best[height<=720]/best",
            "--recode-video", "mp4",
            "--no-playlist", "--no-warnings", "--ignore-errors",
            "-o", str(OUT / f"{cls}_%(autonumber)02d.mp4"),
            "--autonumber-start", str(idx),
            f"ytsearch{per_query}:{q}",
        ]
        print(f"  [{cls}] {q!r} ...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        now = len(list(OUT.glob(f"{cls}_*.mp4")))
        new = now - existing - got
        got += max(0, new)
        print(f"        +{max(0, new)} (total {got})", flush=True)
    print(f"[fetch] {cls}: {got} clip(s) -> {OUT}")
    print("[fetch] EYEBALL THEM before trusting the labels — a search result is a "
          "guess, and a mislabelled clip poisons the measurement.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--class", dest="cls", required=True, choices=sorted(QUERIES))
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--seconds", type=int, default=20, help="clip length to keep")
    p.add_argument("--start", type=int, default=10, help="skip intros")
    a = p.parse_args()
    return fetch(a.cls, a.count, a.seconds, a.start)


if __name__ == "__main__":
    sys.exit(main())
