"""download_training_data.py — Bulk download violence/normal training clips from YouTube.

Downloads ~50-70 clips per class using yt-dlp search queries.
Saves directly into the training folder structure expected by train_classifier.py.

Usage:
    python3 download_training_data.py                        # download all
    python3 download_training_data.py --class violence       # only violence clips
    python3 download_training_data.py --class normal         # only normal clips
    python3 download_training_data.py --limit 20             # limit per query
    python3 download_training_data.py --dry-run              # show what would be downloaded
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from cvti.logging_setup import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Output dirs (matches train_classifier.py structure)
# ---------------------------------------------------------------------------

TRAIN_THEFT    = Path("data/classify/train/theft")
TRAIN_VIOLENCE = Path("data/classify/train/violence")
TRAIN_NORMAL   = Path("data/classify/train/normal")

# ---------------------------------------------------------------------------
# Search queries
# ---------------------------------------------------------------------------

THEFT_QUERIES = [
    "CCTV robbery caught on camera",
    "store robbery surveillance footage",
    "shopping mall robbery security camera",
    "convenience store robbery CCTV",
    "bank robbery security camera footage",
    "petrol station robbery CCTV footage",
    "supermarket robbery caught on camera",
    "gas station armed robbery CCTV",
    "shoplifting caught on camera CCTV",
    "smash and grab robbery CCTV",
]

VIOLENCE_QUERIES = [
    "street fight caught on CCTV security camera",
    "security camera fight footage",
    "parking lot assault CCTV",
    "bar fight caught on camera CCTV",
    "school fight security camera footage",
    "physical assault caught on CCTV",
    "brawl caught on surveillance camera",
    "street mugging caught on camera CCTV",
]

NORMAL_QUERIES = [
    "shopping mall security camera normal footage",
    "convenience store CCTV normal day",
    "office lobby security camera footage",
    "street corner normal CCTV footage",
    "supermarket normal customer footage",
    "parking lot normal security camera",
    "bank lobby normal security camera",
    "public transport station CCTV normal",
]

# Max clips to download per search query
CLIPS_PER_QUERY = 6
# Max clip duration in seconds (skip very long videos)
MAX_DURATION = 120
# Min clip duration in seconds
MIN_DURATION = 5


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def run_yt_dlp(query: str, output_dir: Path, limit: int, dry_run: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        f"ytsearch{limit}:{query}",
        "--extractor-args", "youtube:player_client=android",
        "-f", "best[ext=mp4][height<=480]/best[ext=mp4]/best",
        "--match-filter", f"duration>{MIN_DURATION} & duration<{MAX_DURATION}",
        "-o", str(output_dir / "%(title).40s_%(id)s.%(ext)s"),
        "--no-playlist",
        "--ignore-errors",
        "--no-warnings",
        "--quiet",
        "--progress",
    ]

    if dry_run:
        cmd += ["--simulate", "--print", "%(title)s (%(duration)ss)"]

    log.info(f"\n  Query: \"{query}\"")
    log.info(f"  → {output_dir}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def count_existing(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.mp4")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download violence/normal training clips from YouTube.")
    parser.add_argument(
        "--class",
        dest="cls",
        choices=("theft", "violence", "normal", "all"),
        default="all",
        help="Which class to download. Default: all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=CLIPS_PER_QUERY,
        help=f"Max clips to download per search query. Default: {CLIPS_PER_QUERY}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading.",
    )
    return parser.parse_args()


def main() -> None:
    # Entrypoint: without this, log records have no handler and
    # anything below WARNING is silently discarded.
    from cvti.logging_setup import setup_logging
    setup_logging(component="argus-download")
    args = parse_args()

    log.info("=== CV Threat Intelligence — Training Data Downloader ===")
    log.info(f"Mode     : {'DRY RUN' if args.dry_run else 'DOWNLOAD'}")
    log.info(f"Limit    : {args.limit} clips per query")
    log.info(f"Class    : {args.cls}")
    log.info(f"Theft    → {TRAIN_THEFT}")
    log.info(f"Violence → {TRAIN_VIOLENCE}")
    log.info(f"Normal   → {TRAIN_NORMAL}")

    # Check yt-dlp is available
    check = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
    if check.returncode != 0:
        log.warning("\nError: yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)
    log.info(f"yt-dlp   : {check.stdout.strip()}")

    # Download theft clips
    if args.cls in ("theft", "all"):
        log.info(f"\n--- Downloading THEFT clips ({len(THEFT_QUERIES)} queries × {args.limit} each) ---")
        log.info(f"Existing: {count_existing(TRAIN_THEFT)} clips")
        for query in THEFT_QUERIES:
            run_yt_dlp(query, TRAIN_THEFT, args.limit, args.dry_run)
            if not args.dry_run:
                time.sleep(2)
        log.info(f"\nTheft total: {count_existing(TRAIN_THEFT)} clips")

    # Download violence clips
    if args.cls in ("violence", "all"):
        log.info(f"\n--- Downloading VIOLENCE clips ({len(VIOLENCE_QUERIES)} queries × {args.limit} each) ---")
        log.info(f"Existing: {count_existing(TRAIN_VIOLENCE)} clips")
        for query in VIOLENCE_QUERIES:
            run_yt_dlp(query, TRAIN_VIOLENCE, args.limit, args.dry_run)
            if not args.dry_run:
                time.sleep(2)
        log.info(f"\nViolence total: {count_existing(TRAIN_VIOLENCE)} clips")

    # Download normal clips
    if args.cls in ("normal", "all"):
        log.info(f"\n--- Downloading NORMAL clips ({len(NORMAL_QUERIES)} queries × {args.limit} each) ---")
        log.info(f"Existing: {count_existing(TRAIN_NORMAL)} clips")
        for query in NORMAL_QUERIES:
            run_yt_dlp(query, TRAIN_NORMAL, args.limit, args.dry_run)
            if not args.dry_run:
                time.sleep(2)
        log.info(f"\nNormal total: {count_existing(TRAIN_NORMAL)} clips")

    log.info("\n=== Download complete ===")
    log.info(f"Theft clips    : {count_existing(TRAIN_THEFT)}")
    log.info(f"Violence clips : {count_existing(TRAIN_VIOLENCE)}")
    log.info(f"Normal clips   : {count_existing(TRAIN_NORMAL)}")
    if not args.dry_run:
        log.info("\nNext step — extract frames and train:")
        log.info("  python3 train_classifier.py")


if __name__ == "__main__":
    main()
