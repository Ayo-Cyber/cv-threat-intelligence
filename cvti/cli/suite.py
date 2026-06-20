"""run_suite.py — single command to exercise the full threat-intelligence stack.

Runs in order:
  1. Unit tests (pytest)
  2. Retail pipeline  — theft clips  (zones + concealment + rules + VLM gate)
  3. General detector — weapon clip  (weapon detector + classifier)
  4. General detector — violence clip (pose heuristics + classifier)
  5. General detector — normal clip   (FPR check — should stay quiet)

All video runs are headless (no window); annotated .mp4s and alert JSONs are
written to  runs/suite/<scenario>/  so you can review afterwards.

Usage:
    python run_suite.py
    python run_suite.py --skip-tests          # skip pytest, go straight to video
    python run_suite.py --gate-provider anthropic   # use real VLM for retail gate
    python run_suite.py --show                # pop a window for every scenario
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CLASSIFIER = "runs/classify/runs/classify/robbery_v19/weights/best.pt"
SUITE_DIR  = Path("runs/suite")

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

RETAIL_SCENARIOS = [
    {
        "name":   "theft_shop_01 — shoplifting (zones ON)",
        "source": "data/test_clips/theft_shop_01.mp4",
        "config": "configs/retail_pipeline_v1.json",
        "zones":  "configs/retail_zones.theft_shop_01.json",
        "expect": "ALERT",
    },
    {
        "name":   "theft_shop_01 — shoplifting (zones ON)",
        "source": "data/test_clips/theft_shop_01.mp4",
        "config": "configs/retail_pipeline_v1.json",
        "zones":  "configs/retail_zones.theft_shop_01.json",
        "expect": "ALERT",
    },
    {
        "name":   "theft_shop_02 — shoplifting (zones ON)",
        "source": "data/test_clips/theft_shop_02.mp4",
        "config": "configs/retail_pipeline_v1.json",
        "zones":  "configs/retail_zones.theft_shop_01.json",
        "expect": "ALERT",
    },
    {
        "name":   "empty_warehouse — FPR check (no people, should stay quiet)",
        "source": "data/test_clips/empty_warehouse.mp4",
        "config": "configs/retail_pipeline_v1.json",
        "zones":  "",
        "expect": "QUIET",
    },
    {
        "name":   "normal_street_01 — FPR check (should stay quiet)",
        "source": "data/test_clips/normal_street_01.mp4",
        "config": "configs/retail_pipeline_v1.json",
        "zones":  "",
        "expect": "QUIET",
    },
]

DETECTOR_SCENARIOS = [
    {
        "name":   "weapons_yt_01 — weapon detection",
        "source": "data/test_clips/weapons_yt_01.mp4",
        "expect": "ALERT",
    },
    {
        "name":   "violence_suspected — violence heuristics",
        "source": "data/test_clips/violence_suspected.mp4",
        "expect": "ALERT",
    },
    {
        "name":   "stabbing_event — high-severity threat",
        "source": "data/test_clips/stabbing_event.mp4",
        "expect": "ALERT",
    },
    {
        "name":   "theft_yt_01 — theft detection",
        "source": "data/test_clips/theft_yt_01.mp4",
        "expect": "ALERT",
    },
    {
        "name":   "normal_street_02 — FPR check (should stay quiet)",
        "source": "data/test_clips/normal_street_02.mp4",
        "expect": "QUIET",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP  = "=" * 70
SEP2 = "-" * 70

def _print(msg: str) -> None:
    print(msg, flush=True)

def _header(title: str) -> None:
    _print(f"\n{SEP}\n  {title}\n{SEP}")

def _run(cmd: list[str], label: str) -> tuple[int, str]:
    _print(f"\n{SEP2}")
    _print(f"  Running: {label}")
    _print(f"  CMD: {' '.join(cmd)}")
    _print(SEP2)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    _print(f"  [{elapsed:.1f}s] exit={proc.returncode}")
    return proc.returncode, ""

def _count_alerts(out_dir: Path) -> int:
    return len(list(out_dir.glob("event_*/alert.json"))) + \
           len(list(out_dir.glob("event_*/metadata.json")))

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def run_unit_tests() -> bool:
    _header("STEP 1 — Unit Tests (pytest)")
    code, _ = _run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
        "pytest tests/"
    )
    passed = code == 0
    _print(f"\n  Unit tests: {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


def run_retail(args: argparse.Namespace) -> list[dict]:
    _header("STEP 2 — Retail Pipeline (zones + concealment + rules + gate)")
    results = []
    for s in RETAIL_SCENARIOS:
        out_dir = SUITE_DIR / "retail" / s["name"].split("—")[0].strip().replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(out_dir / "annotated.mp4")

        cmd = [
            "cvti-retail",
            "--source", s["source"],
            "--config", s["config"],
            "--gate-provider", args.gate_provider,
            "--output-dir", str(out_dir),
            "--save", save_path,
        ]
        if s["zones"]:
            cmd += ["--zones", s["zones"]]
        cmd += ["--show"]

        t0 = time.time()
        _print(f"\n{SEP2}")
        _print(f"  Scenario: {s['name']}")
        _print(f"  Expect  : {s['expect']}")
        _print(SEP2)
        proc = subprocess.run(cmd, text=True)
        elapsed = time.time() - t0

        alerts = _count_alerts(out_dir)
        status = "ALERT" if alerts > 0 else "QUIET"
        ok = status == s["expect"]
        results.append({"name": s["name"], "expect": s["expect"],
                        "got": status, "alerts": alerts, "ok": ok, "time": elapsed})
        _print(f"  → {alerts} alert(s) in {elapsed:.1f}s  {'✓' if ok else '✗ (unexpected)'}")
    return results


def run_detector(args: argparse.Namespace) -> list[dict]:
    _header("STEP 3 — General Detector (classifier + weapon + pose)")
    results = []
    classifier = CLASSIFIER if Path(CLASSIFIER).exists() else ""
    if not classifier:
        _print(f"  WARNING: classifier weights not found at {CLASSIFIER} — running without classifier")

    for s in DETECTOR_SCENARIOS:
        out_dir = SUITE_DIR / "detector" / s["name"].split("—")[0].strip().replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        save_video = str(out_dir / "annotated.mp4")
        cmd = [
            "cvti-detect",
            "--source", s["source"],
            "--save-dir", str(out_dir),
            "--save-video", save_video,
        ]
        if classifier:
            cmd += ["--classifier-weights", classifier]
        cmd += ["--show"]

        t0 = time.time()
        _print(f"\n{SEP2}")
        _print(f"  Scenario: {s['name']}")
        _print(f"  Expect  : {s['expect']}")
        _print(SEP2)
        proc = subprocess.run(cmd, text=True)
        elapsed = time.time() - t0

        alerts = _count_alerts(out_dir)
        status = "ALERT" if alerts > 0 else "QUIET"
        ok = status == s["expect"]
        results.append({"name": s["name"], "expect": s["expect"],
                        "got": status, "alerts": alerts, "ok": ok, "time": elapsed})
        _print(f"  → {alerts} alert(s) in {elapsed:.1f}s  {'✓' if ok else '✗ (unexpected)'}")
    return results

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(tests_ok: bool, retail: list[dict], detector: list[dict]) -> None:
    _header("SUITE SUMMARY")

    _print(f"\n  Unit Tests : {'PASSED ✓' if tests_ok else 'FAILED ✗'}\n")

    _print("  Retail Pipeline:")
    for r in retail:
        tick = "✓" if r["ok"] else "✗"
        _print(f"    {tick}  {r['name'][:52]:<52}  expect={r['expect']}  got={r['got']}  alerts={r['alerts']}")

    _print("\n  General Detector:")
    for r in detector:
        tick = "✓" if r["ok"] else "✗"
        _print(f"    {tick}  {r['name'][:52]:<52}  expect={r['expect']}  got={r['got']}  alerts={r['alerts']}")

    all_results = retail + detector
    passed = sum(1 for r in all_results if r["ok"])
    total  = len(all_results)
    total_time = sum(r["time"] for r in all_results)

    _print(f"\n  Scenarios: {passed}/{total} matched expectations")
    _print(f"  Total run time: {total_time:.0f}s")
    _print(f"\n  Evidence saved to: {SUITE_DIR.resolve()}")
    _print(SEP)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full threat-intelligence test suite.")
    p.add_argument("--skip-tests",    action="store_true", help="Skip pytest unit tests.")
    p.add_argument("--skip-retail",   action="store_true", help="Skip retail pipeline scenarios.")
    p.add_argument("--skip-detector", action="store_true", help="Skip general detector scenarios.")
    p.add_argument("--gate-provider", default="mock", choices=("mock", "anthropic"),
                   help="VLM provider for the retail verification gate (default: mock).")
    p.add_argument("--show", action="store_true", help="Show live video window for each scenario.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    SUITE_DIR.mkdir(parents=True, exist_ok=True)

    _print(f"\n{'#' * 70}")
    _print(f"  CV THREAT INTELLIGENCE — FULL SUITE")
    _print(f"  gate={args.gate_provider}  show={args.show}")
    _print(f"{'#' * 70}")

    tests_ok  = run_unit_tests()  if not args.skip_tests    else True
    retail    = run_retail(args)  if not args.skip_retail   else []
    detector  = run_detector(args) if not args.skip_detector else []

    print_summary(tests_ok, retail, detector)


if __name__ == "__main__":
    main()
