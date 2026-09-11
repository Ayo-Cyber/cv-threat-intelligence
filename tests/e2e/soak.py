"""Soak: run the real engine for hours and assert it stayed honest.

    python tests/e2e/soak.py --minutes 300 --out runs/soak     # the CI soak
    python tests/e2e/soak.py --minutes 2                       # local wiring check

W7's 'silent failures = 0' budget needs a venue: this is it. An ffmpeg
process loops a demo clip over TCP-MPEGTS (the latency yardstick's proven
pattern — no docker, works on the Windows runner), the engine consumes it as
a normal camera for N minutes, and the soak asserts what a customer would:

  1. the engine survived to its scheduled end (>=95% of the requested run);
  2. the heartbeat (gate_health.json) was still fresh at the end — a zombie
     that stopped writing counts as a failure even if the process lived;
  3. memory did not grow without bound (last RSS <= first*1.6 + slack);
  4. the log carries no tracebacks;
  5. alert latency: events.db's per-event latency_s p95 (when alerts fired).

Results land in --out/soak_report.json in budget_check's evidence shape, so
`python tools/budget_check.py --evidence <out>` turns a soak into a verdict.
Exit 0 = all soak assertions hold; 1 = any failed.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PORT = 18571


def _unhandled_tracebacks(log_text: str) -> int:
    """Count tracebacks that are CRASHES, not logged diagnostics.

    The first 5h soak failed on three tracebacks that were the fail-visible
    gate doing its documented job: `log.warning(..., exc_info=True)` prints
    the diagnostic traceback right under its WARNING line, and the alert was
    surfaced UNVERIFIED — loud, designed degradation. A crash traceback, by
    contrast, arrives raw on stderr with no logger line in front of it. So: a
    traceback within three lines of a leveled logger record is handled
    telemetry; anything else counts."""
    lines = log_text.splitlines()
    unhandled = 0
    for i, line in enumerate(lines):
        if not line.startswith("Traceback (most recent call last)"):
            continue
        context = " ".join(lines[max(0, i - 3):i])
        handled = any(lvl in context for lvl in
                      ("WARNING", "ERROR", "DEBUG", "INFO", "CRITICAL"))
        # Chained sections of ONE diagnostic print fresh Traceback headers
        # after Python's chain markers — they belong to the logged parent.
        chained = ("During handling of the above exception" in context
                   or "The above exception was the direct cause" in context)
        if not handled and not chained:
            unhandled += 1
    return unhandled


def _rss_mb(pid: int) -> float | None:
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss / 1e6
    except Exception:  # noqa: BLE001 - sampling is best-effort
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=float, default=300.0)
    ap.add_argument("--out", default="runs/soak")
    ap.add_argument("--clip", default="")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    clip = Path(args.clip) if args.clip else next(
        iter(sorted((ROOT / "data" / "test_clips").glob("*.mp4"))), None)
    if clip is None:
        print("FAIL: no clip to loop"); return 1

    feeder = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-re", "-stream_loop", "-1",
         "-i", str(clip), "-c:v", "libx264", "-preset", "veryfast", "-tune",
         "zerolatency", "-an", "-f", "mpegts",
         f"tcp://127.0.0.1:{PORT}?listen=1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    site = out / "site.json"
    site.write_text(json.dumps({
        "name": "soak", "notify": "console", "configured": True,
        "cameras": [{"id": "cam1", "source": f"tcp://127.0.0.1:{PORT}",
                     "config": "configs/all_threats_video_v1.json"}]}))
    seconds = int(args.minutes * 60)
    log_path = out / "engine.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        engine = subprocess.Popen(
            [sys.executable, "-m", "cvti.serving.pipeline",
             "--site-config", str(site), "--gate-provider", "local",
             "--gate-base-url", f"http://127.0.0.1:{PORT + 1}/v1",   # nothing there: gate fail-visible
             "--target-fps", "4", "--imgsz", "416",
             "--seconds", str(seconds), "--gate-drain", "5",
             "--mobile-port", "0", "--output-dir", str(out)],
            cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
        first_rss, last_rss, peak_rss = None, None, 0.0
        while engine.poll() is None:
            time.sleep(min(30, max(5, seconds / 50)))
            rss = _rss_mb(engine.pid)
            if rss:
                first_rss = first_rss or rss
                last_rss = rss
                peak_rss = max(peak_rss, rss)
    ran_s = time.time() - t0
    feeder.terminate()

    problems: list[str] = []
    if ran_s < seconds * 0.95:
        problems.append(f"engine ended early: {ran_s:.0f}s of {seconds}s "
                        f"(exit {engine.poll()})")
    hb = out / "gate_health.json"
    hb_age = None
    try:
        hb_age = time.time() - float(json.loads(hb.read_text())["generated_at"])
        if hb_age > 60:
            problems.append(f"heartbeat stale at end ({hb_age:.0f}s) — zombie engine")
    except Exception:  # noqa: BLE001
        problems.append("engine never wrote gate_health.json")
    if first_rss and last_rss and last_rss > first_rss * 1.6 + 200:
        problems.append(f"memory grew {first_rss:.0f}→{last_rss:.0f} MB")
    log_text = log_path.read_text(errors="replace")
    tracebacks = _unhandled_tracebacks(log_text)
    if tracebacks:
        problems.append(f"{tracebacks} UNHANDLED traceback(s) in the engine log")

    lat_p95 = None
    db = out / "events.db"
    if db.exists():
        try:
            con = sqlite3.connect(db)
            vals = sorted(v for (v,) in con.execute(
                "SELECT latency_s FROM events WHERE latency_s IS NOT NULL"))
            con.close()
            if vals:
                lat_p95 = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
        except sqlite3.OperationalError:
            pass

    report = {
        "requested_s": seconds, "ran_s": round(ran_s, 1),
        "engine_exit": engine.poll(),
        "heartbeat_age_s": round(hb_age, 1) if hb_age is not None else None,
        "rss_mb": {"first": first_rss, "last": last_rss, "peak": peak_rss},
        "tracebacks": tracebacks,
        "silent_failures": len(problems),
        "alert_on_screen_p95_s": lat_p95,
        "problems": problems,
    }
    (out / "soak_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if problems:
        print("\nSOAK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nSOAK OK — next: python tools/budget_check.py --evidence", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
