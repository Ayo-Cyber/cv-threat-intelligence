"""perf_readout.py — a support zip names its own bottleneck, in one command.

    python tools/perf_readout.py argus-diagnostics-2026-09-07-1422.zip
    python tools/perf_readout.py runs/site/perf_report.json
    python tools/perf_readout.py runs/site            # the engine's output dir
    python tools/perf_readout.py <zip> --json         # machine-readable

Every pilot conversation about speed has been adjectives, and the 4 Sep
instrumentation build fixed half of that: perf_report.json now says how long
each stage takes. This is the other half — reading it correctly.

WHY THIS IS NOT A SORT BY p95
-----------------------------
The stages do not measure the same thing, so their latencies are not
comparable and the largest one is routinely not the problem:

  decode        one frame, ~8/s — but at the live edge grab() WAITS for the
                camera, so a big number can be a slow SOURCE on a healthy box.
                The honest signal is sustainable_fps vs the rate asked for,
                which the decoder computes and the series now carries.
  detect_batch  one BATCH of N frames. 35ms that bought 4 frames is 8.9ms of
                detection, not 35.
  english_scan  one scan cycle, 1 per ~12s, with a deliberate 120s budget. Its
                p95 tops any raw ranking while costing ~0.4% of a core.
  verify_wait   how long an alert QUEUED before a worker took it. Not work at
                all — a symptom of something else eating the machine. Ranking
                it as cost double-counts the very thing it is diagnosing.
  verify_infer  one VLM verdict.

So this ranks by `busy_fraction` — time spent divided by wall-clock elapsed,
the share of one thread a stage occupied — which IS comparable across stages,
and reports verify_wait separately as saturation. A stage missing from the
report is printed as "not measured": on a box where verification never ran,
that absence is the finding.

Reports written before the 7 Sep schema (no span_s) still parse — the tool
falls back to latency-only and says so rather than inventing a rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stages whose time is WORK, and may be ranked against each other by the share
# of a thread they occupy.
COST_STAGES = ("decode", "detect_batch", "english_scan", "verify_infer")

# Stages whose time is WAITING. Never ranked as cost — reported as pressure.
SATURATION_STAGES = ("verify_wait",)

# Every stage the engine is known to emit, so a missing one can be named.
KNOWN_STAGES = COST_STAGES + SATURATION_STAGES

# An alert that queued longer than this before a worker looked at it is not a
# slow model, it is a starved one. Martins's pilot reported 360s verdicts.
WAIT_ALARM_S = 30.0

# Below this much free memory a 16 GB box is paging, and every latency in the
# file is a symptom of that rather than of the stage it is attributed to.
MEMORY_ALARM_PERCENT = 85.0
MEMORY_ALARM_FREE_GB = 2.0

# Which workstream fixes which named bottleneck (v1.9 work breakdown).
REMEDY = {
    "detect_batch": "W2 (ONNX accelerated detection) — the GPU is idle while torch runs on CPU",
    "decode": "W1 (go2rtc stream gateway) — one RTSP session per camera, decoded once",
    "english_scan": "W3 (YOLO-World rule router) — object rules answered by a detector, not the VLM",
    "verify_infer": "W5 (tiered verification) — pick the verdict model this box can actually run",
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_report(target: str | Path) -> tuple[dict, str]:
    """(perf_report document, where it came from).

    Takes the support zip itself, the bare json, or the engine's output dir —
    because the person running this has whichever one they were sent.
    """
    path = Path(target)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    if path.is_dir():
        candidate = path / "perf_report.json"
        if not candidate.is_file():
            raise SystemExit(f"no perf_report.json in {path}")
        return json.loads(candidate.read_text()), str(candidate)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if Path(n).name == "perf_report.json"]
            if not names:
                raise SystemExit(
                    f"{path.name} carries no perf_report.json — it was captured by a "
                    "build older than 4 Sep, or the engine never started.")
            with zf.open(names[0]) as fh:
                return json.loads(fh.read().decode()), f"{path.name}:{names[0]}"
    return json.loads(path.read_text()), str(path)


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------

def rank_cost(stages: dict) -> list[dict]:
    """Every cost series, most-expensive first by share of a thread."""
    rows = []
    for stage in COST_STAGES:
        for key, doc in (stages.get(stage) or {}).items():
            rows.append({
                "stage": stage,
                "key": key,
                "busy_fraction": doc.get("busy_fraction"),
                "per_unit_ms": doc.get("per_unit_ms", doc.get("mean_ms")),
                "rate_per_s": doc.get("rate_per_s"),
                "p95_ms": doc.get("p95_ms"),
                "count": doc.get("count"),
                "span_s": doc.get("span_s"),
                "sustainable_fps": doc.get("sustainable_fps"),
                "target_fps": doc.get("target_fps"),
                "limited": doc.get("limited"),
            })
    # A series with no busy_fraction (old schema, or a single sample) sorts
    # last rather than as zero — unknown is not cheap.
    rows.sort(key=lambda r: (r["busy_fraction"] is not None, r["busy_fraction"] or 0.0),
              reverse=True)
    return rows


def saturation(stages: dict) -> list[dict]:
    """Waiting, not working — the queue-pressure lines."""
    rows = []
    for stage in SATURATION_STAGES:
        for key, doc in (stages.get(stage) or {}).items():
            rows.append({"stage": stage, "key": key,
                         "p95_ms": doc.get("p95_ms"), "p50_ms": doc.get("p50_ms"),
                         "count": doc.get("count")})
    rows.sort(key=lambda r: r["p95_ms"] or 0.0, reverse=True)
    return rows


def ingest_warnings(stages: dict) -> list[str]:
    """Cameras this machine cannot keep up with, in its own words."""
    out = []
    for key, doc in (stages.get("decode") or {}).items():
        if not doc.get("limited"):
            continue
        out.append(
            f"decode/{key} sustains ~{doc.get('sustainable_fps', '?')} fps of "
            f"{doc.get('target_fps', '?')} asked"
            + (f" ({doc['width']}x{doc['height']})" if doc.get("width") else ""))
    return out


def missing_stages(stages: dict) -> list[str]:
    return [s for s in KNOWN_STAGES if not (stages.get(s) or {})]


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------

def verdict(report: dict) -> dict:
    """The named bottleneck, why, and which workstream answers it.

    Deliberately conservative: where the evidence does not separate two
    explanations, it says so instead of picking one. A confident wrong answer
    here sends a week of work at the wrong stage.
    """
    stages = report.get("stages") or {}
    system = report.get("system") or {}
    cost = rank_cost(stages)
    waits = saturation(stages)
    cores = system.get("cpu_count")
    notes: list[str] = []

    top = next((r for r in cost if r["busy_fraction"] is not None), None)

    # Memory first: on a box that is paging, every other number in the file is
    # a symptom, and attributing them to a stage is a category error.
    mem_pct = system.get("memory_percent")
    free_gb = system.get("memory_available_gb")
    starved_memory = (
        (mem_pct is not None and mem_pct >= MEMORY_ALARM_PERCENT)
        or (free_gb is not None and free_gb <= MEMORY_ALARM_FREE_GB))

    worst_wait = waits[0] if waits else None
    wait_bad = bool(worst_wait and (worst_wait["p95_ms"] or 0) / 1000.0 >= WAIT_ALARM_S)

    if top is None:
        # Two very different reasons a ranking is impossible, and telling the
        # operator the wrong one wastes a support round-trip.
        legacy = bool(cost) and all(r["busy_fraction"] is None for r in cost)
        if legacy:
            head = "Cannot rank: this report predates the span schema (7 Sep)."
            detail = ("The stages have latencies but no wall-clock window, so their "
                      "costs are not comparable. Upgrade the box and re-capture; the "
                      "latencies below are still readable individually.")
        else:
            head = "No stage carries enough samples to rank."
            detail = ("The engine wrote a report before doing measurable work — "
                      "capture a zip after it has been monitoring for a few minutes.")
        if starved_memory:
            notes.append(
                f"Even so, memory is a binding constraint: {free_gb} GB free of "
                f"{system.get('memory_total_gb', '?')} ({mem_pct}% used). Stage timings "
                "on a paging box measure the paging.")
        return {"headline": head, "detail": detail, "remedy": None, "notes": notes}

    if starved_memory:
        # This outranks any stage. A 16 GB box with a 4 GB model resident is
        # not slow at a stage, it is short of RAM, and every share below is
        # inflated by the paging it causes.
        return {
            "headline": (f"Memory: {free_gb} GB free of "
                         f"{system.get('memory_total_gb', '?')} ({mem_pct}% used)."),
            "detail": (f"This box is at its memory ceiling, so every stage timing here "
                       f"measures the paging as much as the work. Top cost stage is "
                       f"{top['stage']}/{top['key']} at {top['busy_fraction']} of a "
                       f"thread — treat that as provisional until memory has headroom."),
            "remedy": ("W5 (tiered verification) — a smaller verdict model is the "
                       "largest single RAM saving on a 16 GB box"),
            "notes": notes}

    # decode is the one stage whose milliseconds are not necessarily cost: at
    # the live edge it waits for the camera. Only the decoder's own sustain
    # verdict can tell the two apart, so demand it before blaming decode.
    if top["stage"] == "decode" and not top.get("limited"):
        alternatives = [r for r in cost if r["stage"] != "decode"]
        nxt = alternatives[0] if alternatives else None
        notes.append(
            "decode tops the cost ranking but this machine is NOT ingest-limited — "
            "those milliseconds are the camera pacing us at the live edge, not work. "
            "Ranking the next stage instead.")
        top = nxt or top

    headline = (f"{top['stage']}/{top['key']} occupies "
                f"{top['busy_fraction']} of a thread"
                + (f" (of {cores} cores)" if cores else ""))
    remedy = REMEDY.get(top["stage"])

    if wait_bad:
        wait_s = round((worst_wait["p95_ms"] or 0) / 1000.0)
        if top["stage"] == "verify_infer":
            detail = (f"Verification is slow in itself AND queueing "
                      f"({wait_s}s p95 wait) — the model is too heavy for this box.")
            remedy = REMEDY["verify_infer"]
        else:
            detail = (f"Verification is STARVED, not slow: alerts wait {wait_s}s p95 "
                      f"behind {top['stage']}. Fix {top['stage']} first — a lighter "
                      "verdict model alone will not clear this queue.")
        notes.append(f"verify_wait p95 {wait_s}s exceeds the {WAIT_ALARM_S:.0f}s alarm.")
    else:
        detail = "No queue pressure — verification is keeping up with what reaches it."

    return {"headline": headline, "detail": detail, "remedy": remedy, "notes": notes}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _fmt_ms(ms) -> str:
    if ms is None:
        return "—"
    return f"{ms/1000.0:.1f} s" if ms >= 1000 else f"{ms:.1f} ms"


def _fmt_rate(per_s) -> str:
    """A verdict every 60s is 0.017/s, which renders as '0.0/s' and reads as
    nothing at all. Slow stages get their period instead."""
    if not per_s:
        return "—"
    if per_s >= 0.1:
        return f"{per_s:.1f}/s"
    return f"1/{round(1.0 / per_s)}s"


def render(report: dict, origin: str) -> str:
    import time as _time
    stages = report.get("stages") or {}
    system = report.get("system") or {}
    lines: list[str] = []
    generated = report.get("generated_at")
    when = (_time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(generated))
            if generated else "unknown time")

    lines.append(f"ARGUS PERF READOUT — {origin}")
    lines.append(f"captured {when}")
    if system:
        lines.append(
            f"box: {system.get('memory_total_gb', '?')} GB, "
            f"{system.get('cpu_count', '?')} cores · "
            f"cpu {system.get('cpu_percent', '?')}% · "
            f"mem {system.get('memory_percent', '?')}% "
            f"({system.get('memory_available_gb', '?')} GB free)"
            + (f" · load1m {system['loadavg_1m']}" if "loadavg_1m" in system else ""))
    lines.append("")

    cost = rank_cost(stages)
    legacy = any(r["busy_fraction"] is None for r in cost)
    lines.append(f"{'RANKED BY COST':<34}{'share':>8}{'per unit':>12}"
                 f"{'rate':>11}{'p95':>11}")
    if not cost:
        lines.append("  (no cost stages measured)")
    for r in cost:
        share = f"{r['busy_fraction']:.2f} x" if r["busy_fraction"] is not None else "—"
        rate = _fmt_rate(r.get("rate_per_s"))
        lines.append(f"  {r['stage']}/{r['key']:<{max(1, 30 - len(r['stage']))}}"
                     f"{share:>8}{_fmt_ms(r['per_unit_ms']):>12}"
                     f"{rate:>11}{_fmt_ms(r['p95_ms']):>11}")
    if legacy:
        lines.append("  ! some series predate the 7 Sep schema (no span recorded) — "
                     "latency only, not rankable by cost")
    known = [r["busy_fraction"] for r in cost if r["busy_fraction"] is not None]
    if known:
        total = sum(known)
        cores = system.get("cpu_count")
        verdict_word = ""
        if cores:
            verdict_word = ("  OVERSUBSCRIBED" if total > cores * 0.9
                            else f"  ({total / cores:.0%} of the box)")
        lines.append(f"  {'TOTAL':<32}{total:>6.2f} x"
                     + (f" of {cores} cores{verdict_word}" if cores else " (core count unknown)"))
    lines.append("")

    pressure = saturation(stages)
    warn = ingest_warnings(stages)
    if pressure or warn:
        lines.append("SATURATION")
        for r in pressure:
            flag = "!" if (r["p95_ms"] or 0) / 1000.0 >= WAIT_ALARM_S else " "
            lines.append(f"  {flag} {r['stage']}/{r['key']} p95 {_fmt_ms(r['p95_ms'])} "
                         f"(p50 {_fmt_ms(r['p50_ms'])}, n={r['count']}) — alerts queueing")
        for w in warn:
            lines.append(f"  ! {w} — ingest limited")
        lines.append("")

    absent = missing_stages(stages)
    if absent:
        lines.append("NOT MEASURED")
        for stage in absent:
            why = {"verify_infer": "no verdicts reached the gate in this window",
                   "verify_wait": "nothing queued for verification",
                   "english_scan": "no camera has plain-English rules",
                   "decode": "no camera decoded",
                   "detect_batch": "detection never ran"}.get(stage, "")
            lines.append(f"  - {stage}{' — ' + why if why else ''}")
        lines.append("")

    v = verdict(report)
    lines.append("NAMED BOTTLENECK")
    lines.append(f"  {v['headline']}")
    lines.append(f"  {v['detail']}")
    for note in v["notes"]:
        lines.append(f"  · {note}")
    if v["remedy"]:
        lines.append(f"  -> {v['remedy']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Name the top-cost stage in an Argus diagnostics zip.")
    ap.add_argument("target", help="a diagnostics .zip, a perf_report.json, or an output dir")
    ap.add_argument("--json", action="store_true",
                    help="emit the ranking and verdict as JSON")
    args = ap.parse_args(argv)

    report, origin = load_report(args.target)
    if args.json:
        print(json.dumps({
            "origin": origin,
            "generated_at": report.get("generated_at"),
            "system": report.get("system") or {},
            "cost": rank_cost(report.get("stages") or {}),
            "saturation": saturation(report.get("stages") or {}),
            "ingest_warnings": ingest_warnings(report.get("stages") or {}),
            "not_measured": missing_stages(report.get("stages") or {}),
            "verdict": verdict(report),
        }, indent=2))
    else:
        print(render(report, origin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
