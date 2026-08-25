"""Measure the two critical always-on detectors: weapons and violence (EP-07-T3).

    python tools/measure_critical.py status                # what clips do we have?
    python tools/measure_critical.py run violence          # measure one detector
    python tools/measure_critical.py run weapons --smoke   # wiring check on a small set

The two detectors that would summon an armed response have never been
evaluated — this is the harness that ends that, the moment clips exist.

Where clips come from (see docs/BACKLOG.md · EP-07-T3):
    data/critical/violence/{threat,normal}/*.mp4     curated, any origin
    data/critical/weapons/{threat,normal}/*.mp4
    data/ucf_crime/<Category>/...                    official UCF-Crimes layout
                                                     (Fighting/Assault -> violence,
                                                      Shooting -> weapons, Normal* -> normal)

The guard: acceptance asks for >=50 threat clips per detector. Below that this
tool refuses to produce a publishable number — small-n rates are noise wearing
a percent sign. --smoke runs anyway, labels the output SMOKE everywhere, and
archives nothing.

A real run costs VLM time (~12s/clip-candidate on gemma3:4b) and requires the
Ollama server up. Results land in runs/eval/<kind>-v1/metrics.json — the same
archive format the numbers sheet and SENSITIVITY_MEASURED generators read —
and the tool prints the exact DETECTOR_VALIDATION and NUMBERS.md lines to
update, which tests then hold consistent.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.eval.dataset import load_dataset  # noqa: E402
from cvti.eval.metrics import wilson_interval  # noqa: E402

MIN_THREAT = 50     # the acceptance line, verbatim
MIN_NORMAL = 20     # below this, FPR is a guess

KINDS = ("violence", "weapons")


def counts(kind: str) -> tuple:
    clips = load_dataset("ucf_crime", kind=kind)
    threat = sum(1 for c in clips if c.is_threat and c.kind == kind)
    normal = sum(1 for c in clips if not c.is_threat)
    return threat, normal


def cmd_status(_args) -> int:
    print(f"{'detector':10} {'threat':>7} {'normal':>7}   verdict")
    ok = True
    for kind in KINDS:
        t, n = counts(kind)
        enough = t >= MIN_THREAT and n >= MIN_NORMAL
        ok = ok and enough
        verdict = "ready to measure" if enough else \
            f"need >={MIN_THREAT} threat + >={MIN_NORMAL} normal (docs/BACKLOG.md has sources)"
        print(f"{kind:10} {t:>7} {n:>7}   {verdict}")
    return 0 if ok else 1


def cmd_run(args) -> int:
    kind = args.kind
    t, n = counts(kind)
    if (t < MIN_THREAT or n < MIN_NORMAL) and not args.smoke:
        print(f"::error::{kind}: {t} threat / {n} normal clip(s) — below the "
              f"publishable floor (>={MIN_THREAT}/{MIN_NORMAL}).")
        print("  Add clips under data/critical/ or data/ucf_crime/ "
              "(sources: docs/BACKLOG.md · EP-07-T3), or pass --smoke for a "
              "wiring check that archives nothing.")
        return 2
    out = ROOT / "runs" / "eval" / (f"{kind}-smoke" if args.smoke else f"{kind}-v1")
    cmd = [sys.executable, "-m", "cvti.eval",
           "--dataset", "ucf_crime", "--kind", kind, "--detectors", kind,
           "--gate", args.gate, "--gate-model", args.gate_model,
           "--sensitivity", args.sensitivity, "--out", str(out),
           "--max-seconds", str(args.max_seconds),
           "--max-candidates-per-clip", str(args.max_candidates)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    print("running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        return rc
    return _report(kind, out, smoke=args.smoke)


def _report(kind: str, out: Path, *, smoke: bool) -> int:
    m = json.loads((out / "metrics.json").read_text())
    clips = m["clips"]
    threat = [c for c in clips if c["is_threat"]]
    normal = [c for c in clips if not c["is_threat"]]
    flagged = lambda c: c["confirmed"] > 0  # noqa: E731
    tp = sum(1 for c in threat if flagged(c))
    fp = sum(1 for c in normal if flagged(c))
    recall = tp / len(threat) if threat else None
    fpr = fp / len(normal) if normal else None
    precision = tp / (tp + fp) if (tp + fp) else None
    lo, hi = wilson_interval(tp, len(threat)) if threat else (None, None)

    capped = sum(1 for c in clips if c.get("capped"))
    verified = sum(c.get("verified", c.get("candidates", 0)) for c in clips)
    label = "SMOKE — NOT A MEASUREMENT" if smoke else "measured"
    print(f"\n[{label}] {kind}: recall {recall:.1%} (n={len(threat)}, "
          f"CI {lo:.1%}–{hi:.1%}), precision "
          f"{('%.1f%%' % (precision*100)) if precision is not None else '—'}, "
          f"FPR {fpr:.1%} (n={len(normal)})")
    print(f"  {verified} VLM verdicts over {len(clips)} clips"
          + (f" — {capped} clip(s) hit the per-clip cap, so recall is a LOWER BOUND"
             if capped else ""))
    if smoke:
        print("smoke run: numbers above are wiring proof only; nothing to publish.")
        return 0
    print("\nNow make the claims match the measurement (tests enforce both):")
    print(f'  cvti/app/console_backend.py DETECTOR_VALIDATION["{kind}"] =')
    print(f'      {{"measured": True, "summary": "{recall:.1%} caught (n={len(threat)}) '
          f'· {fpr:.1%} false alarms"}}')
    print(f"  docs/NUMBERS.md: flip the '{kind}' coverage row to "
          f"'✅ measured — {recall:.1%} caught (n={len(threat)}, CI {lo:.1%}–{hi:.1%}), "
          f"{fpr:.1%} false alarms'")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status", help="clip inventory vs the >=50 floor")
    st.set_defaults(func=cmd_status)
    run = sub.add_parser("run", help="measure one critical detector")
    run.add_argument("kind", choices=KINDS)
    run.add_argument("--gate", default="ollama", choices=("ollama", "mock"))
    run.add_argument("--gate-model", default="gemma3:4b")
    run.add_argument("--sensitivity", default="balanced",
                     choices=("sensitive", "balanced", "strict"))
    run.add_argument("--max-seconds", type=float, default=30.0)
    run.add_argument("--max-candidates", type=int, default=0,
                     help="cap VLM calls per clip (0 = no cap). Biases recall DOWN, "
                          "never up; capped runs are labelled in the report.")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--smoke", action="store_true",
                     help="run below the clip floor; loudly non-publishable")
    run.set_defaults(func=cmd_run)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
