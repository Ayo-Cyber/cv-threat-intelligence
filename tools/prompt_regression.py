"""Prompt regression: did that wording change cost us precision?

    python tools/prompt_regression.py capture   # freeze the detector stage (slow, once)
    python tools/prompt_regression.py run       # replay it through the current prompts
    python tools/prompt_regression.py check     # CI: has the wording changed unmeasured?

**Why `check` does not measure.** Measuring means running the VLM, and the VLM
is a 3 GB model. A GitHub runner has no GPU and no Ollama; a real measurement
there would take longer than the build and still not be the model we ship. So CI
enforces the one thing it *can* enforce honestly: the prompt text and the
recorded measurement must agree. Change the wording without re-measuring and the
build fails, telling you which command to run.

That is a weaker guarantee than "CI measured it", and it is the true one. The
alternative — a green tick from a measurement CI could not perform — would be
worse than nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.eval.golden import DEFAULT_GOLDEN_DIR, GoldenSet, score  # noqa: E402
from cvti.eval.prompt_fingerprint import describe, fingerprint  # noqa: E402
from cvti.logging_setup import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)

BASELINE = ROOT / "docs" / "prompt_baseline.json"

# How far a rate may move before the build fails.
#
# Justification: the golden set is a few hundred candidates, so the 95% interval
# on a rate near 60% is roughly +/-8 points. A tolerance below that would fail on
# sampling noise and be switched off within a week. 5 points is inside the
# interval but far below the 26-point swing three prompt revisions actually
# produced — it catches the thing this exists to catch without crying wolf.
#
# Recall is tighter than precision on purpose. Losing precision means an
# operator reviews some extra alerts. Losing recall means a threat is not
# reported, and there is no second chance at that.
TOLERANCE = {"precision": 0.05, "recall": 0.03}


def _pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def cmd_capture(args) -> int:
    """Run the detector stage once and freeze every candidate it produced."""
    from cvti.eval.dataset import load_dataset
    from cvti.eval.golden import GoldenSetWriter
    from cvti.eval.harness import EvalHarness

    clips = load_dataset(args.dataset, limit=args.limit, kind=args.kind or "")
    log.info("capture: %d clip(s) from %s", len(clips), args.dataset)
    writer = GoldenSetWriter(args.golden_dir)

    detectors = tuple(d.strip() for d in args.detectors.split(",") if d.strip()) \
        or ("concealment", "video_action")
    harness = EvalHarness(config=args.config, detectors=detectors,
                          max_seconds_per_clip=args.max_seconds)
    for i, clip in enumerate(clips, 1):
        log.info("[capture] %d/%d %s (%s)", i, len(clips), clip.name,
                 "threat" if clip.is_threat else "normal")
        alerts = _candidates_for(harness, clip)
        for alert in alerts:
            payload = alert.payload or {}
            writer.add(clip_name=clip.name, is_threat=clip.is_threat,
                       candidate=payload.get("candidate"),
                       frames=payload.get("frames") or [],
                       scene=payload.get("scene") or {})
    writer.write({"dataset": args.dataset, "kind": args.kind, "clips": len(clips),
                  "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log.info("capture: %d candidate(s) frozen", len(writer.cases))
    return 0


def _candidates_for(harness, clip):
    """Every candidate the detectors raise on one clip, gate stage skipped."""
    collected = []
    original = harness._confirm

    def collect(alert):
        collected.append(alert)
        return False           # never confirm: we only want the candidates

    harness._confirm = collect
    harness.gate = object()    # non-None so the harness takes the gated path
    try:
        harness.run_clip(clip)
    finally:
        harness._confirm = original
    return collected


def cmd_run(args) -> int:
    """Replay the frozen set through the current prompts and score it.

    Progress is written per-verdict to <golden-dir>/replay.jsonl, and a rerun
    resumes from it — so the ~20-minute measurement can run in short chunks
    (--limit N) and an interruption costs one case, not the run. --fresh
    discards the progress file and starts over.
    """
    from cvti.verification.gate import VerificationGate

    golden = GoldenSet(args.golden_dir)
    resume = Path(args.golden_dir) / "replay.jsonl"
    if args.fresh and resume.exists():
        resume.unlink()
        log.info("--fresh: discarded previous replay progress")

    log.info("replaying %d frozen candidate(s) through %s/%s",
             len(golden), args.gate_provider, args.gate_model or "default")

    gate = VerificationGate(provider=args.gate_provider, model=args.gate_model,
                            sensitivity=args.sensitivity)

    def progress(i, total, row):
        log.info("[replay] %d/%d %s -> %s%s", i, total, row["case_id"],
                 "CONFIRMED" if row["confirmed"] else "rejected",
                 f"  ERROR {row['error']}" if row["error"] else "")

    verdicts = golden.replay(gate, progress=progress if args.verbose else None,
                             resume_path=resume, limit=args.limit)
    if len(verdicts) < len(golden):
        log.info("partial run: %d/%d case(s) answered so far — rerun to continue "
                 "(progress kept in %s)", len(verdicts), len(golden), resume)
    result = score(verdicts)
    result.update({"fingerprint": fingerprint(), "prompts": describe()["constants"],
                   "sensitivity": args.sensitivity, "gate_model": args.gate_model,
                   "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "golden_cases": len(golden)})

    print(json.dumps(result, indent=2))
    if result["errors"]:
        log.warning("%d case(s) errored and were excluded from scoring — a gate "
                    "error is not a rejection", result["errors"])

    if args.update_baseline:
        # An incomplete measurement must never become the yardstick.
        if len(verdicts) < len(golden):
            log.error("refusing --update-baseline: only %d/%d cases measured — "
                      "rerun (it resumes) until the set is complete",
                      len(verdicts), len(golden))
            return 2
        BASELINE.write_text(json.dumps({"tolerance": TOLERANCE, **result}, indent=2) + "\n")
        log.info("baseline updated: %s", BASELINE)
        # Archive the progress file: the measurement is banked, and a future
        # run must re-verify rather than silently re-score stale verdicts.
        if resume.exists():
            resume.rename(resume.with_name(f"replay-{time.strftime('%Y%m%d_%H%M%S')}.jsonl"))
        return 0

    return _compare(result)


def _compare(result: dict) -> int:
    if not BASELINE.exists():
        log.error("no baseline at %s — run with --update-baseline first", BASELINE)
        return 2
    base = json.loads(BASELINE.read_text())
    tol = base.get("tolerance", TOLERANCE)
    failed = []
    for metric in ("precision", "recall"):
        was, now = base.get(metric), result.get(metric)
        if was is None or now is None:
            continue
        delta = now - was
        allowed = tol.get(metric, 0.05)
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
        line = (f"{metric:9} {_pct(was)} -> {_pct(now)}  {arrow} {delta * 100:+.1f} pts "
                f"(tolerance ±{allowed * 100:.0f})")
        if delta < -allowed:
            failed.append(line)
            log.error("REGRESSION  %s", line)
        else:
            log.info("ok          %s", line)
    if failed:
        log.error("prompt regression: %d metric(s) beyond tolerance", len(failed))
        return 1
    log.info("prompt regression: within tolerance")
    return 0


def cmd_check(args) -> int:
    """CI: the prompt text and the recorded measurement must agree."""
    current = fingerprint()
    if not BASELINE.exists():
        print(f"::error::No prompt baseline at {BASELINE}. Run:\n"
              f"  python tools/prompt_regression.py run --update-baseline")
        return 2
    base = json.loads(BASELINE.read_text())
    recorded = base.get("fingerprint")
    if recorded == current:
        print(f"Prompt fingerprint matches the recorded measurement "
              f"({current[:12]}…): precision {_pct(base.get('precision'))}, "
              f"recall {_pct(base.get('recall'))} on {base.get('golden_cases')} candidates.")
        return 0
    print("::error::Gate prompt text changed, but the recorded measurement did not.")
    print(f"  recorded fingerprint: {recorded}")
    print(f"  current  fingerprint: {current}")
    print("")
    print("  The headline precision figure is a function of this wording — three")
    print("  revisions moved it 26 points. Re-measure before merging:")
    print("")
    print("      ollama serve &")
    print("      python tools/prompt_regression.py run            # compare to baseline")
    print("      python tools/prompt_regression.py run --update-baseline   # accept it")
    print("")
    print(f"  and commit {BASELINE.relative_to(ROOT)}.")
    return 1


def main() -> int:
    setup_logging(component="argus-prompt-regression")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="freeze the detector stage (slow, run once)")
    cap.add_argument("--dataset", default="camnuvem")
    cap.add_argument("--kind", default="")
    cap.add_argument("--limit", type=int, default=0)
    cap.add_argument("--config", default="configs/all_threats_video_v1.json")
    cap.add_argument("--detectors", default="")
    cap.add_argument("--max-seconds", type=float, default=15.0)
    cap.add_argument("--golden-dir", default=str(DEFAULT_GOLDEN_DIR))
    cap.set_defaults(func=cmd_capture)

    run = sub.add_parser("run", help="replay the frozen set through current prompts")
    run.add_argument("--golden-dir", default=str(DEFAULT_GOLDEN_DIR))
    run.add_argument("--gate-provider", default="ollama")
    run.add_argument("--gate-model", default="gemma3:4b")
    run.add_argument("--sensitivity", default="balanced")
    run.add_argument("--update-baseline", action="store_true",
                     help="accept these numbers as the new baseline")
    run.add_argument("--limit", type=int, default=0,
                     help="verify at most N new cases this run (resume completes the rest)")
    run.add_argument("--fresh", action="store_true",
                     help="discard replay progress and start over")
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=cmd_run)

    chk = sub.add_parser("check", help="CI: prompt text vs recorded measurement")
    chk.set_defaults(func=cmd_check)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
