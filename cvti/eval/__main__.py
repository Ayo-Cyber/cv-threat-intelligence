"""Run the evaluation.

    # quick sanity check — 6 clips, no VLM, ~2 min, tiny RAM
    python -m cvti.eval --limit 6 --gate mock

    # the real measured pass (needs Ollama; ~30-60 min, ~3 GB RAM)
    python -m cvti.eval --gate ollama

Writes runs/eval/metrics.json + runs/eval/report.md. Safe to Ctrl+C — completed
clips are checkpointed to runs/eval/clip_results.jsonl and skipped on the next run
(use --fresh to start over).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvti.eval.dataset import describe, load_dataset
from cvti.eval.harness import EvalHarness, GateUnavailable
from cvti.eval.metrics import compare_stages, render_report
from cvti.logging_setup import get_logger

log = get_logger(__name__)


def _build_gate(kind: str, model: str, save_dir: str | None, sensitivity: str = "balanced"):
    if kind == "none":
        return None
    from cvti.verification.gate import VerificationGate
    provider = "mock" if kind == "mock" else kind
    return VerificationGate(provider=provider, model=("" if kind == "mock" else model),
                            save_dir=save_dir, sensitivity=sensitivity)


# Which detector actually makes each claim. Measuring "weapons" with the
# concealment detector is not a weak result, it is a wrong one.
_KIND_DETECTORS = {
    "weapons": "weapons",
    "violence": "violence",
    "fall": "fall",
    "tamper": "tamper",
    "fire": "fire_smoke",
    "crowd": "crowd_formation",
    "running": "running",
    "theft": "concealment,video_action",
}


def main() -> None:
    # Entrypoint: without this, log records have no handler and
    # anything below WARNING is silently discarded.
    from cvti.logging_setup import setup_logging
    setup_logging(component="argus-eval-harness")
    p = argparse.ArgumentParser(description="Measure detection + TrueSight suppression.")
    p.add_argument("--dataset", choices=("camnuvem", "local", "all", "critical", "ucf_crime"),
                   default="camnuvem",
                   help="camnuvem = held-out theft split; critical = data/critical/<kind>/ "
                        "(EP-07-T3); ucf_crime = data/ucf_crime official layout + critical.")
    p.add_argument("--limit", type=int, default=0, help="cap total clips (keeps both classes).")
    p.add_argument("--kind", default="", help="measure one threat only: fire|crowd|theft|violence")
    p.add_argument("--gate", choices=("ollama", "mock", "none"), default="mock",
                   help="ollama = real measurement; mock = wiring check; none = Stage 1 only.")
    p.add_argument("--gate-model", default="gemma3:4b")
    p.add_argument("--sensitivity", choices=("sensitive", "balanced", "strict"),
                   default="balanced", help="verification strictness to measure")
    p.add_argument("--detectors", default=None,
                   help="comma-separated detectors to enable. Defaults to the "
                        "detector that makes the claim named by --kind.")
    p.add_argument("--max-seconds", type=float, default=30.0, help="max seconds analysed per clip.")
    p.add_argument("--max-candidates-per-clip", type=int, default=0,
                   help="cap VLM calls per clip (0 = uncapped). Lowers recall, never raises it.")
    p.add_argument("--no-dedup", action="store_true",
                   help="verify EVERY detector proposal (measures a firehose production "
                        "never sends; default applies the product's own queue dedup)")
    p.add_argument("--verify-every-candidate", action="store_true",
                   help="do NOT stop a clip after its first confirmation (slower, same result)")
    p.add_argument("--target-fps", type=float, default=4.0)
    p.add_argument("--imgsz", type=int, default=640, help="detector input size")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default="runs/eval")
    p.add_argument("--fresh", action="store_true", help="ignore checkpoints and re-run everything.")
    args = p.parse_args()
    # --kind selects the CLIPS; it must also select the DETECTOR, or the run
    # measures the wrong thing under the right name. A `--kind weapons` pass
    # once ran with the default concealment detector and produced a tidy
    # "16% recall on weapons" — a number describing the shoplifting detector's
    # opinion of shooting footage. It would have gone into NUMBERS.md as a
    # weapons figure. An explicit --detectors still wins.
    if args.detectors is None:
        args.detectors = _KIND_DETECTORS.get(args.kind or "", "concealment,video_action")
        if args.kind:
            log.info(f"[eval] --kind {args.kind} -> detectors={args.detectors}")

    clips = load_dataset(args.dataset, limit=args.limit, kind=args.kind)
    if not clips:
        log.info(f"[eval] no clips found for --dataset {args.dataset}. "
              "Is the CamNuvem dataset present?")
        raise SystemExit(1)
    info = describe(clips)
    log.info(f"[eval] {info['total']} clips ({info['threat']} threat / {info['normal']} normal) "
          f"| gate={args.gate} | detectors={args.detectors}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Results are only comparable within the same gate + detector set, so the
    # checkpoint is keyed by them (a mock run can never be resumed as a real one).
    run_key = f"{args.gate}-{args.sensitivity}-{args.kind or 'all'}-{args.detectors.replace(',', '+')}-{args.imgsz}"
    if args.fresh:
        for f in out_dir.glob("clip_results_*.jsonl"):
            f.unlink(missing_ok=True)

    gate = _build_gate(args.gate, args.gate_model, str(out_dir / "gate"), args.sensitivity)
    harness = EvalHarness(detectors=tuple(d.strip() for d in args.detectors.split(",") if d.strip()),
                          gate=gate, target_fps=args.target_fps,
                          imgsz=args.imgsz, conf=args.conf,
                          max_seconds_per_clip=args.max_seconds, out_dir=args.out,
                          run_key=run_key,
                          stop_on_first_confirm=not args.verify_every_candidate,
                          max_candidates_per_clip=args.max_candidates_per_clip,
                          dedup_like_production=not args.no_dedup)
    try:
        results = harness.run(clips)
    except GateUnavailable as exc:
        log.info(f"\n[eval] ABORTED — {exc}")
        log.info("[eval] No numbers written: a run without a working gate would report "
              "everything as 'suppressed', which is meaningless.")
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        log.info("\n[eval] interrupted — completed clips are checkpointed; "
              "re-run the same command to resume.")
        raise SystemExit(130) from None

    rows = [r.to_dict() for r in results]
    summary = compare_stages(rows)
    meta = {"gate": args.gate, "sensitivity": args.sensitivity, "gate_model": args.gate_model if args.gate == "ollama" else None,
            "detectors": args.detectors, "dataset": args.dataset,
            "max_seconds_per_clip": args.max_seconds}
    payload = {"meta": meta, "dataset": info, "summary": summary, "clips": rows}
    report = render_report(summary, info, meta)
    # latest-run files, plus gate-tagged copies so a mock run never overwrites a
    # real measured one (and vice versa)
    tag = f"{args.gate}-{args.sensitivity}-{args.kind or 'all'}"
    for name, text in (("metrics.json", json.dumps(payload, indent=2)), ("report.md", report)):
        (out_dir / name).write_text(text)
        stem, _, ext = name.partition(".")
        (out_dir / f"{stem}-{tag}.{ext}").write_text(text)

    log.info("\n" + report)
    log.info(f"[eval] wrote {out_dir/'metrics.json'} and {out_dir/'report.md'} "
          f"(also tagged -{tag})")
    if args.gate == "mock":
        log.info("[eval] NOTE: --gate mock confirms everything. Re-run with --gate ollama "
              "for real suppression numbers.")


if __name__ == "__main__":
    main()
