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


def _build_gate(kind: str, model: str, save_dir: str | None):
    if kind == "none":
        return None
    from cvti.verification.gate import VerificationGate
    provider = "mock" if kind == "mock" else kind
    return VerificationGate(provider=provider, model=("" if kind == "mock" else model),
                            save_dir=save_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Measure detection + TrueSight suppression.")
    p.add_argument("--dataset", choices=("camnuvem", "local", "all"), default="camnuvem",
                   help="camnuvem = the held-out test split (the honest one).")
    p.add_argument("--limit", type=int, default=0, help="cap total clips (keeps both classes).")
    p.add_argument("--gate", choices=("ollama", "mock", "none"), default="mock",
                   help="ollama = real measurement; mock = wiring check; none = Stage 1 only.")
    p.add_argument("--gate-model", default="gemma3:4b")
    p.add_argument("--detectors", default="concealment,video_action",
                   help="comma-separated detectors to enable.")
    p.add_argument("--max-seconds", type=float, default=30.0, help="max seconds analysed per clip.")
    p.add_argument("--target-fps", type=float, default=4.0)
    p.add_argument("--out", default="runs/eval")
    p.add_argument("--fresh", action="store_true", help="ignore checkpoints and re-run everything.")
    args = p.parse_args()

    clips = load_dataset(args.dataset, limit=args.limit)
    if not clips:
        print(f"[eval] no clips found for --dataset {args.dataset}. "
              "Is the CamNuvem dataset present?")
        raise SystemExit(1)
    info = describe(clips)
    print(f"[eval] {info['total']} clips ({info['threat']} threat / {info['normal']} normal) "
          f"| gate={args.gate} | detectors={args.detectors}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Results are only comparable within the same gate + detector set, so the
    # checkpoint is keyed by them (a mock run can never be resumed as a real one).
    run_key = f"{args.gate}-{args.detectors.replace(',', '+')}"
    if args.fresh:
        for f in out_dir.glob("clip_results_*.jsonl"):
            f.unlink(missing_ok=True)

    gate = _build_gate(args.gate, args.gate_model, str(out_dir / "gate"))
    harness = EvalHarness(detectors=tuple(d.strip() for d in args.detectors.split(",") if d.strip()),
                          gate=gate, target_fps=args.target_fps,
                          max_seconds_per_clip=args.max_seconds, out_dir=args.out,
                          run_key=run_key)
    try:
        results = harness.run(clips)
    except GateUnavailable as exc:
        print(f"\n[eval] ABORTED — {exc}")
        print("[eval] No numbers written: a run without a working gate would report "
              "everything as 'suppressed', which is meaningless.")
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("\n[eval] interrupted — completed clips are checkpointed; "
              "re-run the same command to resume.")
        raise SystemExit(130) from None

    rows = [r.to_dict() for r in results]
    summary = compare_stages(rows)
    meta = {"gate": args.gate, "gate_model": args.gate_model if args.gate == "ollama" else None,
            "detectors": args.detectors, "dataset": args.dataset,
            "max_seconds_per_clip": args.max_seconds}
    payload = {"meta": meta, "dataset": info, "summary": summary, "clips": rows}
    report = render_report(summary, info, meta)
    # latest-run files, plus gate-tagged copies so a mock run never overwrites a
    # real measured one (and vice versa)
    tag = args.gate
    for name, text in (("metrics.json", json.dumps(payload, indent=2)), ("report.md", report)):
        (out_dir / name).write_text(text)
        stem, _, ext = name.partition(".")
        (out_dir / f"{stem}-{tag}.{ext}").write_text(text)

    print("\n" + report)
    print(f"[eval] wrote {out_dir/'metrics.json'} and {out_dir/'report.md'} "
          f"(also tagged -{tag})")
    if args.gate == "mock":
        print("[eval] NOTE: --gate mock confirms everything. Re-run with --gate ollama "
              "for real suppression numbers.")


if __name__ == "__main__":
    main()
