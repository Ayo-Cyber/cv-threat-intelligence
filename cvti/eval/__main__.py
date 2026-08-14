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
from cvti.eval.harness import EvalHarness
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
    if args.fresh:
        (out_dir / "clip_results.jsonl").unlink(missing_ok=True)

    gate = _build_gate(args.gate, args.gate_model, str(out_dir / "gate"))
    harness = EvalHarness(detectors=tuple(d.strip() for d in args.detectors.split(",") if d.strip()),
                          gate=gate, target_fps=args.target_fps,
                          max_seconds_per_clip=args.max_seconds, out_dir=args.out)
    results = harness.run(clips)

    rows = [r.to_dict() for r in results]
    summary = compare_stages(rows)
    meta = {"gate": args.gate, "gate_model": args.gate_model if args.gate == "ollama" else None,
            "detectors": args.detectors, "dataset": args.dataset,
            "max_seconds_per_clip": args.max_seconds}
    payload = {"meta": meta, "dataset": info, "summary": summary, "clips": rows}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    report = render_report(summary, info, meta)
    (out_dir / "report.md").write_text(report)

    print("\n" + report)
    print(f"[eval] wrote {out_dir/'metrics.json'} and {out_dir/'report.md'}")
    if args.gate == "mock":
        print("[eval] NOTE: --gate mock confirms everything. Re-run with --gate ollama "
              "for real suppression numbers.")


if __name__ == "__main__":
    main()
