"""kpi_scorecard.py — the customer's sheet, answered with measurements.

    python tools/kpi_scorecard.py --gate mock              # wiring check, free
    python tools/kpi_scorecard.py --gate ollama            # the real numbers
    python tools/kpi_scorecard.py --rows theft,suspicious  # one row at a time
    python tools/kpi_scorecard.py --smoke                  # tiny subset (CI)

Runs the REAL detection path (cvti.eval.harness — the same PerCameraState the
engine uses) over the frozen KPI manifest and scores every sheet row with
Wilson bounds: MET only when the conservative bound clears the target, SMOKE
when a row is below the n floor. Results land in runs/eval/kpi/ as scorecard
.json + .md, resumable per clip like every harness run.

The gate matters: --gate mock answers the wiring and Stage-1 candidate counts;
the numbers anyone SIGNS come from --gate ollama, which costs real VLM time
(~12s per candidate) — on this repo's rules that run is started by a human on
a machine that has the RAM to spare, never casually. A mock scorecard says
MOCK on every surface for exactly that reason.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.eval.dataset import EvalClip  # noqa: E402
from cvti.eval.kpi import (  # noqa: E402
    KPI_ROWS, load_manifest, render_scorecard, score_row,
)

OUT = ROOT / "runs" / "eval" / "kpi"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", choices=("mock", "ollama"), default="mock")
    ap.add_argument("--gate-model", default="gemma3:4b")
    ap.add_argument("--rows", default="",
                    help="comma-separated row keys (default: all)")
    ap.add_argument("--smoke", action="store_true",
                    help="3 clips per row — wiring check only, always SMOKE")
    ap.add_argument("--manifest", default="")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest) if args.manifest else None)
    wanted = {r.strip() for r in args.rows.split(",") if r.strip()} or \
             {row.key for row in KPI_ROWS}

    from cvti.eval.harness import EvalHarness
    from cvti.verification.gate import VerificationGate
    # The same gate builder semantics as cvti.eval.__main__: mock is a REAL
    # VerificationGate in mock provider mode (candidates flow, verdicts are
    # canned) — not gate=None, which verifies nothing and reads as 0% recall.
    gate = VerificationGate(provider=args.gate,
                            model=(args.gate_model if args.gate == "ollama" else ""),
                            save_dir=str(OUT / "gate"))

    OUT.mkdir(parents=True, exist_ok=True)
    scored = []
    for row in KPI_ROWS:
        if row.key not in wanted:
            continue
        row_doc = manifest["rows"][row.key]
        clips = [EvalClip(str(ROOT / c["path"]) if not c["path"].startswith("/")
                          else c["path"],
                          c["is_threat"], c["kind"], c["source"])
                 for c in row_doc["clips"]]
        if args.smoke:
            clips = clips[:3]
        if not clips:
            scored.append(score_row(row_doc, []))
            continue
        harness = EvalHarness(detectors=row.detectors, gate=gate,
                              out_dir=str(OUT),
                              run_key=f"{row.key}-{args.gate}")
        results = harness.run(clips, progress=True)
        s = score_row(row_doc, results)
        if args.smoke:
            s["verdict"] = "SMOKE (subset run)"
            s["publishable"] = False
        scored.append(s)

    banner = "" if args.gate == "ollama" else \
        "MOCK GATE — wiring check only; nothing here is a signable number\n"
    text = banner + render_scorecard(scored, manifest["digest"])
    doc = {"generated_at": time.time(), "gate": args.gate,
           "smoke": args.smoke, "digest": manifest["digest"], "rows": scored}
    (OUT / "scorecard.json").write_text(json.dumps(doc, indent=1))
    (OUT / "scorecard.md").write_text("```\n" + text + "\n```\n")
    print(text)
    print(f"\nwritten: {OUT.relative_to(ROOT)}/scorecard.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
