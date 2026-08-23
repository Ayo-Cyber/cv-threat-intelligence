"""Regenerate SENSITIVITY_MEASURED in gate.py FROM the archived eval runs.

    python tools/make_sensitivity.py           # rewrite the block in place
    python tools/make_sensitivity.py --check   # exit 1 if code and archives disagree

EP-07-T2: the constants used to be hand-maintained, so the code could drift
from the reality it claims to describe. Now the archives are the source of
truth and this file is a compiler. Clip-level scoring, matching how the
numbers were originally computed:

    a clip is FLAGGED when >=1 of its candidates was confirmed
    recall    = flagged threat clips / threat clips
    precision = flagged threat clips / all flagged clips
    fpr       = flagged normal clips / normal clips
    alerts    = total confirmed candidates across all clips
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "cvti" / "verification" / "gate.py"
EVAL = ROOT / "runs" / "eval"

# sensitivity level -> the archived run that measured it
RUNS = {"balanced": "v2-tightened", "strict": "v3-strict"}

BEGIN = "# --- BEGIN GENERATED: SENSITIVITY_MEASURED (tools/make_sensitivity.py) ---"
END = "# --- END GENERATED: SENSITIVITY_MEASURED ---"


def score_run(name: str) -> dict:
    clips = json.loads((EVAL / name / "metrics.json").read_text())["clips"]
    threat = [c for c in clips if c["is_threat"]]
    normal = [c for c in clips if not c["is_threat"]]
    flag = lambda c: c["confirmed"] > 0  # noqa: E731
    tp = sum(1 for c in threat if flag(c))
    fp = sum(1 for c in normal if flag(c))
    return {
        "recall": round(tp / len(threat), 3),
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "fpr": round(fp / len(normal), 3),
        "alerts": sum(c["confirmed"] for c in clips),
        "_clips": (len(threat), len(normal)),
    }


def render() -> str:
    scored = {level: score_run(run) for level, run in RUNS.items()}
    n_threat, n_normal = next(iter(scored.values()))["_clips"]
    lines = [BEGIN,
             "# Computed from the archived eval runs (runs/eval/v2-tightened, v3-strict),",
             "# not hand-maintained: regenerate with `python tools/make_sensitivity.py`",
             "# after any re-measurement. A test asserts these equal what the archives",
             "# produce, so the code cannot drift from the reality it claims to describe.",
             "SENSITIVITY_MEASURED = {"]
    for level, m in scored.items():
        lines.append(f'    "{level}": {{"recall": {m["recall"]}, "precision": {m["precision"]}, '
                     f'"fpr": {m["fpr"]}, "alerts": {m["alerts"]}}},')
    lines.append(f'    "_dataset": "{n_threat + n_normal} held-out CamNuvem test clips '
                 f'({n_threat} threat / {n_normal} normal), gemma3:4b",')
    lines.append("}")
    lines.append(END)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the committed block matches the archives; change nothing")
    args = ap.parse_args()

    missing = [run for run in RUNS.values() if not (EVAL / run / "metrics.json").exists()]
    if missing:
        print(f"archived runs not on this machine ({', '.join(missing)}) — nothing to do")
        return 0

    src = GATE.read_text()
    block = re.search(re.escape(BEGIN) + r".*?" + re.escape(END), src, re.S)
    if not block:
        print(f"::error::generated markers missing from {GATE}")
        return 2
    fresh = render()
    if block.group(0) == fresh:
        print("SENSITIVITY_MEASURED matches the archived runs.")
        return 0
    if args.check:
        print("::error::SENSITIVITY_MEASURED does not match the archives. "
              "Run: python tools/make_sensitivity.py")
        return 1
    GATE.write_text(src.replace(block.group(0), fresh))
    print(f"rewrote SENSITIVITY_MEASURED in {GATE} from the archives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
