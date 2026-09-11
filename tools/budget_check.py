"""budget_check.py — assert the v1.9 SLO budgets against their evidence.

    python tools/budget_check.py                  # check committed evidence
    python tools/budget_check.py --evidence DIR   # a soak/lane output dir
    python tools/budget_check.py --json

budgets.json is the acceptance backbone; this is its enforcer. Every budget
with an evidence file gets re-read from disk (never trusted from the
'measured' field — that is documentation, this is verification) and compared
to its target. Budgets whose venue hasn't produced evidence yet are reported
UNMEASURED — visible, never counted as passing.

Exit codes: 0 all measurable budgets hold · 1 any budget breached ·
2 evidence file named but unreadable (a lying pipeline is a failure too).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = ROOT / "budgets.json"


def _dig(doc, dotted: str):
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check(evidence_dir: Path | None = None) -> dict:
    spec = json.loads(BUDGETS.read_text())["budgets"]
    rows = []
    for name, b in spec.items():
        row = {"budget": name, "target": b["target"], "direction": b["direction"],
               "owner": b.get("owner", ""), "venue": b.get("venue", "")}
        path = b.get("evidence")
        if not path:
            row["status"] = "UNMEASURED"
            rows.append(row)
            continue
        candidates = [ROOT / path]
        if evidence_dir is not None:
            # a soak/lane dir may carry a fresher copy under the same basename
            candidates.insert(0, evidence_dir / Path(path).name)
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            row["status"] = "UNMEASURED"
            row["note"] = f"evidence not present: {path}"
            rows.append(row)
            continue
        try:
            value = _dig(json.loads(src.read_text()), b["evidence_key"])
        except Exception:  # noqa: BLE001 - unreadable evidence is its own failure
            row["status"] = "EVIDENCE_ERROR"
            row["note"] = f"could not read {src}"
            rows.append(row)
            continue
        if value is None:
            # The evidence file exists and is honest about not having this
            # number (a 5h soak whose fail-visible gate confirmed nothing has
            # no alert latency to report — null, not a lie). That is
            # UNMEASURED, never an error: the first green soak failed its
            # own budget step over exactly this.
            row["status"] = "UNMEASURED"
            row["note"] = f"{b['evidence_key']} is null in {Path(src).name}"
            rows.append(row)
            continue
        if not isinstance(value, (int, float)):
            row["status"] = "EVIDENCE_ERROR"
            row["note"] = f"{b.get('evidence_key')} is not a number in {src}"
            rows.append(row)
            continue
        row["measured"] = value
        row["source"] = str(src)
        ok = value <= b["target"] if b["direction"] == "max" else value >= b["target"]
        row["status"] = "OK" if ok else "BREACHED"
        rows.append(row)
    summary = {"rows": rows,
               "breached": [r["budget"] for r in rows if r["status"] == "BREACHED"],
               "errors": [r["budget"] for r in rows if r["status"] == "EVIDENCE_ERROR"],
               "unmeasured": [r["budget"] for r in rows if r["status"] == "UNMEASURED"]}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evidence", default="", help="directory of fresher evidence files")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    s = check(Path(args.evidence) if args.evidence else None)
    if args.json:
        print(json.dumps(s, indent=1))
    else:
        print(f"{'BUDGET':<34} {'target':>8} {'measured':>9}  status")
        for r in s["rows"]:
            m = r.get("measured")
            sign = "<=" if r["direction"] == "max" else ">="
            print(f"{r['budget']:<34} {sign}{r['target']:>6} "
                  f"{m if m is not None else '—':>9}  {r['status']}"
                  + (f"  ({r['note']})" if r.get("note") else ""))
        if s["unmeasured"]:
            print(f"\nunmeasured (venue named in budgets.json): "
                  f"{', '.join(s['unmeasured'])}")
    if s["errors"]:
        return 2
    return 1 if s["breached"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
