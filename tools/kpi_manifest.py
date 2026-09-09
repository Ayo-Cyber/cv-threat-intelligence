"""kpi_manifest.py — freeze the KPI test set, and say what is still missing.

    python tools/kpi_manifest.py status      # per-row counts vs the n floor
    python tools/kpi_manifest.py build       # freeze configs/eval/kpi_manifest_v1.json

The customer's MINIMUM KPI sheet is the acceptance frame; rows are only
signable as measured numbers on a frozen clip set (cvti/eval/kpi.py). `status`
is the shopping list: which rows have enough footage, which are SMOKE, and the
exact source to fill each gap. `build` freezes the set with a digest — the
scorecard refuses to run if the footage changes underneath the freeze.

WHERE THE FOOTAGE COMES FROM (the W6 sourcing runbook):
  theft        UCF-Crime Shoplifting + Stealing (kaggle mirror, see below)
               + CamNuvem test split (already on disk)
  suspicious   UCF-Crime Robbery/Burglary/Fighting/Assault + RWF-2000
  loitering /  VIRAT Ground 2.0 (event annotations!), PETS2007; near-term the
  intrusion    fetcher's staged-CCTV queries (tools/fetch_eval_clips.py)
  person       any of the above's normal footage WITH people, prefix person_
  normals      UCF Normal_* + our recorded live feeds + test_clips normal_*

  UCF categories we lack need ONE-TIME setup (2 min): a Kaggle account token
  at ~/.kaggle/kaggle.json, then
      kaggle datasets download minhajuddinmeraj/anomalydetectiondatasetucf \
             -p data/ucf_crime --unzip
  (The official 93 GB UCF zip does not fit this machine; the mirror carries
  the same categories trimmed.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.eval.kpi import (  # noqa: E402
    KPI_ROWS, MANIFEST_PATH, build_manifest, n_floor,
)

SOURCES = {
    "false_positive": "UCF Normal_* · recorded live feeds · fetcher normal_ queries",
    "person": "prefix person_ clips (fetcher) · VIRAT",
    "intrusion": "VIRAT event annotations · fetcher intrusion_ queries (staged)",
    "loitering": "VIRAT/PETS2007 · fetcher loitering_ queries (staged)",
    "theft": "UCF Shoplifting+Stealing (kaggle, header) · CamNuvem (on disk)",
    "suspicious": "UCF Robbery/Burglary (kaggle) · RWF-2000 (HF) · on-disk Fighting/Assault",
}


def status() -> int:
    doc = build_manifest()
    print("KPI MANIFEST STATUS — per-row n floor: the smallest n at which a "
          "PERFECT run can clear the row's target (misses push it higher)\n")
    print(f"{'SN':>3}  {'row':<16}{'have':>6}{'need':>6}   fill from")
    short = 0
    for row in KPI_ROWS:
        r = doc["rows"][row.key]
        have = r["negatives"] if row.metric == "fpr" else r["positives"]
        need = r["n_floor"]
        mark = "ok " if r["publishable"] else "LOW"
        if not r["publishable"]:
            short += 1
        print(f"{row.sn:>3}  {row.key:<16}{have:>6}{need:>6}   "
              f"[{mark}] {SOURCES.get(row.key, '')}")
    print(f"\ndigest if frozen now: {doc['digest']}")
    if short:
        print(f"{short} row(s) below the floor — their scorecard verdicts "
              "will read SMOKE until filled.")
    return 0


def build() -> int:
    doc = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(doc, indent=1))
    total = sum(r["positives"] + r["negatives"] for r in doc["rows"].values())
    print(f"frozen {MANIFEST_PATH.relative_to(ROOT)} — "
          f"{total} row-memberships, digest {doc['digest']}")
    print("Commit it: the digest is what makes every future scorecard "
          "comparable to this one.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("status", "build"))
    args = ap.parse_args()
    return status() if args.command == "status" else build()


if __name__ == "__main__":
    raise SystemExit(main())
