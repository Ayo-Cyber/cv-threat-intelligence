"""Separate this run's candidate outcomes from historical incident rows."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("evidence", type=Path)
args = parser.parse_args()
start = json.loads((args.evidence / "run-start.json").read_text())["started_at"]
health = json.loads((args.evidence / "gate-health.json").read_text())
end = health["generated_at"]
site = json.loads((args.evidence / "site-config.json").read_text())
report = {"start": start, "snapshot_end": end, "gate_snapshot": health["gate"], "cameras": []}
with sqlite3.connect("file:runs/chi_validation/desktop/events.db?mode=ro", uri=True) as db:
    db.row_factory = sqlite3.Row
    for camera in site["cameras"]:
        rows = db.execute("SELECT * FROM concealment_audit WHERE camera_id=? AND generated_at BETWEEN ? AND ?",
                          (camera["id"], start, end)).fetchall()
        outcomes = Counter((r["verdict"] if r["verdict_at"] and r["verdict_at"] <= end
                            else r["admission_status"]) or "unknown" for r in rows)
        events = db.execute("SELECT id,reason,unverified FROM events WHERE camera_id=? AND ts BETWEEN ? AND ?",
                            (camera["id"], start, end)).fetchall()
        report["cameras"].append({"camera": camera["id"], "source": camera["source"],
                                  "candidate_rows": len(rows), "outcomes": dict(outcomes),
                                  "incidents": [dict(e) for e in events]})
(args.evidence / "checkpoint-summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
