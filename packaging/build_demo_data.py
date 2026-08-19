"""Assemble packaging/demo_data/ — a self-contained demo the app can play back on
ANY Mac with no engine/Ollama/clips installed.

Contents:
  demo_data/events.db          real recorded alerts (evidence_dir rewritten relative)
  demo_data/events/<dir>/*.jpg  real evidence frames
  demo_data/clips/<cam>.mp4     small real CCTV clips for the live wall

Run from the repo root (needs runs/demo/events.db from a real run + CamNuvem clips):
    python packaging/build_demo_data.py
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DB = ROOT / "runs" / "demo" / "events.db"
SRC_EV = ROOT / "runs" / "demo" / "events"
DST = ROOT / "packaging" / "demo_data"
SITE = ROOT / "configs" / "site_demo_live.json"  # 5-camera live-wall clips


def main() -> None:
    if not SRC_DB.exists():
        raise SystemExit(f"missing {SRC_DB} — run the engine once (see docs/DEMO_RUNBOOK.md)")
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "clips").mkdir(parents=True)

    # real evidence frames + db (rewrite evidence_dir to paths relative to DST)
    shutil.copytree(SRC_EV, DST / "events")
    shutil.copy(SRC_DB, DST / "events.db")
    con = sqlite3.connect(DST / "events.db")
    for eid, ed in con.execute("SELECT id, evidence_dir FROM events").fetchall():
        if ed:
            rel = ed.replace("runs/demo/", "").replace(str(ROOT) + "/", "")
            con.execute("UPDATE events SET evidence_dir=? WHERE id=?", (rel, eid))
    con.commit()
    con.close()

    # live-wall clips = the exact cameras from the demo site, so playback footage
    # matches the alerts it produced.
    cams = json.loads(SITE.read_text()).get("cameras", [])
    for c in cams:
        src = ROOT / c["source"]
        if src.exists():
            shutil.copy(src, DST / "clips" / f"{c['id']}.mp4")

    total = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
    print(f"built {DST.relative_to(ROOT)}: {len(list((DST/'clips').iterdir()))} clips, "
          f"{len(list((DST/'events').iterdir()))} evidence dirs, {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
