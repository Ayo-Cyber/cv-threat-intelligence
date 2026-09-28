"""Build the KPI 1 & 2 site the way the desktop app builds it: through the API.

KPI 1 (person entering / exiting) needs a zone; KPI 2 (vehicle entering /
exiting) needs the drawn gate line. Both are set here the way the app sets
them -- first owner, sign in, POST /cameras, POST /cameras/{id}/zones in
original pixels (what the rectangle tool sends), POST
/cameras/{id}/vehicle-line in fractions with `flip` (what the line editor
sends) -- so the CI run proves the product's own configuration path, not a
hand-written JSON that happens to work.

    python tests/e2e/kpi_site.py --clips <dir with the kpi-clips-v1 files> --out <dir>

Writes <out>/site.json (notify: console; the runner passes --notify itself,
so no token ever lands in a file that could be uploaded as an artifact).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Run as a script, sys.path[0] is tests/e2e -- and `import cvti` would then
# find whatever copy pip installed, which on a developer box was a stale
# checkout with no vehicle-line route (404, 28 Sep). This checkout, always.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# One person clip and two vehicle clips: enough for all four alert kinds
# (person entered/exited, vehicle entered/exited) without flooding a chat.
# Lines are where the gate sits in each clip; `flip` is which way is in.
CAMERAS = [
    {"id": "kpi1_person", "clip": "intrusion_23.mp4", "kind": "person",
     "zone_from_x": 0.54},
    {"id": "kpi2_vehicle_a", "clip": "-KVYTHtljA0.mp4", "kind": "vehicle",
     "line_x": 0.42, "flip": True},
    {"id": "kpi2_vehicle_b", "clip": "7jITHZShdX8.mp4", "kind": "vehicle",
     "line_x": 0.50, "flip": False},
]


def probe(path: Path) -> tuple[int, int, float]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, fps)
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise SystemExit(f"could not read {path}")
    return w, h, dur


def build(clips: Path, out: Path, cameras: list[dict] = CAMERAS) -> dict:
    from fastapi.testclient import TestClient
    from cvti.api.app import create_app

    out.mkdir(parents=True, exist_ok=True)
    (out / "rules_person.json").write_text(json.dumps({
        "use_case_id": "kpi1",
        "rules": [
            {"name": "person_entered", "trigger": {"detector": "zone_entry"}, "priority": "high"},
            {"name": "person_exited", "trigger": {"detector": "zone_exit"}, "priority": "medium"},
        ]}, indent=2))
    # No shipped preset carries vehicle rules: the API adds vehicle_entered /
    # vehicle_exited to the camera's own config the moment a line exists.
    (out / "rules_none.json").write_text(json.dumps({"use_case_id": "kpi2", "rules": []}, indent=2))
    site = out / "site.json"
    site.write_text(json.dumps({"name": "KPI 1 & 2 (CI)", "configured": True,
                                "notify": "console", "cameras": []}, indent=2))
    client = TestClient(create_app(db_path=str(out / "runs" / "events.db"), site_path=str(site)))
    r = client.post("/api/v1/auth/first-owner", json={"username": "ci", "password": "ci-pw-123456"})
    assert r.status_code in (200, 201), r.text
    r = client.post("/api/v1/auth/session", json={"username": "ci", "password": "ci-pw-123456"})
    assert r.status_code == 200, r.text
    hdr = {"Authorization": f"Bearer {r.json()['token']}"}

    summary: dict = {"cameras": [], "longest_s": 0.0}
    for c in cameras:
        src = clips / c["clip"]
        if not src.exists():
            raise SystemExit(f"missing clip {src}")
        w, h, dur = probe(src)
        summary["longest_s"] = max(summary["longest_s"], dur)
        cfg = out / ("rules_person.json" if c["kind"] == "person" else "rules_none.json")
        r = client.post("/api/v1/cameras", headers=hdr,
                        json={"camera": {"id": c["id"], "source": str(src), "config": str(cfg)}})
        assert r.status_code in (200, 201), f"{c['id']} add_camera: {r.status_code} {r.text}"
        if c["kind"] == "person":
            x0 = int(c["zone_from_x"] * w)
            r = client.post(f"/api/v1/cameras/{c['id']}/zones", headers=hdr,
                            json={"name": "monitored area", "dwell_seconds": 5,
                                  "points": [[x0, 0], [w - 1, 0], [w - 1, h - 1], [x0, h - 1]]})
        else:
            r = client.post(f"/api/v1/cameras/{c['id']}/vehicle-line", headers=hdr,
                            json={"start": [c["line_x"], 0.05], "end": [c["line_x"], 0.95],
                                  "flip": c["flip"], "name": "gate"})
        assert r.status_code in (200, 201), f"{c['id']} {c['kind']} configuration: {r.status_code} {r.text}"
        summary["cameras"].append({"id": c["id"], "kind": c["kind"], "size": f"{w}x{h}",
                                   "seconds": round(dur)})

    # What the engine will actually read: the site file the API wrote.
    written = json.loads(site.read_text())
    for cam in written["cameras"]:
        rules = json.loads(Path(cam["config"]).read_text())["rules"]
        names = [x["name"] for x in rules]
        entry = next(s for s in summary["cameras"] if s["id"] == cam["id"])
        entry["rules"] = names
        entry["vehicle_line"] = cam.get("vehicle_line")
        entry["zones"] = bool(cam.get("zones"))
        want = ({"vehicle_entered", "vehicle_exited"} if entry["kind"] == "vehicle"
                else {"person_entered", "person_exited"})
        missing = want - set(names)
        if missing:
            raise SystemExit(f"{cam['id']}: rules missing after configuration: {sorted(missing)}")
    summary["site"] = str(site)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--clips", required=True, help="directory holding the kpi-clips-v1 files")
    p.add_argument("--out", required=True, help="where site.json, rules and runs go")
    a = p.parse_args(argv)
    summary = build(Path(a.clips).resolve(), Path(a.out).resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
