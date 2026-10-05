"""Prepare a real CAVIAR movement test without changing production settings."""
import json
from pathlib import Path

root = Path("runs/chi_validation/scenarios/05_multiple_people_moving")
site = json.loads((root / "negative-site.json").read_text())
site["name"] = "KPI 5 real CAVIAR four-person movement (not stationary)"
camera = site["cameras"][1]
camera["source"] = "runs/chi_validation/caviar/Meet_Crowd.mpg"
camera["detection_inference"] = dict(site["cameras"][0]["detection_inference"])
(root / "real-crowd-site.json").write_text(json.dumps(site, indent=2) + "\n")
