#!/usr/bin/env python
"""Export the shipping detectors to ONNX — run by CI before the bundle builds.

    python scripts/export_onnx.py            # models/yolov8n{,-pose}.onnx
    python scripts/export_onnx.py --force

opset 17 on purpose: the newest the shipping onnxruntime loads without
"support is limited" caveats (the default opset 22 refuses to load, 9 Sep).
Dynamic batch on purpose: the engine detects on frame BATCHES. Idempotent:
an existing export younger than its .pt is kept, so local builds don't pay
the export twice. The .pt files still ship — the ONNX path falls back to
torch when anything is off, and a fallback that isn't in the bundle is a lie.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ("models/yolov8n.pt", "models/yolov8n-pose.pt")
OPSET = 17


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-export even when fresh")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()

    from ultralytics import YOLO
    failed = 0
    for pt in args.models:
        src = ROOT / pt
        if not src.exists():
            print(f"skip {pt}: not present on this machine")
            continue
        dst = src.with_suffix(".onnx")
        if dst.exists() and not args.force \
                and dst.stat().st_mtime >= src.stat().st_mtime:
            print(f"keep {dst.name}: already newer than {src.name}")
            continue
        try:
            out = YOLO(str(src)).export(format="onnx", imgsz=640, dynamic=True,
                                        opset=OPSET, verbose=False)
            print(f"exported {out}")
        except Exception as exc:  # noqa: BLE001 - name every failure, export the rest
            failed += 1
            print(f"FAILED {pt}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
