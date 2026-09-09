"""detect_bench.py — the W2 verdict: does this box's accelerator earn its keep?

    python tools/detect_bench.py                    # torch-CPU vs onnx, b1+b4
    python tools/detect_bench.py --json
    python tools/detect_bench.py --frames 60

W2's acceptance is ">=2x detection throughput vs the CPU-torch build, on the
replica" — a number to be measured on the box that matters, not asserted.
This is the yardstick (the same discipline as tools/latency_baseline.py):
run it on the replica with onnxruntime-directml installed and the verdict
line at the bottom IS the committed number. On a dev Mac it reports the
CPU-vs-CPU comparison honestly and says why that is not the criterion.

Real footage, not noise: x264 and the detector both behave differently on
synthetic static. Batches of 1 and 4 because the engine detects on batches
and the 9 Sep spike saw ONNX dynamic-batch pay a CPU penalty at b4 — if that
shows up on DirectML, W2 buckets to fixed batch sizes, decided by this tool's
output rather than by argument.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLIP = "data/test_clips/normal_street_01.mp4"
TARGET = 2.0          # the W2 SLO: >= 2x vs CPU-torch


def _frames(n: int):
    import cv2
    cap = cv2.VideoCapture(str(ROOT / CLIP))
    out = []
    while len(out) < n:
        ok, img = cap.read()
        if not ok:
            break
        out.append(img)
    cap.release()
    if not out:
        raise SystemExit(f"no frames from {CLIP} — run from the repo root")
    return out


def _bench(model, frames, batch: int, device=None) -> float:
    """ms per frame, batched like the engine batches."""
    kw = {"device": device} if device else {}
    batches = [frames[i:i + batch] for i in range(0, len(frames) - batch + 1, batch)]
    model.predict(batches[0], imgsz=640, conf=0.25, verbose=False, **kw)   # warm
    t0 = time.perf_counter()
    done = 0
    for chunk in batches:
        model.predict(chunk, imgsz=640, conf=0.25, verbose=False, **kw)
        done += len(chunk)
    return (time.perf_counter() - t0) / done * 1000.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="torch-CPU vs ONNX detection bench")
    ap.add_argument("--weights", default="models/yolov8n.pt")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    from cvti.detector.accel import (best_available_provider, detector_health,
                                     load_detector)

    frames = _frames(args.frames)
    torch_model = YOLO(str(ROOT / args.weights))
    onnx_model, info = load_detector(args.weights, backend="onnx")
    if info["backend"] != "onnx":
        print(f"ONNX path unavailable: {info['reason']}", file=sys.stderr)
        return 2

    rows = {}
    for batch in (1, 4):
        t = _bench(torch_model, frames, batch, device="cpu")
        o = _bench(onnx_model, frames, batch)
        rows[f"b{batch}"] = {"torch_cpu_ms": round(t, 1), "onnx_ms": round(o, 1),
                             "speedup": round(t / o, 2)}
    health = detector_health(info, onnx_model)
    provider = health["provider"] or "CPUExecutionProvider"
    accelerated = provider != "CPUExecutionProvider"
    best = max(r["speedup"] for r in rows.values())
    verdict = {
        "provider": provider,
        "rows": rows,
        "best_speedup": best,
        "meets_2x": best >= TARGET,
        "criterion_measurable_here": accelerated,
    }

    if args.json:
        print(json.dumps(verdict, indent=2))
        return 0
    print(f"DETECTION BENCH — {Path(args.weights).name} · {len(frames)} real frames")
    print(f"  onnx provider: {provider}")
    for k, r in rows.items():
        print(f"  {k}: torch-CPU {r['torch_cpu_ms']:6.1f} ms/f   "
              f"onnx {r['onnx_ms']:6.1f} ms/f   speedup {r['speedup']:.2f}x")
    if accelerated:
        word = "MEETS" if verdict["meets_2x"] else "does NOT meet"
        print(f"  VERDICT: best {best:.2f}x — {word} the >=2x criterion "
              f"on {provider}")
    else:
        print(f"  VERDICT: CPU-vs-CPU only ({best:.2f}x) — the >=2x criterion "
              "is scored on a box with DirectML/OpenVINO (the replica), not here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
