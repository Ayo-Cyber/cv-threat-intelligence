#!/usr/bin/env python3
"""Phase 8.0 throughput benchmark — sizes the multi-stream edge pipeline.

Measures per-model inference latency (detection, pose, optional VideoMAE) at
several batch sizes on whatever device is available (cuda / mps / cpu), then
projects how many camera streams one box can sustain at a target per-camera FPS.

The point: you cannot design batch windows or the VLM funnel without real
numbers on the *target* GPU. Run this on the edge box, not just the laptop.

Examples
--------
    # Detection + pose, batch sweep, synthetic frames
    python tools/throughput_bench.py

    # Real GPU sizing on the edge box
    python tools/throughput_bench.py --weights models/yolov8n.pt \
        --pose-weights models/yolov8n-pose.pt --batch-sizes 1 4 8 16 \
        --imgsz 640 --iters 50 --half --target-fps 5

    # Include the VideoMAE cost (needs requirements-video.txt)
    python tools/throughput_bench.py --video-action-model MCG-NJU/videomae-base
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _detect_device() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _synthetic_frames(n: int, h: int, w: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _time_calls(fn, iters: int) -> dict:
    """Run fn() `iters` times after 3 warmups; return timing stats in ms."""
    for _ in range(3):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "mean_ms": statistics.mean(samples),
        "p50_ms": samples[len(samples) // 2],
        "p95_ms": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
        "min_ms": samples[0],
    }


def _bench_yolo(label: str, weights: str, batch_sizes: list[int], imgsz: int,
                iters: int, device: str, half: bool) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(weights)
    use_half = half and device == "cuda"
    rows = []
    print(f"\n=== {label}: {weights}  (device={device}, imgsz={imgsz}, half={use_half}) ===")
    print(f"{'batch':>6} {'ms/batch':>10} {'ms/frame':>10} {'frames/s':>10} {'p95 ms':>9}")
    for b in batch_sizes:
        frames = _synthetic_frames(b, imgsz, imgsz)

        def _call():
            model.predict(frames, imgsz=imgsz, device=device, half=use_half,
                          verbose=False)

        try:
            stats = _time_calls(_call, iters)
        except Exception as exc:  # noqa: BLE001
            print(f"{b:>6}   FAILED: {str(exc)[:70]}")
            continue
        ms_batch = stats["mean_ms"]
        ms_frame = ms_batch / b
        fps = 1000.0 / ms_frame
        rows.append({"model": label, "batch": b, "ms_frame": ms_frame, "fps": fps,
                     "p95_ms": stats["p95_ms"]})
        print(f"{b:>6} {ms_batch:>10.1f} {ms_frame:>10.2f} {fps:>10.0f} {stats['p95_ms']:>9.1f}")
    return rows


def _bench_videomae(model_name: str, iters: int, device: str, frame_count: int) -> list[dict]:
    try:
        from cvti.video_action_model import VideoMAEActionModel
    except Exception as exc:  # noqa: BLE001
        print(f"\n=== VideoMAE: SKIPPED ({str(exc)[:80]}) ===")
        return []
    print(f"\n=== VideoMAE: {model_name}  (device={device}, {frame_count} frames/clip) ===")
    try:
        model = VideoMAEActionModel(model_name, device=None if device == "cpu" else device,
                                    frame_count=frame_count, verbose=False)
    except Exception as exc:  # noqa: BLE001
        print(f"    load FAILED: {str(exc)[:90]}")
        return []
    clip = _synthetic_frames(frame_count, 224, 224)

    def _call():
        model.predict_frames(clip, top_k=5)

    try:
        stats = _time_calls(_call, max(5, iters // 5))
    except Exception as exc:  # noqa: BLE001
        print(f"    infer FAILED: {str(exc)[:90]}")
        return []
    per_call = stats["mean_ms"]
    print(f"    {per_call:.0f} ms / clip   (p95 {stats['p95_ms']:.0f} ms)   "
          f"~{1000.0 / per_call:.1f} clips/s")
    return [{"model": "videomae", "batch": 1, "ms_frame": per_call, "fps": 1000.0 / per_call,
             "p95_ms": stats["p95_ms"]}]


def _bench_gate(provider: str, model: str, base_url: str, frames: int, imgsz: int,
                iters: int) -> None:
    """Time the VLM gate — the true scaling ceiling. Needs a live provider
    (e.g. a local Ollama server); reports gracefully if it can't reach one."""
    print(f"\n=== VLM gate: provider={provider} model={model or '(default)'} "
          f"frames={frames} ===")
    try:
        from cvti.verification.gate import VerificationGate
        from cvti.contracts import CandidateAlert
    except Exception as exc:  # noqa: BLE001
        print(f"    SKIPPED (import: {str(exc)[:80]})")
        return
    gate = VerificationGate(provider=provider, model=model, base_url=base_url)
    alert = CandidateAlert(rule_name="violence", priority="critical", detector="pose",
                           title="VIOLENCE SUSPECTED", person_id=1, object_label=None,
                           timestamp=0.0)
    clip = _synthetic_frames(frames, imgsz, imgsz)
    scene = {"environment_type": "retail_shop", "scene_description": "aisle"}

    # One real call first to surface auth/connection errors clearly.
    try:
        gate.verify(clip if frames > 1 else clip[0], alert, scene)
    except Exception as exc:  # noqa: BLE001
        print(f"    could not reach provider: {str(exc)[:100]}")
        print("    (start the Ollama server / set the API key, then rerun with --time-gate)")
        return

    def _call():
        gate.verify(clip if frames > 1 else clip[0], alert, scene)

    stats = _time_calls(_call, max(4, iters // 4))
    per_call = stats["mean_ms"]
    per_s = 1000.0 / per_call
    print(f"    {per_call:.0f} ms/verify (p95 {stats['p95_ms']:.0f})  ~{per_s:.2f} verifies/s")
    print(f"    => the whole box can confirm ~{per_s:.2f} alerts/s across ALL cameras.")
    print("    Dedup/throttle (alert queue) must keep candidates under this ceiling.")


def _project(rows: list[dict], target_fps: float, cams: int) -> None:
    print(f"\n=== Multi-stream projection (target {target_fps:g} FPS/camera) ===")
    print(f"{'model':>12} {'best fps':>9} {'max cams @target':>17} {'used @'+str(cams)+'cams':>14}")
    for label in dict.fromkeys(r["model"] for r in rows):
        best = max((r for r in rows if r["model"] == label), key=lambda r: r["fps"])
        max_cams = best["fps"] / target_fps
        needed = cams * target_fps
        util = needed / best["fps"] * 100
        print(f"{label:>12} {best['fps']:>9.0f} {max_cams:>17.1f} {util:>13.0f}%")
    print("\nNote: detection runs on every sampled frame; pose runs on person-frames only.")
    print("The VLM gate is the true ceiling — measure it separately (Ollama server).")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 8.0 multi-stream throughput benchmark.")
    p.add_argument("--weights", default="models/yolov8n.pt")
    p.add_argument("--pose-weights", default="models/yolov8n-pose.pt")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--half", action="store_true", help="fp16 (CUDA only; ignored elsewhere).")
    p.add_argument("--device", default="", help="Force cuda/mps/cpu; blank = auto-detect.")
    p.add_argument("--target-fps", type=float, default=5.0, help="Per-camera detection FPS budget.")
    p.add_argument("--cams", type=int, default=16, help="Camera count to report utilization for.")
    p.add_argument("--video-action-model", default="", help="Also benchmark this VideoMAE checkpoint.")
    p.add_argument("--video-action-frames", type=int, default=16)
    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--time-gate", action="store_true",
                   help="Also time the VLM gate (the true ceiling). Needs a live provider.")
    p.add_argument("--gate-provider", default="ollama")
    p.add_argument("--gate-model", default="")
    p.add_argument("--gate-base-url", default="")
    p.add_argument("--gate-frames", type=int, default=3)
    args = p.parse_args()

    device = args.device or _detect_device()
    print(f"Device: {device}   |  batch sizes: {args.batch_sizes}  |  iters: {args.iters}")

    rows: list[dict] = []
    rows += _bench_yolo("detection", args.weights, args.batch_sizes, args.imgsz,
                        args.iters, device, args.half)
    if not args.no_pose:
        rows += _bench_yolo("pose", args.pose_weights, args.batch_sizes, args.imgsz,
                            args.iters, device, args.half)
    if args.video_action_model:
        rows += _bench_videomae(args.video_action_model, args.iters, device,
                                args.video_action_frames)

    if args.time_gate:
        _bench_gate(args.gate_provider, args.gate_model, args.gate_base_url,
                    args.gate_frames, args.imgsz, args.iters)

    if rows:
        _project(rows, args.target_fps, args.cams)


if __name__ == "__main__":
    main()
