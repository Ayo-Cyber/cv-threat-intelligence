#!/usr/bin/env python3
"""Probe a video clip with a pretrained VideoMAE action classifier.

This is an evaluation tool, not part of the live detector yet. It tells us
whether a pretrained video model produces useful temporal labels on our clips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from video_action_model import (  # noqa: E402
    DEFAULT_FRAME_COUNT,
    DEFAULT_VIDEOMAE_MODEL,
    DEFAULT_X3D_MODEL,
    MissingVideoActionDependency,
    VideoMAEActionModel,
    X3DActionModel,
    build_centered_window,
    build_segment_windows,
    read_video_frames,
    sample_evenly_with_indices,
)
from video_action_hybrid import predictions_to_events  # noqa: E402
from customization import CustomizationEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video action classification on a video clip.")
    parser.add_argument("video", help="Path to a local .mp4/.mov/.avi clip.")
    parser.add_argument("--backend", choices=("videomae", "x3d"), default="videomae", help="Video model backend.")
    parser.add_argument("--model", default="", help="Model id/name. Defaults depend on backend.")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_COUNT, help="Number of frames sent to VideoMAE.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of labels to print.")
    parser.add_argument(
        "--window-mode",
        choices=("single", "segments", "event"),
        default="single",
        help="single = one sample; segments = beginning/middle/ending; event = centered around --center-frame.",
    )
    parser.add_argument("--center-frame", type=int, default=-1, help="Event-center frame for --window-mode event.")
    parser.add_argument("--window-seconds", type=float, default=4.0, help="Total event window length in seconds.")
    parser.add_argument(
        "--max-decode-frames",
        type=int,
        default=240,
        help="Maximum decoded frames before even sampling. Use 0 for all frames.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Decode every Nth frame.")
    parser.add_argument("--device", default=None, help="Force device: cpu, mps, or cuda. Default auto-selects.")
    parser.add_argument("--json-out", default="", help="Optional path to save predictions as JSON.")
    parser.add_argument("--save-frames-dir", default="", help="Optional directory to save the exact sampled frames.")
    parser.add_argument("--hybrid-events", action="store_true", help="Map video model labels into weak RawEvents.")
    parser.add_argument("--config", default="", help="Optional user config to evaluate hybrid RawEvents.")
    parser.add_argument("--verbose", action="store_true", help="Print loading/progress messages.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_decode_frames = None if args.max_decode_frames == 0 else args.max_decode_frames

    model_name = args.model or (DEFAULT_VIDEOMAE_MODEL if args.backend == "videomae" else DEFAULT_X3D_MODEL)
    model = build_model(args.backend, model_name, device=args.device, frame_count=args.frames, verbose=args.verbose)

    started = time.time()
    try:
        frames = read_video_frames(args.video, max_frames=max_decode_frames, stride=args.stride)
        fps = read_video_fps(args.video)
        windows = build_windows(
            frames,
            args.window_mode,
            count=args.frames,
            center_frame=args.center_frame,
            window_seconds=args.window_seconds,
            fps=fps,
        )
        window_results = []
        for window in windows:
            if args.save_frames_dir:
                save_sampled_frames(window.sampled, Path(args.save_frames_dir) / window.name)
            predictions = model.predict_frames([item.frame for item in window.sampled], top_k=args.top_k)
            hybrid_events = (
                predictions_to_events(
                    predictions,
                    backend=args.backend,
                    model_name=model_name,
                    window_name=window.name,
                    sampled_frame_indices=[item.index for item in window.sampled],
                )
                if args.hybrid_events else []
            )
            candidate_alerts = []
            if args.config and hybrid_events:
                engine = CustomizationEngine(args.config)
                candidate_alerts = engine.evaluate(hybrid_events)
            window_results.append({
                **window.to_dict(),
                "predictions": [prediction.to_dict() for prediction in predictions],
                "hybrid_events": [event_to_dict(event) for event in hybrid_events],
                "candidate_alerts": [alert.to_dict() for alert in candidate_alerts],
            })
    except MissingVideoActionDependency as exc:
        print(str(exc), file=sys.stderr)
        return 2

    elapsed = time.time() - started
    payload = {
        "video": args.video,
        "backend": args.backend,
        "model": model_name,
        "window_mode": args.window_mode,
        "frames": args.frames,
        "top_k": args.top_k,
        "latency_seconds": round(elapsed, 3),
        "windows": window_results,
    }

    print(f"\nVideo action probe: {args.video}")
    print(f"Backend: {args.backend} | Model: {model_name}")
    print(f"Window mode: {args.window_mode}")
    print(f"Frames sent: {args.frames} | latency: {elapsed:.2f}s\n")
    for result in window_results:
        print(f"[{result['name']}] source frames {result['start_index']}..{result['end_index']}")
        print(f"  sampled: {result['sampled_frame_indices']}")
        for prediction in result["predictions"]:
            print(f"  {prediction['rank']:>2}. {prediction['label']:<36} {prediction['confidence']:.3f}")
        for event in result.get("hybrid_events", []):
            print(
                f"  event: {event['extra']['signal_type']} "
                f"raw={event['extra']['raw_confidence']:.3f} "
                f"adjusted={event['extra']['adjusted_confidence']:.3f}"
            )
        for alert in result.get("candidate_alerts", []):
            print(f"  config alert: {alert['rule_name']} ({alert['priority']})")
        print()

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON: {out_path}")

    return 0


def build_model(backend: str, model_name: str, *, device: str | None, frame_count: int, verbose: bool):
    if backend == "videomae":
        return VideoMAEActionModel(model_name, device=device, frame_count=frame_count, verbose=verbose)
    if backend == "x3d":
        return X3DActionModel(model_name, device=device, frame_count=frame_count, verbose=verbose)
    raise ValueError(f"Unsupported backend: {backend}")


def build_windows(
    frames: list,
    mode: str,
    *,
    count: int,
    center_frame: int,
    window_seconds: float,
    fps: float,
):
    if mode == "single":
        sampled = sample_evenly_with_indices(frames, count=count)
        return [SingleWindow(sampled)]
    if mode == "segments":
        return build_segment_windows(frames, count=count)
    if mode == "event":
        if center_frame < 0:
            raise ValueError("--window-mode event requires --center-frame")
        radius_frames = max(0, int(round((window_seconds * fps) / 2)))
        return [build_centered_window(frames, center_index=center_frame, radius_frames=radius_frames, count=count)]
    raise ValueError(f"Unsupported window mode: {mode}")


class SingleWindow:
    name = "single"
    start_index = 0

    def __init__(self, sampled: list) -> None:
        self.sampled = sampled
        self.end_index = sampled[-1].index if sampled else 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "sampled_frame_indices": [item.index for item in self.sampled],
        }


def save_sampled_frames(sampled: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sample_*.jpg"):
        old.unlink()
    for rank, item in enumerate(sampled):
        # read_video_frames returns RGB; cv2.imwrite expects BGR.
        bgr = cv2.cvtColor(item.frame, cv2.COLOR_RGB2BGR)
        out_path = out_dir / f"sample_{rank:02d}_src_{item.index:05d}.jpg"
        cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def read_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()
    return fps


def event_to_dict(event) -> dict:
    return {
        "detector": event.detector,
        "active": event.active,
        "title": event.title,
        "level": event.level,
        "state": event.state,
        "person_id": event.person_id,
        "object_label": event.object_label,
        "timestamp": event.timestamp,
        "extra": event.extra,
    }


if __name__ == "__main__":
    raise SystemExit(main())
