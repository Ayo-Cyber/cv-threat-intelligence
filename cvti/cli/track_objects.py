"""Offline/local general-object tracking for a video file or webcam."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class RuntimeDependencies:
    """Heavy runtime dependencies, injectable so CLI tests need no camera/model."""

    cv2: Any
    yolo_factory: Callable[[str], Any]
    detections_from_ultralytics: Callable[[Any], Any]
    tracker_factory: Callable[..., Any]
    monotonic: Callable[[], float] = time.monotonic


class TrackingError(RuntimeError):
    """An actionable local tracking failure."""


TIMING_WINDOW_SIZE = 2048


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track general COCO objects in a local video or webcam without downloads."
    )
    parser.add_argument("--source", required=True, help="Existing video path or webcam index (0)")
    parser.add_argument("--weights", required=True, help="Existing local YOLO .pt weights file")
    parser.add_argument("--device", required=True, choices=("mps", "cpu"))
    parser.add_argument("--output-dir", required=True, help="New directory for JSONL/summary output")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum sampled/processed frames (not source frames read)",
    )
    parser.add_argument(
        "--every-n-frames", type=int, default=1, help="Process one of every N source frames"
    )
    parser.add_argument("--save-video", action="store_true", help="Write annotated.mp4")
    parser.add_argument("--show", action="store_true", help="Show a local preview; Q stops")
    return parser


def _validated_args(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]]) -> argparse.Namespace:
    args = parser.parse_args(argv)
    weights = Path(args.weights).expanduser()
    if not weights.is_file():
        parser.error(f"--weights must be an existing file (automatic downloads are disabled): {weights}")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")
    if args.every_n_frames <= 0:
        parser.error("--every-n-frames must be a positive integer")

    source_text = str(args.source)
    if source_text.isdecimal():
        args.source_value = int(source_text)
        args.source_kind = "webcam"
    else:
        source = Path(source_text).expanduser()
        if not source.is_file():
            parser.error(f"--source must be an existing video file or webcam index: {source}")
        args.source_value = str(source)
        args.source_kind = "video"

    output = Path(args.output_dir).expanduser()
    if output.exists():
        parser.error(f"--output-dir already exists; choose a new directory: {output}")
    args.weights_path = weights.resolve()
    args.output_path = output.resolve()
    return args


def _load_dependencies() -> RuntimeDependencies:
    # Keep imports out of argument validation: a missing file must never initialize
    # Ultralytics (which may otherwise interpret a model name as downloadable).
    try:
        import cv2
        import supervision as sv
        from ultralytics import YOLO

        from cvti.detector.object_tracks import GeneralObjectTracker
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise TrackingError(f"required local runtime dependency is unavailable: {exc}") from exc
    return RuntimeDependencies(cv2, YOLO, sv.Detections.from_ultralytics, GeneralObjectTracker)


def _positive_fps(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _names(model: Any) -> Mapping[int, str]:
    raw = getattr(model, "names", None)
    if isinstance(raw, Mapping):
        names = {int(key): str(value) for key, value in raw.items()}
    elif isinstance(raw, (list, tuple)):
        names = {index: str(value) for index, value in enumerate(raw)}
    else:
        raise TrackingError("YOLO model did not provide a class-name mapping")
    if not names or any(not value for value in names.values()):
        raise TrackingError("YOLO model provided an invalid class-name mapping")
    return names


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _timings(values: Sequence[float], total: int) -> dict[str, Any]:
    return {
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "sample_count": len(values),
        "observations_total": total,
        "window": "most_recent",
        "window_capacity": TIMING_WINDOW_SIZE,
    }


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _weight_identifier(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _annotate(frame: Any, overlays: Sequence[dict[str, Any]], cv2: Any) -> Any:
    annotated = frame.copy()
    for overlay in overlays:
        x1, y1, x2, y2 = overlay["bbox"]
        colour = tuple(int(value) for value in overlay.get("colour", (255, 180, 0)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            annotated,
            str(overlay["label"]),
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
        )
    return annotated


def _public_counter_max(snapshot: Mapping[str, Any], prefix: str, current: int) -> int:
    """Count sequential wrapper identities without retaining historical IDs."""
    maximum = current
    for track in snapshot.get("tracks", ()):
        public_id = track.get("id") if isinstance(track, Mapping) else None
        if not isinstance(public_id, str) or not public_id.startswith(prefix):
            continue
        try:
            counter = int(public_id[len(prefix):])
        except ValueError:
            continue
        if counter > maximum:
            maximum = counter
    return maximum


def _media_timestamp(
    capture: Any,
    cv2: Any,
    frame_index: int,
    timestamp_fps: float,
    last_timestamp: Optional[float],
) -> tuple[float, bool]:
    """Use monotonic container media time, otherwise deterministic frame time."""
    try:
        candidate = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    except (AttributeError, TypeError, ValueError):
        candidate = float("nan")
    if math.isfinite(candidate) and candidate >= 0.0 and (
        last_timestamp is None or candidate > last_timestamp
    ):
        return candidate, False
    fallback = frame_index / timestamp_fps
    if last_timestamp is not None and fallback <= last_timestamp:
        fallback = last_timestamp + (1.0 / timestamp_fps)
    return fallback, True


def run(args: argparse.Namespace, dependencies: Optional[RuntimeDependencies] = None) -> dict[str, Any]:
    deps = dependencies or _load_dependencies()
    cv2 = deps.cv2
    capture = None
    writer = None
    output_dir: Path = args.output_path

    try:
        model = deps.yolo_factory(str(args.weights_path))
        names = _names(model)
        capture = cv2.VideoCapture(args.source_value)
        if capture is None or not capture.isOpened():
            raise TrackingError(f"cannot open {args.source_kind} source: {args.source}")

        source_fps = _positive_fps(capture.get(cv2.CAP_PROP_FPS))
        timestamp_fps = source_fps if source_fps is not None else 30.0
        effective_fps = timestamp_fps / args.every_n_frames
        session_id = uuid.uuid4().hex
        tracker = deps.tracker_factory(
            camera_id="webcam" if args.source_kind == "webcam" else "local-video",
            session_id=session_id,
            names=names,
            expected_fps=effective_fps,
        )

        output_dir.mkdir(parents=True, exist_ok=False)
        snapshots_path = output_dir / "snapshots.jsonl"
        summary_path = output_dir / "summary.json"
        video_path = output_dir / "annotated.mp4"

        source_frames = processed = skipped = 0
        inference_ms: deque[float] = deque(maxlen=TIMING_WINDOW_SIZE)
        tracking_ms: deque[float] = deque(maxlen=TIMING_WINDOW_SIZE)
        timing_observations = 0
        tracks_created = 0
        width = height = 0
        stop_reason = "eof"
        outcome = "complete"
        pending_error: Optional[BaseException] = None
        error_metadata: Optional[dict[str, str]] = None
        video_created = False
        media_timestamp_frames = 0
        fallback_timestamp_frames = 0
        started = deps.monotonic()
        last_timestamp: Optional[float] = None
        prefix = f"{'webcam' if args.source_kind == 'webcam' else 'local-video'}/{session_id}/"

        with snapshots_path.open("x", encoding="utf-8") as snapshots:
            try:
                while args.max_frames is None or processed < args.max_frames:
                    try:
                        ok, frame = capture.read()
                    except Exception as exc:
                        stop_reason = "capture_failure"
                        raise TrackingError(
                            f"{args.source_kind} capture failed while reading a frame: {exc}"
                        ) from exc
                    if not ok:
                        if args.source_kind == "webcam":
                            stop_reason = "capture_failure"
                            raise TrackingError("webcam capture failed while reading a frame")
                        if source_frames == 0:
                            stop_reason = "capture_failure"
                            raise TrackingError("video source opened but returned zero frames")
                        stop_reason = "eof"
                        break
                    frame_index = source_frames
                    source_frames += 1
                    if frame_index % args.every_n_frames:
                        skipped += 1
                        continue

                    height, width = (int(value) for value in frame.shape[:2])
                    if args.source_kind == "video":
                        timestamp, used_fallback = _media_timestamp(
                            capture, cv2, frame_index, timestamp_fps, last_timestamp
                        )
                        fallback_timestamp_frames += int(used_fallback)
                        media_timestamp_frames += int(not used_fallback)
                    else:
                        timestamp = max(0.0, deps.monotonic() - started)
                        if last_timestamp is not None:
                            timestamp = max(last_timestamp, timestamp)
                    last_timestamp = timestamp

                    before = deps.monotonic()
                    result = model.predict(source=frame, device=args.device, verbose=False)[0]
                    detections = deps.detections_from_ultralytics(result)
                    after_inference = deps.monotonic()
                    tracker.update(detections, timestamp)
                    after_tracking = deps.monotonic()
                    inference_ms.append((after_inference - before) * 1000.0)
                    tracking_ms.append((after_tracking - after_inference) * 1000.0)
                    timing_observations += 1
                    processed += 1

                    snapshot = tracker.snapshot(timestamp)
                    tracks_created = _public_counter_max(snapshot, prefix, tracks_created)
                    snapshot.update({"event": "frame", "source_frame_index": frame_index})
                    snapshots.write(json.dumps(snapshot, sort_keys=True) + "\n")
                    snapshots.flush()

                    if args.save_video or args.show:
                        annotated = _annotate(frame, tracker.overlays(timestamp), cv2)
                        if args.save_video and writer is None:
                            writer = cv2.VideoWriter(
                                str(video_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                effective_fps,
                                (width, height),
                            )
                            if not writer.isOpened():
                                raise TrackingError(f"cannot create annotated video: {video_path}")
                            video_created = True
                        if writer is not None:
                            writer.write(annotated)
                        if args.show:
                            cv2.imshow("CVTI General Object Tracking", annotated)
                            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                                stop_reason = "user_quit"
                                break
                    del frame, result, detections
                else:
                    stop_reason = "max_frames"
            except BaseException as exc:
                pending_error = exc
                if isinstance(exc, KeyboardInterrupt):
                    outcome = "interrupted"
                    stop_reason = "interrupted"
                else:
                    outcome = "error"
                    if stop_reason not in ("capture_failure",):
                        stop_reason = "error"
                error_metadata = {"type": type(exc).__name__, "message": str(exc)}

            final_timestamp = 0.0 if last_timestamp is None else last_timestamp
            final_snapshot = tracker.snapshot(final_timestamp)
            final_snapshot.update(
                {
                    "event": "source_end",
                    "source_end": {"reason": stop_reason, "outcome": outcome},
                    "source_frame_index": source_frames - 1 if source_frames else None,
                }
            )
            snapshots.write(json.dumps(final_snapshot, sort_keys=True) + "\n")
            snapshots.flush()

        wall_seconds = max(0.0, deps.monotonic() - started)
        summary = {
            "schema_version": 1,
            "outcome": outcome,
            "error": error_metadata,
            "source": {"kind": args.source_kind, "value": str(args.source)},
            "source_end": stop_reason,
            "counts": {
                "source_frames_read": source_frames,
                "processed_frames": processed,
                "skipped_frames": skipped,
                "tracks_created": tracks_created,
            },
            "timing_ms": {
                "inference": _timings(inference_ms, timing_observations),
                "tracking": _timings(tracking_ms, timing_observations),
            },
            "wall_seconds": round(wall_seconds, 6),
            "processed_fps_wall": round(processed / wall_seconds, 3) if wall_seconds else 0.0,
            "video": {
                "width": width,
                "height": height,
                "reported_fps": source_fps,
                "timestamp_fps": timestamp_fps if args.source_kind == "video" else None,
                "timestamp_basis": (
                    "media_pos_msec_with_frame_index_fallback"
                    if args.source_kind == "video" else "monotonic_elapsed"
                ),
                "fps_fallback_used": args.source_kind == "video" and source_fps is None,
                "media_timestamp_frames": media_timestamp_frames,
                "frame_index_fallback_frames": fallback_timestamp_frames,
            },
            "sampling": {
                "every_n_frames": args.every_n_frames,
                "max_processed_frames": args.max_frames,
                "effective_fps": effective_fps,
            },
            "device": args.device,
            "weights": _weight_identifier(args.weights_path),
            "dependencies": {
                name: _package_version(name)
                for name in ("ultralytics", "supervision", "opencv-python", "numpy")
            },
            "artifacts": {
                "snapshots_jsonl": "snapshots.jsonl",
                "summary_json": "summary.json",
                "annotated_video": "annotated.mp4" if video_created else None,
            },
            "notes": [
                "Track histories contain observations emitted by GeneralObjectTracker only.",
                "Unlabelled media provides operational measurements, not accuracy evidence.",
                "Latency percentiles cover only the bounded most-recent timing window.",
                "Annotated MP4 is constant-rate visualization and not a time-faithful recording.",
            ],
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if pending_error is not None:
            raise pending_error
        return summary
    except FileExistsError as exc:
        raise TrackingError(f"output path appeared during startup; no files overwritten: {output_dir}") from exc
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        if args.show:
            cv2.destroyAllWindows()


def main(argv: Optional[Sequence[str]] = None, *, dependencies: Optional[RuntimeDependencies] = None) -> int:
    parser = build_parser()
    args = _validated_args(parser, argv)
    try:
        summary = run(args, dependencies)
    except KeyboardInterrupt:
        print("track_objects: interrupted; partial artifacts were preserved", file=sys.stderr)
        return 130
    except (OSError, RuntimeError) as exc:
        print(f"track_objects: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
