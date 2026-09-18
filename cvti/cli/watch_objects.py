"""Repeatable, local-only semantic object recognition experiments."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from cvti.object_watch.candidate_tracks import CandidateTrackAssociator
from cvti.object_watch.embeddings import embed_examples
from cvti.object_watch.matcher import ObjectCandidate, ObjectMatcher
from cvti.object_watch.presence import PresenceDebouncer
from cvti.object_watch.proposals import YoloWorldProposalProvider, merge_proposals, yolo_world_preflight
from cvti.object_watch.runtime_config import (
    ObjectWatchConfig, load_configured_backend, preflight, resolve_config, write_config,
)
from cvti.object_watch.store import (
    ObjectTarget, activate_target, add_example, load_targets, save_target, target_readiness,
)


class WatchObjectsError(RuntimeError):
    pass


def _yolo_world_provider(config: ObjectWatchConfig) -> YoloWorldProposalProvider:
    if config.world_weights is None or config.clip_weights is None:
        raise RuntimeError("configured local YOLO-World and CLIP weights are required")
    return YoloWorldProposalProvider(
        config.world_weights, config.clip_weights, device=config.device,
    )


@dataclass(frozen=True)
class RunDependencies:
    cv2: Any
    backend_loader: Callable[[ObjectWatchConfig], Any] = load_configured_backend
    monotonic: Callable[[], float] = time.monotonic
    proposal_provider_factory: Callable[[ObjectWatchConfig], Any] = _yolo_world_provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage and evaluate a local semantic object-recognition library (no downloads)."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    configure = commands.add_parser("configure", help="Persist local model configuration")
    configure.add_argument("--site-dir", required=True)
    configure.add_argument("--model-path", required=True)
    configure.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    configure.add_argument("--world-weights")
    configure.add_argument("--clip-weights")

    doctor = commands.add_parser("doctor", help="Print truthful JSON readiness")
    doctor.add_argument("--site-dir", required=True)

    enroll = commands.add_parser("enroll", help="Add canonical reviewed or draft examples")
    enroll.add_argument("--site-dir", required=True)
    enroll.add_argument("--object-id", required=True)
    enroll.add_argument("--label", required=True)
    enroll.add_argument("--description", default="")
    enroll.add_argument("--category", choices=("product", "vehicle", "pallet", "ppe", "custom"), default="custom")
    enroll.add_argument("--image", action="append", required=True)
    enroll.add_argument("--negative-image", action="append", default=[])
    enroll.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    enroll.add_argument("--review", action="store_true", help="Explicitly mark every supplied crop reviewed")

    embed = commands.add_parser("embed", help="Embed reviewed examples using configured SigLIP")
    embed.add_argument("--site-dir", required=True)

    activate = commands.add_parser("activate", help="Activate one ready object")
    activate.add_argument("--site-dir", required=True)
    activate.add_argument("--object-id", required=True)

    run = commands.add_parser("run", help="Run synchronous offline recognition over sampled frames")
    _run_args(run)

    evaluate = commands.add_parser("eval-crops", aliases=["eval"], help="Evaluate supplied crop boxes (detector bypass)")
    evaluate.add_argument("--site-dir", required=True)
    evaluate.add_argument("--candidates", required=True, help="JSONL with image, bbox, expected_object_id")
    evaluate.add_argument("--output-dir", required=True)
    return parser


def _run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--site-dir", required=True)
    parser.add_argument("--source", required=True, help="Existing video path or webcam index 0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", help="Existing generic YOLO weights; never downloaded")
    parser.add_argument("--proposals", choices=("none", "yolo_world"), default="none")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--show", action="store_true")


def _local_path(value: str, flag: str) -> Path:
    if "://" in value or value.lower().startswith(("http:/", "https:/")):
        raise WatchObjectsError(f"{flag} must be a local filesystem path")
    return Path(value).expanduser().resolve()


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise WatchObjectsError(f"OpenCV is unavailable: {exc}") from exc
    return cv2


def _config_and_backend(site: Path, loader=load_configured_backend):
    config = resolve_config(site)
    readiness = preflight(config)
    if readiness.status == "unavailable":
        raise WatchObjectsError("object watch unavailable: " + "; ".join(readiness.reasons))
    return config, loader(config)


def _configure(args: argparse.Namespace) -> dict[str, Any]:
    site = _local_path(args.site_dir, "--site-dir")
    model = _local_path(args.model_path, "--model-path")
    world = _local_path(args.world_weights, "--world-weights") if args.world_weights else None
    clip = _local_path(args.clip_weights, "--clip-weights") if args.clip_weights else None
    path = write_config(site, ObjectWatchConfig(
        backend="siglip", model_path=model, device=args.device,
        world_weights=world, clip_weights=clip,
    ))
    return {"status": "configured", "config": str(path), "downloads": False}


def _doctor(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    site = _local_path(args.site_dir, "--site-dir")
    try:
        config = resolve_config(site)
        ready = preflight(config)
        targets = load_targets(site)
        target_rows = []
        backend = None
        if ready.status != "unavailable":
            backend = load_configured_backend(config)
            target_rows = [asdict(target_readiness(site, target, backend)) for target in targets]
        proposal = yolo_world_preflight(config.world_weights, config.clip_weights)
        executable = backend is not None
        result = {
            "status": "ready" if executable else ready.status,
            "ready": executable, "reasons": list(ready.reasons),
            "backend": ready.backend, "model_fingerprint": ready.model_fingerprint,
            "site_dir": str(site), "target_count": len(targets), "targets": target_rows,
            "proposal_readiness": {
                "provider": "yolo_world",
                "configured": config.world_weights is not None or config.clip_weights is not None,
                "ready": proposal.ready,
                "reasons": list(proposal.reasons),
            },
            "downloads": False,
        }
        return result, 0 if executable else 2
    except (OSError, ValueError, RuntimeError) as exc:
        return {"status": "unavailable", "ready": False, "reasons": [str(exc)],
                "model_fingerprint": None, "site_dir": str(site), "downloads": False}, 2


def _enroll(args: argparse.Namespace) -> dict[str, Any]:
    site = _local_path(args.site_dir, "--site-dir")
    images = [(_local_path(value, "--image"), False) for value in args.image]
    images += [(_local_path(value, "--negative-image"), True) for value in args.negative_image]
    missing = [str(path) for path, _ in images if not path.is_file()]
    if missing:
        raise WatchObjectsError("image is missing: " + ", ".join(missing))
    existing = next((item for item in load_targets(site) if item.id == args.object_id), None)
    if existing and (existing.label != args.label or existing.category != args.category):
        raise WatchObjectsError("existing object label/category differs; use a new isolated site or matching values")
    target = existing or ObjectTarget(args.object_id, args.label, args.category)
    save_target(site, replace(target, grounding_description=args.description or target.grounding_description))
    bbox = tuple(args.bbox) if args.bbox else (0.0, 0.0, 1.0, 1.0)
    bbox_format = "pixel_xyxy" if args.bbox else "normalized_xyxy"
    added = []
    for path, negative in images:
        example = add_example(site, args.object_id, path.read_bytes(), bbox, str(path),
                              negative=negative, bbox_format=bbox_format, reviewed=args.review)
        added.append({"example_id": example.id, "negative": negative, "reviewed": example.reviewed})
    return {"status": "enrolled", "object_id": args.object_id, "examples": added,
            "review_explicit": bool(args.review)}


def _embed(args: argparse.Namespace) -> dict[str, Any]:
    site = _local_path(args.site_dir, "--site-dir")
    _config, backend = _config_and_backend(site)
    return {"status": "embedded", "written": embed_examples(site, backend),
            "model_fingerprint": backend.fingerprint}


def _activate(args: argparse.Namespace) -> dict[str, Any]:
    site = _local_path(args.site_dir, "--site-dir")
    _config, backend = _config_and_backend(site)
    target = activate_target(site, args.object_id, backend)
    return {"status": "active", "object_id": target.id, "revision": target.revision}


def _generic_provider(weights: Path, device: str):
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights))
    except Exception as exc:
        raise WatchObjectsError(f"generic proposal model unavailable: {exc}") from exc

    def propose(frame) -> list[ObjectCandidate]:
        result = model.predict(source=frame, device=device, verbose=False)[0]
        names = getattr(result, "names", getattr(model, "names", {}))
        rows = []
        for box in result.boxes:
            coords = tuple(int(round(v)) for v in box.xyxy[0].tolist())
            cls = int(box.cls[0]); confidence = float(box.conf[0])
            rows.append(ObjectCandidate(coords, str(names.get(cls, cls)), confidence))
        return rows
    return propose


def _decision_row(decision, frame_index: int, timestamp: float, elapsed_ms: float,
                  proposal_count: int = 1) -> dict[str, Any]:
    return {
        "frame_index": frame_index, "timestamp": timestamp, "status": decision.status,
        "reason": decision.reason, "object_id": decision.object_id,
        "best_similarity": decision.best_similarity,
        "runner_up_similarity": decision.runner_up_similarity,
        "bbox": list(decision.candidate.bbox), "track_id": decision.candidate.track_id,
        "recognition_ms": round(elapsed_ms, 3), "proposal_count": proposal_count,
    }


def _validated_run(args: argparse.Namespace) -> tuple[Path, Any, Path]:
    site = _local_path(args.site_dir, "--site-dir")
    output = _local_path(args.output_dir, "--output-dir")
    if output.exists():
        raise WatchObjectsError(f"--output-dir already exists: {output}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise WatchObjectsError("--max-frames must be positive")
    source = int(args.source) if str(args.source).isdecimal() else _local_path(args.source, "--source")
    if not isinstance(source, int) and not source.is_file():
        raise WatchObjectsError(f"--source is not an existing video: {source}")
    return site, source, output


def _run(args: argparse.Namespace, dependencies: RunDependencies | None) -> dict[str, Any]:
    site, source, output = _validated_run(args)
    deps = dependencies or RunDependencies(_load_cv2())
    config, backend = _config_and_backend(site, deps.backend_loader)  # before output creation
    matcher = ObjectMatcher(site, backend, max_candidates_per_frame=config.max_candidates)
    if not matcher.index.targets:
        raise WatchObjectsError("recognition index has no active ready targets")
    generic_provider = None
    if args.weights:
        weights = _local_path(args.weights, "--weights")
        if not weights.is_file():
            raise WatchObjectsError("--weights must be an existing local file; downloads are disabled")
        generic_provider = _generic_provider(weights, config.device)
    proposal_provider = None
    if args.proposals == "yolo_world":
        if config.world_weights is None or config.clip_weights is None:
            raise WatchObjectsError(
                "YOLO-World unavailable: configured local YOLO-World and CLIP weights are required"
            )
        try:
            proposal_provider = deps.proposal_provider_factory(config)
        except Exception as exc:
            raise WatchObjectsError(f"YOLO-World unavailable: {exc}") from exc
    phrases = tuple(
        item.target.grounding_description
        for item in matcher.index.targets
        if item.target.grounding_description
    )
    cv2, capture, writer = deps.cv2, None, None
    output.mkdir(parents=True, exist_ok=False)
    decisions_path, summary_path = output / "decisions.jsonl", output / "summary.json"
    evidence = output / "evidence"
    counts, timings = Counter(), []
    tracker, presence = None, PresenceDebouncer(min_observations=2)
    outcome, error, frame_index = "complete", None, -1
    video_created = False
    try:
        capture = cv2.VideoCapture(source)
        if capture is None or not capture.isOpened():
            raise WatchObjectsError(f"cannot open source: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        tracker = CandidateTrackAssociator(
            camera_id="local-experiment", source_generation=0,
            expected_fps=fps, max_tracks=max(8, config.max_candidates),
        )
        with decisions_path.open("x") as stream:
            while args.max_frames is None or counts["processed_frames"] < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    if counts["source_frames_read"] == 0:
                        raise WatchObjectsError("source opened but returned zero frames")
                    break
                frame_index += 1; counts["source_frames_read"] += 1
                timestamp = frame_index / fps
                # OpenCV and the production object-watch path both provide BGR
                # ndarrays to Ultralytics; SigLIP conversion remains in matcher.
                generic = generic_provider(frame) if generic_provider else ()
                generated = (proposal_provider.propose(
                    frame, phrases, limit_candidates=config.max_candidates,
                ) if proposal_provider else ())
                candidates = tracker.assign(merge_proposals(
                    generic, generated, max_candidates=config.max_candidates,
                ), timestamp)
                counts["proposal_candidates"] += len(candidates)
                started = deps.monotonic()
                matches = matcher.match("local-experiment", frame, list(candidates), timestamp)
                elapsed = (deps.monotonic() - started) * 1000
                timings.append(elapsed); counts["processed_frames"] += 1
                for decision in matcher.last_decisions:
                    counts[decision.status] += 1
                    stream.write(json.dumps(_decision_row(
                        decision, frame_index, timestamp, elapsed, len(candidates)
                    ), sort_keys=True)+"\n")
                for match in matches:
                    if match.track_id is not None:
                        token = presence.ready(
                            "local-experiment", 1, "semantic-recognition", match.object_id,
                            match.track_id, frame_index, timestamp,
                        )
                        if token and presence.commit(token):
                            counts["presence_candidates"] += 1
                            stream.write(json.dumps({
                                "frame_index": frame_index, "timestamp": timestamp,
                                "status": "presence_candidate", "state": "object_seen",
                                "object_id": match.object_id, "track_id": match.track_id,
                                "bbox": list(match.bbox), "similarity": match.similarity,
                                "verified_alert": False,
                            }, sort_keys=True)+"\n")
                    if counts["evidence_pairs"] < 20:
                        crop, _ = matcher._crop_png(frame, match.bbox)
                        reference_id = match.reference_example_ids[0] if match.reference_example_ids else None
                        target = next(item for item in load_targets(site) if item.id == match.object_id)
                        example = next((item for item in target.examples if item.id == reference_id), None)
                        if crop and example:
                            evidence.mkdir(exist_ok=True)
                            number = counts["evidence_pairs"]
                            (evidence/f"{number:04d}-candidate.png").write_bytes(crop)
                            library = site if site.name == "object_library" else site/"object_library"
                            shutil.copyfile(library/example.path, evidence/f"{number:04d}-reference.png")
                            counts["evidence_pairs"] += 1
                annotated = frame.copy()
                if args.save_video or args.show:
                    for match in matches:
                        cv2.rectangle(annotated, match.bbox[:2], match.bbox[2:], (0, 255, 0), 2)
                    if args.save_video and writer is None:
                        writer = cv2.VideoWriter(str(output/"annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                 (int(frame.shape[1]), int(frame.shape[0])))
                        if not writer.isOpened(): raise WatchObjectsError("cannot create annotated video")
                        video_created = True
                    if writer is not None: writer.write(annotated)
                    if args.show:
                        cv2.imshow("CVTI local semantic recognition", annotated)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")): break
    except BaseException as exc:
        outcome, error = ("interrupted" if isinstance(exc, KeyboardInterrupt) else "error"), str(exc)
    finally:
        if writer is not None: writer.release()
        if capture is not None: capture.release()
        if args.show: cv2.destroyAllWindows()
    summary = {
        "schema_version": 1, "experiment": "offline_semantic_recognition", "outcome": outcome,
        "error": error, "counts": {
            key: counts[key] for key in (
                "source_frames_read", "processed_frames", "proposal_candidates",
                "matched", "rejected", "ambiguous", "presence_candidates", "evidence_pairs",
            )
        }, "model_fingerprint": backend.fingerprint,
        "proposal_mode": ("yolo_world" if proposal_provider else
                          "generic_yolo" if generic_provider else "none"),
        "detector_bypass": False,
        "timing_ms": {"recognition_total": round(sum(timings), 3), "samples": len(timings)},
        "artifacts": {"decisions": "decisions.jsonl",
                      "evidence_dir": "evidence" if evidence.exists() else None,
                      "annotated_video": "annotated.mp4" if video_created else None},
        "notes": ["Synchronous sampled-frame experiment; not a live-performance benchmark.",
                  "Presence candidates are diagnostic decisions, not verified alerts.",
                  "Annotated MP4 is a constant-rate visualization."],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    if outcome == "interrupted": raise KeyboardInterrupt
    if outcome == "error": raise WatchObjectsError(error or "run failed")
    return summary


def _eval_crops(args: argparse.Namespace, dependencies: RunDependencies | None) -> dict[str, Any]:
    site = _local_path(args.site_dir, "--site-dir")
    candidates_path = _local_path(args.candidates, "--candidates")
    output = _local_path(args.output_dir, "--output-dir")
    if output.exists(): raise WatchObjectsError(f"--output-dir already exists: {output}")
    if not candidates_path.is_file(): raise WatchObjectsError("--candidates must be an existing JSONL file")
    deps = dependencies or RunDependencies(_load_cv2())
    _config, backend = _config_and_backend(site, deps.backend_loader)
    matcher = ObjectMatcher(site, backend)
    if not matcher.index.targets: raise WatchObjectsError("recognition index has no active ready targets")
    rows = []
    for number, line in enumerate(candidates_path.read_text().splitlines(), 1):
        try: row = json.loads(line)
        except json.JSONDecodeError as exc: raise WatchObjectsError(f"invalid candidates JSONL line {number}") from exc
        image = _local_path(str(row.get("image", "")), "candidate image")
        bbox = row.get("bbox")
        if not image.is_file() or not isinstance(bbox, list) or len(bbox) != 4:
            raise WatchObjectsError(f"invalid candidate line {number}: image and four-coordinate bbox required")
        rows.append((row, image, tuple(int(v) for v in bbox)))
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"; evidence.mkdir()
    counts = Counter(); decisions = output / "decisions.jsonl"
    try:
        with decisions.open("x") as stream:
            for index, (ground, image, bbox) in enumerate(rows):
                frame = deps.cv2.imread(str(image))
                if frame is None: raise WatchObjectsError(f"candidate image is undecodable: {image}")
                started = deps.monotonic()
                matches = matcher.match("crop-eval", frame, [ObjectCandidate(bbox, confidence=1.0, track_id=index)], float(index))
                elapsed = (deps.monotonic()-started)*1000
                decision = matcher.last_decisions[0]; expected = ground.get("expected_object_id")
                predicted = matches[0].object_id if matches else None
                counts[decision.status] += 1; counts["correct"] += int(predicted == expected)
                out = _decision_row(decision, index, float(index), elapsed)
                out.update({"expected_object_id": expected, "correct": predicted == expected, "detector_bypass": True})
                stream.write(json.dumps(out, sort_keys=True)+"\n")
                if matches and index < 20:
                    crop, _ = matcher._crop_png(frame, bbox)
                    if crop: (evidence/f"{index:04d}-candidate.png").write_bytes(crop)
                    ref = matches[0].reference_example_ids[0] if matches[0].reference_example_ids else None
                    target = next(item for item in load_targets(site) if item.id == matches[0].object_id)
                    example = next((item for item in target.examples if item.id == ref), None)
                    if example:
                        library = site if site.name == "object_library" else site/"object_library"
                        shutil.copyfile(library/example.path, evidence/f"{index:04d}-reference.png")
    except BaseException:
        raise  # preserve partial output
    total = len(rows)
    summary = {"schema_version": 1, "experiment": "crop_only_recognition",
               "detector_bypass": True, "localization_accuracy_measured": False,
               "counts": {"candidates": total, "correct": counts["correct"],
                          "matched": counts["matched"], "rejected": counts["rejected"],
                          "ambiguous": counts["ambiguous"]},
               "recognition_accuracy": counts["correct"]/total if total else None,
               "model_fingerprint": backend.fingerprint,
               "notes": ["Ground-truth boxes bypass proposal discovery; this does not measure localization accuracy."]}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    return summary


def main(argv: Sequence[str] | None = None, *, dependencies: RunDependencies | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "configure": result, code = _configure(args), 0
        elif args.command == "doctor": result, code = _doctor(args)
        elif args.command == "enroll": result, code = _enroll(args), 0
        elif args.command == "embed": result, code = _embed(args), 0
        elif args.command == "activate": result, code = _activate(args), 0
        elif args.command == "run": result, code = _run(args, dependencies), 0
        else: result, code = _eval_crops(args, dependencies), 0
    except KeyboardInterrupt:
        print("watch_objects: interrupted; partial artifacts were preserved", file=sys.stderr); return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"watch_objects: error: {exc}", file=sys.stderr); return 2
    print(json.dumps(result, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
