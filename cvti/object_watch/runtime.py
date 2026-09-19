"""Bounded background runtime for reference object recognition."""

from __future__ import annotations

import inspect
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from cvti.object_watch.evidence import build_object_watch_evidence
from cvti.object_watch.matcher import ObjectCandidate, ObjectMatch, ObjectMatcher, build_recognition_index
from cvti.object_watch.presence import PresenceDebouncer, PresenceToken
from cvti.object_watch.proposals import YoloWorldProposalProvider, merge_proposals
from cvti.object_watch.runtime_config import (
    ObjectWatchConfig,
    load_configured_backend,
    preflight,
    resolve_config,
)
from cvti.object_watch.store import library_revision
from cvti.logging_setup import get_logger


log = get_logger(__name__)


@dataclass(frozen=True)
class WatchZone:
    """Immutable, frame-fitted zone geometry handed to the worker."""

    name: str
    polygon: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("zone name is required")
        polygon = tuple((int(point[0]), int(point[1])) for point in self.polygon)
        if len(polygon) < 3:
            raise ValueError("zone polygon must contain at least three points")
        object.__setattr__(self, "polygon", polygon)


@dataclass(frozen=True)
class WatchSample:
    camera_id: str
    source_generation: int
    sample_sequence: int
    observed_at_monotonic: float
    event_timestamp: float
    frame: Any
    candidates: tuple[Any, ...] = ()
    zones: tuple[Any, ...] = ()
    proposal_provider: str = "none"
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id is required")
        if self.source_generation < 0 or self.sample_sequence < 0:
            raise ValueError("source generation and sequence must be non-negative")
        owned = np.ascontiguousarray(np.asarray(self.frame).copy())
        if owned.ndim != 3 or owned.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 array")
        owned.setflags(write=False)
        object.__setattr__(self, "frame", owned)
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "zones", tuple(self.zones))


@dataclass(frozen=True)
class WatchResult:
    camera_id: str
    source_generation: int
    sample_sequence: int
    observed_at_monotonic: float
    event_timestamp: float
    library_revision: int
    target_revision: int
    model_fingerprint: str
    matches: tuple[ObjectMatch, ...] = ()
    evidence: tuple[Any, ...] = ()
    status: str = "ok"
    timings: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""
    _config_signature: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        frozen = []
        for item in self.evidence:
            if isinstance(item, np.ndarray):
                item = np.ascontiguousarray(item.copy())
                item.setflags(write=False)
            frozen.append(item)
        object.__setattr__(self, "matches", tuple(self.matches))
        object.__setattr__(self, "evidence", tuple(frozen))


class ObjectWatchRuntime:
    """One semantic worker with latest-only input and bounded output."""

    def __init__(self, config: ObjectWatchConfig, *, backend_factory=None,
                 proposal_factory=None) -> None:
        if not isinstance(config, ObjectWatchConfig):
            raise TypeError("config must be ObjectWatchConfig")
        if config.library_path is None:
            raise ValueError("config.library_path is required")
        self.config = config
        self._library = Path(config.library_path)
        self._backend_factory = backend_factory
        self._proposal_factory = proposal_factory
        self._condition = threading.Condition(threading.Lock())
        self._pending: dict[str, WatchSample] = {}
        self._round_robin: deque[str] = deque()
        self._results: dict[str, deque[tuple[WatchResult, ...]]] = {}
        self._generation: dict[str, int] = {}
        self._last_sequence: dict[tuple[str, int], int] = {}
        self._trackers: dict[tuple[str, int], Any] = {}
        self._track_labels: dict[tuple[str, int, int], tuple[str, float]] = {}
        self._stability = PresenceDebouncer()
        self._presence = PresenceDebouncer(min_observations=1)
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._max_cameras = 32
        # Bound complete samples, never individual matches: truncating a six-match
        # sample to four made the same last two targets starve forever.
        self._max_result_groups_per_camera = 2
        self._status = "stopped"
        self._reason = ""
        self._processed = 0
        self._dropped = 0
        # Published only by the worker. submit/drain read memory and never touch
        # runtime.json, targets.json, or advisory filesystem locks.
        self._snapshot_signature: tuple[Any, ...] = ()
        self._snapshot_revision = -1
        self._snapshot_active: dict[str, int] = {}
        self._snapshot_model_fingerprint = ""
        self._proposal_lock = threading.Lock()
        self._proposal: Any = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._status = "starting"
            self._thread = threading.Thread(target=self._run, name="object-watch", daemon=True)
            self._thread.start()

    def submit(self, sample: WatchSample) -> bool:
        if not isinstance(sample, WatchSample):
            raise TypeError("sample must be WatchSample")
        with self._condition:
            if self._stopping or self._thread is None:
                return False
            generation = self._generation.get(sample.camera_id)
            if generation is not None and generation != sample.source_generation:
                return False
            key = (sample.camera_id, sample.source_generation)
            if sample.sample_sequence <= self._last_sequence.get(key, -1):
                return False
            if sample.camera_id not in self._generation and len(self._generation) >= self._max_cameras:
                self._dropped += 1
                return False
            self._generation.setdefault(sample.camera_id, sample.source_generation)
            self._last_sequence[key] = sample.sample_sequence
            if sample.camera_id in self._pending:
                self._dropped += 1
            else:
                self._round_robin.append(sample.camera_id)
            self._pending[sample.camera_id] = sample
            self._condition.notify()
            return True

    def drain(self, camera_id: str, source_generation: int,
              now_monotonic: float | None = None) -> list[WatchResult]:
        # Never trust a batch timestamp captured before shared inference.
        now_monotonic = time.monotonic()
        with self._condition:
            queue = self._results.get(camera_id)
            rows = [row for group in (queue or ()) for row in group]
            if queue is not None:
                queue.clear()
            signature = self._snapshot_signature
            revision = self._snapshot_revision
            active = dict(self._snapshot_active)
            model_fingerprint = self._snapshot_model_fingerprint
            ttl_seconds = self.config.result_ttl_seconds
        valid: list[WatchResult] = []
        for row in rows:
            if row.source_generation != source_generation:
                continue
            if now_monotonic - row.observed_at_monotonic > ttl_seconds:
                continue
            if row._config_signature != signature or row.library_revision != revision:
                continue
            if (row.model_fingerprint and model_fingerprint
                    and row.model_fingerprint != model_fingerprint):
                continue
            if row.matches and any(active.get(match.object_id) != match.target_revision
                                   for match in row.matches):
                continue
            valid.append(row)
        return valid

    def reset_camera(self, camera_id: str, source_generation: int) -> None:
        with self._condition:
            self._generation[camera_id] = source_generation
            self._pending.pop(camera_id, None)
            self._round_robin = deque(value for value in self._round_robin if value != camera_id)
            self._results.pop(camera_id, None)
            self._last_sequence = {key: value for key, value in self._last_sequence.items()
                                   if key[0] != camera_id}
            self._trackers = {key: value for key, value in self._trackers.items()
                              if key[0] != camera_id}
            self._track_labels = {key: value for key, value in self._track_labels.items()
                                  if key[0] != camera_id}
            self._stability.reset_camera(camera_id)
            self._presence.reset_camera(camera_id)

    def reserve(self, result: WatchResult, rule_key: str):
        """Reserve this stable match for one rule without consuming it.

        Call :meth:`commit` only after the corresponding QueuedAlert is
        accepted. A rejected/deduplicated queue attempt can therefore retry on
        the next distinct sample. Rule identity is part of the key, so two
        scoped rules for one target do not suppress each other.
        """
        if not rule_key or result.status != "matched" or len(result.matches) != 1:
            return None
        with self._condition:
            if not self._result_current_locked(result):
                return None
            match = result.matches[0]
            if match.track_id is None:
                return None
            return self._presence.ready(
                result.camera_id, result.source_generation, str(rule_key), match.object_id,
                match.track_id,
                result.sample_sequence, result.observed_at_monotonic,
            )

    def commit(self, token: Any) -> bool:
        """Consume one reservation after its exact alert was queue-admitted."""
        if (not isinstance(token, PresenceToken) or len(token.key) < 5
                or token.key[4] is None):
            return False
        with self._condition:
            return self._presence.commit(token)

    def result_current(self, result: WatchResult) -> bool:
        """Cheap in-memory admission check for the serving frame path."""
        with self._condition:
            return self._result_current_locked(result)

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {"status": self._status, "reason": self._reason,
                    "pending_cameras": len(self._pending),
                    "result_cameras": len(self._results),
                    "processed": self._processed, "dropped": self._dropped}

    def stop(self, timeout: float = 0.25) -> None:
        """Signal shutdown without waiting on an in-flight model invocation."""
        with self._condition:
            self._stopping = True
            self._pending.clear()
            self._round_robin.clear()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout)))
        with self._condition:
            if thread is None or not thread.is_alive():
                self._status = "stopped"
                self._thread = None
            else:
                self._status = "stopping"

    def _run(self) -> None:
        loaded_signature = self._worker_config_signature()
        backend, proposal, loaded_signature = self._reload_runtime(loaded_signature)

        while True:
            with self._condition:
                while not self._round_robin and not self._stopping:
                    self._condition.wait(timeout=1.0)
                    if not self._round_robin:
                        break
                if self._stopping:
                    self._status = "stopped"
                    self._thread = None
                    return
                if self._round_robin:
                    camera_id = self._round_robin.popleft()
                    sample = self._pending.pop(camera_id, None)
                else:
                    sample = None
            signature = self._worker_config_signature()
            if backend is None or signature != loaded_signature:
                backend, proposal, loaded_signature = self._reload_runtime(signature)
            if sample is None:
                continue
            if backend is None:
                self._admit(self._result(sample, status="unavailable", reason=self._reason))
                continue
            try:
                results = self._process(sample, backend, proposal)
            except Exception as exc:
                log.debug("object-watch sample processing failed", exc_info=True)
                results = [self._result(sample, status="error", reason=str(exc)[:300])]
            self._admit_results(results)

    def _reload_runtime(self, signature: tuple[Any, ...]):
        backend = proposal = None
        self._proposal = None
        readiness = None
        try:
            config = self.config
            # Production config is canonical and live. Injected test backends
            # retain their explicit config when runtime.json is absent.
            if (self._library / "runtime.json").exists():
                config = resolve_config(self._library.parent)
                self.config = config
                signature = self._worker_config_signature()
            if self._backend_factory is None:
                readiness = preflight(config)
                if readiness.status == "unavailable":
                    raise RuntimeError("; ".join(readiness.reasons))
                backend = load_configured_backend(config)
            else:
                backend = _call_factory(self._backend_factory, config)
            if self._proposal_factory is not None:
                proposal = _call_factory(self._proposal_factory, config)
            self._proposal = proposal
            with self._condition:
                self._snapshot_signature = signature
                self._snapshot_model_fingerprint = str(getattr(backend, "fingerprint", ""))
                self._status = "degraded" if (
                    readiness is not None and readiness.status == "degraded"
                ) else "ready"
                self._reason = "; ".join(readiness.reasons) if (
                    readiness is not None and readiness.reasons
                ) else ""
        except Exception as exc:  # model readiness must never kill camera threads
            log.debug("object-watch runtime reload unavailable", exc_info=True)
            with self._condition:
                self._snapshot_signature = signature
                self._status, self._reason = "unavailable", str(exc)[:300]
        return backend, proposal, signature

    def _process(self, sample: WatchSample, backend: Any, proposal: Any) -> list[WatchResult]:
        started = time.monotonic()
        config_signature = self._snapshot_signature
        if not self._current(sample):
            return [self._result(sample, status="stale", reason="source generation changed")]
        provider = sample.proposal_provider.strip().lower().replace("-", "_")
        if provider == "disabled":
            provider = "none"
        if provider not in {"none", "yolo_world"}:
            return [self._result(sample, status="unavailable",
                                 reason="unsupported proposal provider")]
        # Embedding backends own their inference lock. Holding that same OS lock
        # around matcher.match deadlocks SiglipEmbeddingBackend.embed_image.
        index = build_recognition_index(self._library, backend)
        with self._condition:
            self._snapshot_revision = index.library_revision
            self._snapshot_active = {
                item.target.id: item.target.revision for item in index.targets
            }
            self._snapshot_model_fingerprint = index.model_fingerprint
        candidates = tuple(sample.candidates)
        if provider == "yolo_world":
            proposal = self._proposal if proposal is None else proposal
            if proposal is None:
                if self.config.world_weights is None or self.config.clip_weights is None:
                    return [self._result(sample, status="unavailable",
                                         reason="local YOLO-World and CLIP weights are required")]
                proposal = YoloWorldProposalProvider(
                    self.config.world_weights, self.config.clip_weights,
                    device=self.config.device,
                )
                self._proposal = proposal
            phrases = [item.target.grounding_description for item in index.targets
                       if item.target.grounding_description]
            with self._proposal_lock:
                generated = (proposal.propose(sample.frame, phrases)
                             if hasattr(proposal, "propose") else proposal(sample.frame, phrases))
        else:
            generated = ()
        limit = sample.max_candidates or self.config.max_candidates
        candidates = merge_proposals(candidates, generated, max_candidates=limit)
        candidates = _assign_candidate_zones(candidates, sample.zones)
        candidates = self._track(sample, candidates)
        matcher = ObjectMatcher.from_index(backend, index, max_candidates_per_frame=limit)
        matches = matcher.match(sample.camera_id, sample.frame, list(candidates),
                                sample.event_timestamp)
        matches = self._enforce_track_labels(sample, matches)

        stable: list[ObjectMatch] = []
        for match in matches:
            if match.track_id is None:
                stable.append(match)
                continue
            with self._condition:
                token = self._stability.ready(
                    sample.camera_id, sample.source_generation, "recognition", match.object_id,
                    match.track_id,
                    sample.sample_sequence, sample.observed_at_monotonic,
                )
            if token is not None:
                stable.append(match)
        elapsed = (time.monotonic() - started) * 1000.0
        if not stable:
            return [WatchResult(
                sample.camera_id, sample.source_generation, sample.sample_sequence,
                sample.observed_at_monotonic, sample.event_timestamp, index.library_revision,
                0, index.model_fingerprint, (), (), "no_match", {"total_ms": elapsed}, "",
                config_signature,
            )]
        results = []
        for match in stable:
            evidence = tuple(self._evidence(sample, match, index))
            if len(evidence) < 2:
                continue
            results.append(WatchResult(
                sample.camera_id, sample.source_generation, sample.sample_sequence,
                sample.observed_at_monotonic, sample.event_timestamp, index.library_revision,
                match.target_revision, index.model_fingerprint, (match,), evidence,
                "matched", {"total_ms": elapsed}, "", config_signature,
            ))
        return results or [WatchResult(
            sample.camera_id, sample.source_generation, sample.sample_sequence,
            sample.observed_at_monotonic, sample.event_timestamp, index.library_revision,
            0, index.model_fingerprint, (), (), "evidence_unavailable",
            {"total_ms": elapsed}, "reference evidence unavailable", config_signature,
        )]

    def _enforce_track_labels(self, sample: WatchSample,
                              matches: list[ObjectMatch]) -> list[ObjectMatch]:
        """Keep one semantic identity on a live proposal track."""
        kept: list[ObjectMatch] = []
        now = sample.observed_at_monotonic
        with self._condition:
            for match in matches:
                if match.track_id is None:
                    kept.append(match)
                    continue
                key = (sample.camera_id, sample.source_generation, int(match.track_id))
                previous = self._track_labels.get(key)
                if (previous is not None and previous[0] != match.object_id
                        and now - previous[1] <= self._stability.rearm_gap_seconds
                        and previous[0] in self._snapshot_active):
                    continue
                self._track_labels[key] = (match.object_id, now)
                kept.append(match)
            if len(self._track_labels) > 512:
                oldest = sorted(self._track_labels,
                                key=lambda key: self._track_labels[key][1])
                for key in oldest[:len(self._track_labels) - 512]:
                    self._track_labels.pop(key, None)
        return kept

    def _track(self, sample: WatchSample,
               candidates: tuple[ObjectCandidate, ...]) -> tuple[ObjectCandidate, ...]:
        # Shared detector candidates already carry stable track IDs. Avoid
        # constructing the proposal tracker unless at least one generated box
        # actually needs association.
        if all(candidate.track_id is not None for candidate in candidates):
            return candidates
        from cvti.object_watch.candidate_tracks import CandidateTrackAssociator

        key = (sample.camera_id, sample.source_generation)
        tracker = self._trackers.get(key)
        if tracker is None:
            tracker = CandidateTrackAssociator(
                camera_id=sample.camera_id,
                source_generation=sample.source_generation,
                expected_fps=self.config.sample_fps,
                max_tracks=max(8, self.config.max_candidates),
            )
            self._trackers[key] = tracker
        return tracker.assign(candidates, sample.observed_at_monotonic)

    def _evidence(self, sample: WatchSample, match: ObjectMatch, index: Any) -> list[Any]:
        indexed = next((item for item in index.targets
                        if item.target.id == match.object_id), None)
        target = indexed.target if indexed is not None else None
        if target is None:
            return []
        example_id = match.reference_example_ids[0] if match.reference_example_ids else None
        example = next((item for item in target.examples if item.id == example_id), None)
        if example is None or ".." in Path(example.path).parts:
            return []
        root = self._library.resolve()
        path = (root / example.path).resolve()
        if root not in path.parents or not path.is_file():
            return []
        try:
            evidence = build_object_watch_evidence(sample.frame, match.bbox, path.read_bytes(),
                                                   match.object_label)
            if library_revision(self._library) != index.library_revision:
                return []
            return evidence
        except (OSError, ValueError):
            return []

    def _admit(self, result: WatchResult) -> None:
        self._admit_results([result])

    def _admit_results(self, results: list[WatchResult]) -> None:
        if not results:
            return
        result = results[0]
        if (not self._current_result(result) or result.status == "stale"
                or time.monotonic() - result.observed_at_monotonic
                > self.config.result_ttl_seconds):
            with self._condition:
                current_generation = self._generation.get(result.camera_id)
                self._trackers.pop((result.camera_id, result.source_generation), None)
                self._track_labels = {
                    key: value for key, value in self._track_labels.items()
                    if not (key[0] == result.camera_id
                            and key[1] == result.source_generation)
                }
                self._stability.reset_camera(result.camera_id, current_generation)
                self._presence.reset_camera(result.camera_id, current_generation)
            return
        with self._condition:
            queue = self._results.setdefault(result.camera_id,
                                             deque(maxlen=self._max_result_groups_per_camera))
            current = tuple(row for row in results if self._result_current_locked(row))
            if current:
                queue.append(current)
                self._processed += len(current)

    def _result(self, sample: WatchSample, *, status: str, reason: str) -> WatchResult:
        return WatchResult(sample.camera_id, sample.source_generation, sample.sample_sequence,
                           sample.observed_at_monotonic, sample.event_timestamp,
                           self._snapshot_revision, 0, "", (), (), status, {}, reason,
                           self._snapshot_signature)

    def _current(self, sample: WatchSample) -> bool:
        with self._condition:
            return self._generation.get(sample.camera_id) == sample.source_generation

    def _current_result(self, result: WatchResult) -> bool:
        with self._condition:
            return self._result_current_locked(result)

    def _result_current_locked(self, result: WatchResult) -> bool:
        return (self._generation.get(result.camera_id) == result.source_generation
                and result._config_signature == self._snapshot_signature
                and result.library_revision == self._snapshot_revision
                and (not result.model_fingerprint or not self._snapshot_model_fingerprint
                     or result.model_fingerprint == self._snapshot_model_fingerprint)
                and all(self._snapshot_active.get(match.object_id) == match.target_revision
                        for match in result.matches))

    def _worker_config_signature(self) -> tuple[Any, ...]:
        path = self._library / "runtime.json"
        try:
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        except OSError:
            stamp = ()
        artifacts = []
        config = self.config
        for configured in (config.model_path, config.world_weights, config.clip_weights):
            if configured is None:
                continue
            path = Path(configured)
            paths = sorted(path.iterdir()) if path.is_dir() else [path]
            for artifact in paths:
                try:
                    stat = artifact.stat()
                    artifacts.append((str(artifact.resolve()), stat.st_size, stat.st_mtime_ns))
                except OSError:
                    artifacts.append((str(artifact), None, None))
        return (stamp, tuple(artifacts))


def _call_factory(factory: Callable[..., Any], config: ObjectWatchConfig) -> Any:
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = {"config": None}
    return factory() if not parameters else factory(config)


def _assign_candidate_zones(
    candidates: tuple[ObjectCandidate, ...], zones: tuple[Any, ...],
) -> tuple[ObjectCandidate, ...]:
    """Apply camera-equivalent centre-point membership after proposal NMS."""
    geometry = tuple(zone for zone in zones if isinstance(zone, WatchZone))
    if not geometry:
        return candidates

    import cv2

    polygons = tuple(
        (zone.name, np.asarray(zone.polygon, dtype=np.float32)) for zone in geometry
    )
    assigned = []
    for candidate in candidates:
        # Detector candidates were already zoned on the camera thread.  Only
        # generated proposals lack that context; preserve supplied identities
        # while filling the post-NMS YOLO-World gaps from the immutable copy.
        if candidate.zone_id is not None:
            assigned.append(candidate)
            continue
        x1, y1, x2, y2 = candidate.bbox
        point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        zone_id = next((name for name, polygon in polygons
                        if cv2.pointPolygonTest(polygon, point, False) >= 0), None)
        assigned.append(replace(candidate, zone_id=zone_id))
    return tuple(assigned)
