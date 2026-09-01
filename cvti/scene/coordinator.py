"""Persistent bounded orchestration for per-camera scene mapping."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from cvti.scene.aggregation import aggregate_area, aggregate_site
from cvti.scene.context_store import SceneContextStore, _atomic_json_write, source_fingerprint
from cvti.scene.hierarchy import HierarchyContextStore
from cvti.serving.onboarding import camera_area_id, normalized_areas


@dataclass
class MappingJob:
    camera_id: str
    area_id: str
    source_fingerprint: str
    state: str = "pending"
    attempts: int = 0
    error: str = ""


@dataclass(frozen=True)
class MappingProgress:
    total: int
    pending: int
    running: int
    ready: int
    failed: int
    conflicts: int


class SceneMappingCoordinator:
    def __init__(self, service, site_path, output_dir, max_workers=1,
                 policy: str = "auto") -> None:
        self.service = service
        self.site_path = Path(site_path)
        self.output_dir = Path(output_dir)
        self.queue_path = self.output_dir / "context/mapping_queue.json"
        self.max_workers = max(1, int(max_workers))
        self.policy = policy
        self.jobs: list[MappingJob] = []
        self._stopped = False
        self.resume()

    def _site(self) -> dict:
        return json.loads(self.site_path.read_text())

    def _save(self) -> None:
        _atomic_json_write(self.queue_path, {"jobs": [asdict(job) for job in self.jobs]})

    def resume(self) -> None:
        try:
            payload = json.loads(self.queue_path.read_text())
            self.jobs = [MappingJob(**job) for job in payload.get("jobs", [])]
        except (OSError, TypeError, ValueError):
            self.jobs = []
        for job in self.jobs:
            if job.state == "running":
                job.state = "pending"
            elif job.state == "failed":
                # Failed scene maps heal themselves (the #69 contract): a
                # persisted failure must not outlive the condition that caused
                # it. Every restart gets a fresh attempt budget — without
                # this, four losses against a still-downloading model left a
                # camera unmapped FOREVER unless a human hit Remap.
                job.state = "pending"
                job.attempts = 0
                job.error = ""
        if self.jobs:
            self._save()

    def enqueue(self, camera_ids, force: bool = False) -> None:
        cameras = {str(camera["id"]): camera for camera in self._site().get("cameras", [])}
        # Prune jobs the config has outgrown — a removed camera or a changed
        # source must not haunt the persistent queue (and its progress totals)
        # forever.
        self.jobs = [
            job for job in self.jobs
            if job.camera_id in cameras
            and source_fingerprint(cameras[job.camera_id]["source"]) == job.source_fingerprint
        ]
        existing = {(job.camera_id, job.source_fingerprint) for job in self.jobs}
        for camera_id in camera_ids:
            camera = cameras.get(str(camera_id))
            if camera is None:
                raise ValueError(f"unknown camera: {camera_id}")
            fingerprint = source_fingerprint(camera["source"])
            key = (str(camera_id), fingerprint)
            matching = next(
                (job for job in self.jobs
                 if (job.camera_id, job.source_fingerprint) == key),
                None,
            )
            if matching is not None and force:
                matching.state = "pending"
                matching.error = ""
            elif key not in existing:
                self.jobs.append(MappingJob(str(camera_id), camera_area_id(camera), fingerprint))
                existing.add(key)
        self._save()

    def pending_camera_ids(self) -> list[str]:
        return [job.camera_id for job in self.jobs if job.state == "pending"]

    def requeue_failed(self, camera_ids=None, max_attempts: int = 4) -> list[str]:
        """Requeue eligible failures without allowing an unbounded retry loop."""
        selected = None if camera_ids is None else {str(value) for value in camera_ids}
        requeued = []
        for job in self.jobs:
            if job.state != "failed" or job.attempts >= max(1, int(max_attempts)):
                continue
            if selected is not None and job.camera_id not in selected:
                continue
            job.state = "pending"
            job.error = ""
            requeued.append(job.camera_id)
        if requeued:
            self._save()
        return requeued

    def run_next(self) -> bool:
        if self._stopped:
            return False
        job = next((item for item in self.jobs if item.state == "pending"), None)
        if job is None:
            return False
        site = self._site()
        camera = next((item for item in site.get("cameras", [])
                       if str(item["id"]) == job.camera_id), None)
        if camera is None:
            # The queue is persistent; the config is not. A camera deleted
            # while its job waited used to raise StopIteration here and kill
            # the mapping thread for good. A vanished camera is a finished
            # job, not a dead worker.
            job.state = "failed"
            job.error = "camera no longer exists in the site config"
            self._save()
            return True
        area = next(
            (item for item in site.get("areas", [])
             if str(item.get("id", "")) == job.area_id),
            {},
        )
        area_fields = {key: area[key] for key in ("area_type", "expected_actors")
                       if key in area}
        # The area's note is operator knowledge for the PROMPT — _operator_hints
        # reads scene_hint, so 'note' has to arrive under that name or it never
        # reaches the mapper. A camera's own scene_hint still wins.
        if area.get("note") and not camera.get("scene_hint"):
            area_fields["scene_hint"] = area["note"]
        camera = {
            **({"site_type": site["site_type"]} if site.get("site_type") else {}),
            **area_fields,
            **camera,
        }
        job.state = "running"
        job.attempts += 1
        self._save()
        result = self.service.prepare([camera], self.policy)
        status = result.statuses.get(job.camera_id, {})
        if status.get("status") in {"ready_unreviewed", "ready_reviewed"}:
            job.state = "ready"
            job.error = ""
            self._recompute(job.area_id, site)
        else:
            job.state = "failed"
            job.error = str(status.get("error", "mapping failed"))[:240]
        self._save()
        return True

    def run_until_idle(self) -> None:
        while self.run_next():
            pass

    def stop(self) -> None:
        self._stopped = True

    def progress(self) -> MappingProgress:
        counts = {state: sum(job.state == state for job in self.jobs)
                  for state in ("pending", "running", "ready", "failed")}
        conflicts = 0
        for area in normalized_areas(self.site_path):
            context = HierarchyContextStore(self.output_dir / "context").load_area(area["id"])
            conflicts += len((context or {}).get("conflicts", []))
        return MappingProgress(len(self.jobs), conflicts=conflicts, **counts)

    def _recompute(self, area_id: str, site: dict) -> None:
        hierarchy = HierarchyContextStore(self.output_dir / "context")
        area = next((item for item in normalized_areas(self.site_path)
                     if item["id"] == area_id), None)
        if area is None:
            return    # the area was deleted while this job ran; nothing to aggregate
        contexts = []
        for camera_id in area["camera_ids"]:
            context = SceneContextStore(
                self.output_dir / "context", camera_id
            )._load_context()
            if context:
                contexts.append(context)
        reviewed_path = hierarchy.area_dir(area_id) / "area_context.json"
        reviewed = hierarchy.load_area(area_id) if reviewed_path.exists() else None
        proposal = aggregate_area(area, contexts, reviewed=reviewed)
        if reviewed is None:
            hierarchy.save_area_proposal(proposal.context)

        area_contexts = []
        for item in normalized_areas(self.site_path):
            context = hierarchy.load_area(item["id"])
            if context:
                area_contexts.append(context)
        site_reviewed_path = hierarchy.site_dir / "site_context.json"
        reviewed_site = hierarchy.load_site() if site_reviewed_path.exists() else None
        site_proposal = aggregate_site(site, area_contexts, reviewed=reviewed_site)
        if reviewed_site is None:
            hierarchy.save_site_proposal(site_proposal)
