"""Bounded, site-scoped background jobs for object-watch enrollment."""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cvti.object_watch.runtime_config import ObjectWatchConfig
from cvti.logging_setup import get_logger
from cvti.utils import redact_credentials


log = get_logger(__name__)


_MAX_RETAINED_JOBS = 128
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="object-enrollment")


def _run_enrollment(root: Path, config: ObjectWatchConfig) -> dict:
    """Run without a ConsoleBackend or API principal in the worker thread."""
    from cvti.object_watch.embeddings import embed_examples
    from cvti.object_watch.runtime_config import load_configured_backend

    backend = load_configured_backend(config)
    written = embed_examples(root, backend)
    return {
        "ok": True,
        "written": written,
        "model": backend.name,
        "model_fingerprint": backend.fingerprint,
    }


class EnrollmentJobs:
    """A small process-local queue with at most one active job per site."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: "OrderedDict[str, dict]" = OrderedDict()
        self._active_by_site: dict[str, str] = {}

    def enqueue(self, root: str | Path, config: ObjectWatchConfig) -> dict:
        site = str(Path(root).resolve())
        with self._lock:
            active_id = self._active_by_site.get(site)
            if active_id is not None:
                active = self._jobs.get(active_id)
                if active is not None and active["status"] in {"queued", "running"}:
                    return {"job_id": active_id, "status": active["status"]}

            job_id = uuid.uuid4().hex
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "site": site,
                "created_at": time.time(),
            }
            self._active_by_site[site] = job_id
            self._prune_locked()
            _EXECUTOR.submit(self._work, job_id, Path(site), config)
            # The submission contract is deliberately stable even if the worker
            # starts before this thread gets to return.
            return {"job_id": job_id, "status": "queued"}

    def status(self, root: str | Path, job_id: str) -> dict:
        site = str(Path(root).resolve())
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None or job["site"] != site:
                raise KeyError(f"unknown object watch job: {job_id}")
            return self._public(job)

    def _work(self, job_id: str, root: Path, config: ObjectWatchConfig) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"
            job["started_at"] = time.time()
        try:
            result = _run_enrollment(root, config)
        except Exception as exc:  # noqa: BLE001 - failure is reported through status
            safe_error = redact_credentials(str(exc))[:300]
            safe_exc = RuntimeError(safe_error)
            log.error("object-watch enrollment job %s failed", job_id,
                      exc_info=(type(safe_exc), safe_exc, exc.__traceback__))
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.update(status="failed", error=safe_error, finished_at=time.time())
        else:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.update(status="completed", result=result, finished_at=time.time())
        finally:
            with self._lock:
                if self._active_by_site.get(str(root)) == job_id:
                    self._active_by_site.pop(str(root), None)
                self._prune_locked()

    @staticmethod
    def _public(job: dict) -> dict:
        out = {"job_id": job["job_id"], "status": job["status"]}
        if job["status"] == "completed":
            out.update(job.get("result") or {})
        elif job["status"] == "failed":
            out["error"] = job.get("error") or "object watch enrollment failed"
        return out

    def _prune_locked(self) -> None:
        while len(self._jobs) > _MAX_RETAINED_JOBS:
            removable = next(
                (key for key, value in self._jobs.items()
                 if value["status"] not in {"queued", "running"}),
                None,
            )
            if removable is None:
                break
            self._jobs.pop(removable, None)


JOBS = EnrollmentJobs()
