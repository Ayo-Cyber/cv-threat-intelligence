"""Model registry — track detector checkpoints and support rollback.

A retrain produces a new checkpoint; we record it (path, metrics, timestamp) and
mark it active. The pipeline can read the active path from here instead of a
hard-coded one, and a bad retrain is one `rollback()` away from the previous
known-good model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cvti.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_REGISTRY = "runs/models/registry.json"


@dataclass
class ModelRegistry:
    path: str = DEFAULT_REGISTRY

    def _load(self) -> dict:
        p = Path(self.path)
        if not p.exists():
            return {"active": None, "versions": []}
        try:
            return json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            log.warning("model registry unreadable; using baseline model", exc_info=True)
            return {"active": None, "versions": []}

    def _save(self, data: dict) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).write_text(json.dumps(data, indent=2))

    def register(self, model_path: str, *, created: float, metrics: dict | None = None,
                 activate: bool = True) -> dict:
        """Add a checkpoint. `created` is a caller-supplied timestamp (kept explicit
        so this stays deterministic/testable)."""
        data = self._load()
        version = len(data["versions"]) + 1
        entry = {"version": version, "path": str(model_path),
                 "created": created, "metrics": metrics or {}}
        data["versions"].append(entry)
        if activate:
            data["active"] = version
        self._save(data)
        return entry

    def versions(self) -> list[dict]:
        return self._load()["versions"]

    def active(self) -> dict | None:
        data = self._load()
        av = data.get("active")
        for v in data["versions"]:
            if v["version"] == av:
                return v
        return None

    def active_path(self) -> str | None:
        a = self.active()
        return a["path"] if a else None

    def rollback(self) -> dict | None:
        """Activate the version before the current active one."""
        data = self._load()
        vers = data["versions"]
        if len(vers) < 2:
            return None
        cur = data.get("active") or len(vers)
        prev = max(1, cur - 1)
        data["active"] = prev
        self._save(data)
        return next((v for v in vers if v["version"] == prev), None)

    def activate(self, version: int) -> dict | None:
        data = self._load()
        if not any(v["version"] == version for v in data["versions"]):
            return None
        data["active"] = version
        self._save(data)
        return self.active()
