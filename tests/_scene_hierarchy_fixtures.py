from __future__ import annotations

from pathlib import Path
import threading
import time

import numpy as np

from cvti.scene.agent_mapper import MappingResult


FIXED_NOW = "2026-09-01T12:00:00Z"


def camera_context(
    camera_id: str = "cam1",
    site_type: str = "manufacturing_plant",
    area_id: str = "production",
    area_type: str = "production_floor",
    confidence: float = 0.9,
) -> dict:
    return {
        "camera_id": camera_id,
        "source_type": "video_file",
        "environment_type": area_type,
        "scene_description": f"View from {camera_id}.",
        "expected_actors": ["staff"],
        "zones": [],
        "confidence": confidence,
        "generated_at": FIXED_NOW,
        "source_frame_path": f"context/{camera_id}/source_frame.jpg",
        "notes": "",
        "area_id": area_id,
        "site_type_candidate": site_type,
        "area_type_candidate": area_type,
        "view_description": f"Independent view from {camera_id}.",
    }


def site_context(site_type: str = "manufacturing_plant") -> dict:
    return {
        "site_id": "test_site",
        "site_type": site_type,
        "site_description": "A test deployment.",
        "confidence": 0.9,
        "evidence_area_ids": ["production"],
        "generated_at": FIXED_NOW,
    }


def area_context(
    area_type: str = "production_floor", area_id: str = "production"
) -> dict:
    return {
        "area_id": area_id,
        "name": "Production",
        "site_type": "manufacturing_plant",
        "area_type": area_type,
        "area_description": "A shared test area.",
        "expected_actors": ["staff"],
        "confidence": 0.9,
        "evidence_camera_ids": ["cam1"],
        "conflicts": [],
        "generated_at": FIXED_NOW,
    }


def write_site(tmp_path: Path, camera_ids: list[str], grouped: bool = True) -> Path:
    import json

    path = tmp_path / "site.json"
    areas = [{"id": "production", "name": "Production"}] if grouped else []
    cameras = [
        {
            "id": camera_id,
            "source": str(tmp_path / f"{camera_id}.mp4"),
            **({"area_id": "production"} if grouped else {}),
        }
        for camera_id in camera_ids
    ]
    for camera in cameras:
        Path(camera["source"]).write_bytes(b"clip")
    path.write_text(json.dumps({"areas": areas, "cameras": cameras}, indent=2))
    return path


def camera_observation(
    camera_id: str, site_type: str, area_type: str, confidence: float
) -> dict:
    return camera_context(
        camera_id, site_type, "shared", area_type, confidence
    )


def area(area_type: str, area_id: str = "area1") -> dict:
    value = area_context(area_type, area_id)
    value["site_type"] = "unknown"
    return value


def site_with_areas(area_count: int, cameras_per_area: int) -> dict:
    areas = [{"id": f"area_{i}", "name": f"Area {i}"} for i in range(area_count)]
    cameras = [
        {"id": f"cam_{a}_{c}", "source": f"clip_{a}_{c}.mp4", "area_id": f"area_{a}"}
        for a in range(area_count) for c in range(cameras_per_area)
    ]
    return {"site_id": "scale_test", "areas": areas, "cameras": cameras}


class DeterministicMapper:
    def __init__(self, delay: float = 0) -> None:
        self.delay = delay
        self.calls: list[str] = []
        self.hints_seen: list[dict] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def map_result(self, source, camera_id, sample_count=3,
                   source_frame_path="", operator_hints=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls.append(camera_id)
            self.hints_seen.append(dict(operator_hints or {}))
            context = camera_context(camera_id)
            context["source_frame_path"] = source_frame_path
            return MappingResult(
                context, np.zeros((40, 60, 3), dtype=np.uint8), "{}"
            )
        finally:
            with self.lock:
                self.active -= 1
