from __future__ import annotations

import json
from pathlib import Path

from tools.object_model_bakeoff import run_bakeoff


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "cases": [
            {
                "case_id": "OBJ-P01",
                "clip_path": "clips/OBJ-P01.mp4",
                "clip_sha256": "a" * 64,
                "object_library": "object_library",
            }
        ],
    }, sort_keys=True))
    return path


def test_bakeoff_uses_same_frozen_cases_for_every_provider(tmp_path):
    manifest = _manifest(tmp_path)

    first = run_bakeoff(manifest, "generic-yolo-embeddings", tmp_path / "out")
    second = run_bakeoff(manifest, "generic-yolo-embeddings", tmp_path / "out")

    assert first["manifest_digest"] == second["manifest_digest"]
    assert first["rows"][0]["case_id"] == "OBJ-P01"
    assert (tmp_path / "out" / "bakeoff_generic-yolo-embeddings.json").is_file()


def test_bakeoff_marks_missing_provider_as_unavailable_not_failed_accuracy(tmp_path):
    result = run_bakeoff(_manifest(tmp_path), "yolo-world", tmp_path / "out")

    assert result["status"] == "unavailable"
    assert result["accuracy"] is None
    assert result["rows"] == []
    assert "unavailable_reason" in result


def test_latency_summary_reports_median_p95_and_memory_when_available(tmp_path):
    result = run_bakeoff(
        _manifest(tmp_path),
        "generic-yolo-embeddings",
        tmp_path / "out",
        latencies_ms=[10.0, 30.0, 20.0],
        peak_mb=256.5,
    )

    assert result["latency"] == {"median_ms": 20.0, "p95_ms": 30.0}
    assert result["memory"] == {"peak_mb": 256.5}
