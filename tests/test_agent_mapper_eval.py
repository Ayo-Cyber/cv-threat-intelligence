from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


EVAL_PATH = Path(__file__).parent / "agent_mapper/eval.py"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("agent_mapper_eval", EVAL_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(tmp_path: Path, clips: list[dict]) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"clips": clips}))
    return path


def test_eval_harness_imports_the_production_mapper() -> None:
    module = _load_eval_module()

    assert module.mapper.__name__ == "cvti.scene.agent_mapper"


def test_load_labels_rejects_environment_outside_runtime_vocabulary(tmp_path) -> None:
    module = _load_eval_module()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    labels = _write_manifest(tmp_path, [{
        "clip_path": str(clip),
        "expected_environment_type": "grocery_store",
        "acceptable_environment_types": ["grocery_store"],
    }])

    with pytest.raises(ValueError, match="unsupported expected_environment_type"):
        module.load_labels(labels, tmp_path, "")


def test_load_labels_fails_before_eval_when_selected_clip_is_missing(tmp_path) -> None:
    module = _load_eval_module()
    labels = _write_manifest(tmp_path, [{
        "clip_path": str(tmp_path / "missing.mp4"),
        "expected_environment_type": "retail_shop",
        "acceptable_environment_types": ["retail_shop"],
    }])

    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        module.load_labels(labels, tmp_path, "")


def test_load_labels_rejects_duplicate_clip_entries(tmp_path) -> None:
    module = _load_eval_module()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    entry = {
        "clip_path": str(clip),
        "expected_environment_type": "retail_shop",
        "acceptable_environment_types": ["retail_shop"],
    }
    labels = _write_manifest(tmp_path, [entry, entry])

    with pytest.raises(ValueError, match="duplicate clip_path"):
        module.load_labels(labels, tmp_path, "")


def test_load_labels_rejects_empty_clip_path(tmp_path) -> None:
    module = _load_eval_module()
    labels = _write_manifest(tmp_path, [{
        "clip_path": "  ",
        "expected_environment_type": "retail_shop",
        "acceptable_environment_types": ["retail_shop"],
    }])

    with pytest.raises(ValueError, match="clip_path.*empty"):
        module.load_labels(labels, tmp_path, "")


def test_load_labels_rejects_clip_with_wrong_checksum(tmp_path) -> None:
    module = _load_eval_module()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    labels = _write_manifest(tmp_path, [{
        "clip_path": str(clip),
        "sha256": hashlib.sha256(b"different").hexdigest(),
        "expected_environment_type": "retail_shop",
        "acceptable_environment_types": ["retail_shop"],
    }])

    with pytest.raises(ValueError, match="sha256 mismatch.*clip.mp4"):
        module.load_labels(labels, tmp_path, "")


def test_filter_ignores_missing_clips_from_other_environments(tmp_path) -> None:
    module = _load_eval_module()
    retail = tmp_path / "retail.mp4"
    retail.write_bytes(b"clip")
    labels = _write_manifest(tmp_path, [
        {
            "clip_path": str(retail),
            "expected_environment_type": "retail_shop",
            "acceptable_environment_types": ["retail_shop"],
        },
        {
            "clip_path": str(tmp_path / "missing.mp4"),
            "expected_environment_type": "parking_lot",
            "acceptable_environment_types": ["parking_lot"],
        },
    ])

    clips = module.load_labels(labels, tmp_path, "retail_shop")

    assert [clip.clip_path for clip in clips] == [retail]


def test_cli_reports_invalid_manifest_without_traceback(tmp_path) -> None:
    labels = _write_manifest(tmp_path, [{
        "clip_path": str(tmp_path / "missing.mp4"),
        "expected_environment_type": "retail_shop",
        "acceptable_environment_types": ["retail_shop"],
    }])

    result = subprocess.run(
        [
            sys.executable,
            str(EVAL_PATH),
            "--models",
            "mock",
            "--provider",
            "mock",
            "--labels",
            str(labels),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Invalid evaluation manifest" in result.stderr
    assert "missing.mp4" in result.stderr
    assert "Traceback" not in result.stderr
