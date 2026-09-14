"""Run a local object-watch proposal-provider bakeoff on a frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any


REQUIRED_PROVIDERS = {"generic-yolo-embeddings"}
OPTIONAL_PROVIDERS = {"yolo-world", "yoloe", "grounding-dino-offline"}
ALL_PROVIDERS = REQUIRED_PROVIDERS | OPTIONAL_PROVIDERS


class InputError(ValueError):
    """Raised when the bakeoff manifest or provider name is invalid."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InputError(f"manifest file does not exist: {path}")
    try:
        manifest = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise InputError("manifest must be a JSON object with a cases list")
    for index, case in enumerate(manifest["cases"], 1):
        if not isinstance(case, dict):
            raise InputError(f"manifest case {index} must be an object")
        for field in ("case_id", "clip_path", "clip_sha256"):
            if not str(case.get(field) or "").strip():
                raise InputError(f"manifest case {index} missing {field}")
    return manifest


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _latency_summary(latencies_ms: list[float] | None) -> dict[str, float]:
    values = [float(value) for value in (latencies_ms or [])]
    return {
        "median_ms": float(statistics.median(values)) if values else 0.0,
        "p95_ms": float(_p95(values)),
    }


def run_bakeoff(
    manifest_path: Path,
    provider: str,
    output_dir: Path,
    *,
    latencies_ms: list[float] | None = None,
    peak_mb: float | None = None,
) -> dict[str, Any]:
    if provider not in ALL_PROVIDERS:
        raise InputError(f"unknown provider {provider!r}")
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    digest = _sha256(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if provider in OPTIONAL_PROVIDERS:
        result = {
            "schema_version": 1,
            "provider": provider,
            "status": "unavailable",
            "unavailable_reason": (
                f"{provider} dependencies or local weights are not configured"
            ),
            "manifest_digest": digest,
            "model_versions": {},
            "latency": _latency_summary([]),
            "memory": {"peak_mb": None},
            "accuracy": None,
            "rows": [],
        }
    else:
        cases = manifest["cases"]
        values = latencies_ms if latencies_ms is not None else [0.0 for _ in cases]
        result = {
            "schema_version": 1,
            "provider": provider,
            "status": "completed",
            "manifest_digest": digest,
            "model_versions": {"provider": "generic-yolo-embeddings"},
            "latency": _latency_summary(values),
            "memory": {"peak_mb": peak_mb},
            "accuracy": None,
            "rows": [
                {
                    "case_id": case["case_id"],
                    "clip_sha256": case["clip_sha256"],
                    "status": "not_run_in_ci",
                    "proposals": [],
                }
                for case in cases
            ],
        }
    output = output_dir / f"bakeoff_{provider}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--output-dir", default=Path("runs/eval/object_watch"), type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_bakeoff(args.manifest, args.provider, args.output_dir)
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
