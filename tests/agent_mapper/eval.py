"""Evaluate Agent Mapper output across one or more VLM providers.

Loads labels.json, runs agent_mapper on each clip per requested model, and writes a
CSV of results plus a console summary. Designed to import agent_mapper.py directly
so the eval exercises the same code path the CLI uses.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import agent_mapper as mapper  # noqa: E402

# Words that should NOT appear in scene_description or notes when the prompt is
# working — they indicate the model leaked threat semantics into the descriptive layer.
LEAKED_THREAT_TERMS = (
    "threat",
    "danger",
    "suspicious",
    "suspect",
    "intrud",
    "loiter",
    "theft",
    "thieving",
    "steal",
    "weapon",
    "armed",
    "firearm",
    "assault",
    "attack",
    "violence",
    "violent",
    "fight",
    "tailgat",
    "tamper",
    "abandoned",
    "crime",
    "criminal",
    "unauthor",
    "robber",
    "burglar",
    "trespass",
    "hostile",
)


@dataclass
class ClipLabel:
    clip_path: Path
    expected_environment_type: str
    acceptable_environment_types: list[str]
    notes: str


@dataclass
class EvalRow:
    run_id: str
    model: str
    clip_id: str
    expected_env: str
    predicted_env: str
    env_match: bool
    env_acceptable: bool
    valid_json: bool
    latency_s: float
    leaked_terms_count: int
    leaked_terms: str
    scene_description: str
    notes: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent Mapper VLM evaluation harness.")
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated list of model names to evaluate (e.g. 'qwen2.5vl:7b,gemma3:12b').",
    )
    parser.add_argument(
        "--provider",
        default="openai_compatible",
        choices=("mock", "anthropic", "openai_compatible"),
        help="Provider type. For local Ollama / vLLM / LM Studio, use openai_compatible.",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:11434/v1",
        help="Base URL for openai_compatible provider. Default is Ollama's local endpoint.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OLLAMA_API_KEY",
        help="Env var name holding API key. Ollama doesn't validate, so any non-empty value works.",
    )
    parser.add_argument(
        "--labels",
        default=str(Path(__file__).parent / "labels.json"),
        help="Path to labels.json.",
    )
    parser.add_argument(
        "--clips-dir",
        default=str(Path(__file__).parent / "clips"),
        help="Root directory containing clip files referenced by labels.json.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "results"),
        help="Where to write the results CSV.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="Frames to sample per clip before choosing a representative frame.",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=4.0,
        help="Fallback interval between samples for non-seekable sources.",
    )
    parser.add_argument(
        "--filter-env",
        default="",
        help="Optional: only evaluate clips whose expected_environment_type matches this value.",
    )
    parser.add_argument(
        "--max-zone-suggestions",
        type=int,
        default=4,
    )
    return parser.parse_args()


def load_labels(labels_path: Path, clips_root: Path, filter_env: str) -> list[ClipLabel]:
    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    clips: list[ClipLabel] = []
    for entry in raw.get("clips", []):
        clip_path = (clips_root.parent.parent / entry["clip_path"]).resolve()
        if not clip_path.is_absolute():
            clip_path = (REPO_ROOT / entry["clip_path"]).resolve()
        # If the path doesn't resolve under clips_root, fall back to interpreting it as repo-relative.
        if not clip_path.exists():
            clip_path = (REPO_ROOT / entry["clip_path"]).resolve()
        expected = entry["expected_environment_type"]
        if filter_env and expected != filter_env:
            continue
        clips.append(
            ClipLabel(
                clip_path=clip_path,
                expected_environment_type=expected,
                acceptable_environment_types=entry.get("acceptable_environment_types", [expected]),
                notes=entry.get("notes", ""),
            )
        )
    return clips


def detect_leaked_terms(text: str) -> list[str]:
    lowered = text.lower()
    return sorted({term for term in LEAKED_THREAT_TERMS if term in lowered})


def evaluate_clip(
    clip: ClipLabel,
    model: str,
    provider: str,
    api_base_url: str,
    api_key_env: str,
    sample_count: int,
    sample_interval_seconds: float,
    max_zone_suggestions: int,
    run_id: str,
    prompt_template: str,
    schema: dict,
) -> EvalRow:
    clip_id = clip.clip_path.stem
    if not clip.clip_path.exists():
        return EvalRow(
            run_id=run_id,
            model=model,
            clip_id=clip_id,
            expected_env=clip.expected_environment_type,
            predicted_env="",
            env_match=False,
            env_acceptable=False,
            valid_json=False,
            latency_s=0.0,
            leaked_terms_count=0,
            leaked_terms="",
            scene_description="",
            notes="",
            error=f"clip file not found: {clip.clip_path}",
        )

    source = str(clip.clip_path)
    source_type = mapper.detect_source_type(source)
    camera_id = mapper.build_camera_id(clip_id, source)
    try:
        samples = mapper.capture_sample_frames(
            source=source,
            source_type=source_type,
            sample_count=max(1, sample_count),
            sample_interval_seconds=max(0.0, sample_interval_seconds),
        )
        selected = mapper.choose_representative_frame(samples)
        frame_bytes = mapper.encode_frame_as_jpeg_bytes(selected.image)
        prompt = mapper.build_prompt(
            template=prompt_template,
            camera_id=camera_id,
            source_type=source_type,
            source_frame_path=str(clip.clip_path).replace("\\", "/"),
            max_zone_suggestions=max(1, max_zone_suggestions),
        )

        started = time.perf_counter()
        raw_response = mapper.call_provider(
            provider=provider,
            prompt=prompt,
            frame_bytes=frame_bytes,
            frame_shape=selected.image.shape,
            camera_id=camera_id,
            source_type=source_type,
            model=model,
            api_key_env=api_key_env,
            api_base_url=api_base_url,
        )
        latency_s = time.perf_counter() - started

        context = mapper.parse_and_validate_scene_context(
            raw_response=raw_response,
            schema=schema,
            camera_id=camera_id,
            source_type=source_type,
            source_frame_path=str(clip.clip_path).replace("\\", "/"),
        )
    except Exception as exc:  # noqa: BLE001
        return EvalRow(
            run_id=run_id,
            model=model,
            clip_id=clip_id,
            expected_env=clip.expected_environment_type,
            predicted_env="",
            env_match=False,
            env_acceptable=False,
            valid_json=False,
            latency_s=0.0,
            leaked_terms_count=0,
            leaked_terms="",
            scene_description="",
            notes="",
            error=f"{type(exc).__name__}: {exc}",
        )

    predicted_env = context.get("environment_type", "")
    scene_description = context.get("scene_description", "")
    context_notes = context.get("notes", "")
    leaked = detect_leaked_terms(f"{scene_description}\n{context_notes}")
    return EvalRow(
        run_id=run_id,
        model=model,
        clip_id=clip_id,
        expected_env=clip.expected_environment_type,
        predicted_env=predicted_env,
        env_match=predicted_env == clip.expected_environment_type,
        env_acceptable=predicted_env in clip.acceptable_environment_types,
        valid_json=True,
        latency_s=round(latency_s, 3),
        leaked_terms_count=len(leaked),
        leaked_terms=",".join(leaked),
        scene_description=scene_description,
        notes=context_notes,
        error="",
    )


def write_results_csv(rows: list[EvalRow], results_dir: Path, run_id: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"eval_{run_id}.csv"
    fields = list(EvalRow.__annotations__.keys())
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return out_path


def summarize(rows: list[EvalRow]) -> None:
    by_model: dict[str, list[EvalRow]] = {}
    for row in rows:
        by_model.setdefault(row.model, []).append(row)

    print("\n=== Agent Mapper VLM eval summary ===")
    for model, model_rows in by_model.items():
        total = len(model_rows)
        valid = sum(1 for row in model_rows if row.valid_json)
        env_correct = sum(1 for row in model_rows if row.env_match)
        env_acceptable = sum(1 for row in model_rows if row.env_acceptable)
        leaked = sum(1 for row in model_rows if row.leaked_terms_count > 0)
        avg_latency = (
            round(sum(row.latency_s for row in model_rows if row.valid_json) / max(1, valid), 3)
            if valid
            else 0.0
        )
        errors = sum(1 for row in model_rows if row.error)
        print(f"\n[{model}]")
        print(f"  clips evaluated      : {total}")
        print(f"  valid_json           : {valid}/{total}")
        print(f"  env exact match      : {env_correct}/{total}")
        print(f"  env acceptable match : {env_acceptable}/{total}")
        print(f"  clips with leak      : {leaked}/{total}")
        print(f"  avg latency (valid)  : {avg_latency}s")
        print(f"  hard errors          : {errors}")


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels)
    clips_dir = Path(args.clips_dir)
    results_dir = Path(args.results_dir)

    clips = load_labels(labels_path, clips_dir, args.filter_env.strip())
    if not clips:
        print("No clips matched the filter (or labels.json is empty). Add clips and entries first.")
        sys.exit(1)

    prompt_template = mapper.load_text_file(Path(mapper.DEFAULT_PROMPT_PATH))
    schema = mapper.load_schema(Path(mapper.DEFAULT_SCHEMA_PATH))

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    if not models:
        print("--models must include at least one model name.")
        sys.exit(1)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[EvalRow] = []
    for model in models:
        print(f"\n--- Evaluating model: {model} ({len(clips)} clip(s)) ---")
        for clip in clips:
            print(f"  [{model}] {clip.clip_path.name} ...", end=" ", flush=True)
            row = evaluate_clip(
                clip=clip,
                model=model,
                provider=args.provider,
                api_base_url=args.api_base_url,
                api_key_env=args.api_key_env,
                sample_count=args.sample_count,
                sample_interval_seconds=args.sample_interval_seconds,
                max_zone_suggestions=args.max_zone_suggestions,
                run_id=run_id,
                prompt_template=prompt_template,
                schema=schema,
            )
            rows.append(row)
            if row.error:
                print(f"ERROR ({row.error[:60]})")
            else:
                marker = "OK" if row.env_match else ("OK*" if row.env_acceptable else "MISS")
                leak = f" leak={row.leaked_terms}" if row.leaked_terms else ""
                print(f"{marker} pred={row.predicted_env} latency={row.latency_s}s{leak}")

    out_path = write_results_csv(rows, results_dir, run_id)
    summarize(rows)
    print(f"\nResults CSV written to: {out_path}")


if __name__ == "__main__":
    main()
