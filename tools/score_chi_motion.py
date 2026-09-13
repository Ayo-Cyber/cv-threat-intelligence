"""Score labeled Chi motion scenarios and paired overlay performance runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


LABEL_COLUMNS = {
    "case_id",
    "clip_duration_s",
    "start_s",
    "end_s",
    "label",
    "person_ref",
    "expected_zone",
    "scenario4_expected",
    "scenario5_expected",
    "incident_id",
}
OBSERVATION_COLUMNS = {
    "case_id",
    "timestamp_s",
    "person_ref",
    "visible",
    "detected",
    "track_id",
    "moving",
    "zone",
    "boxed",
}
AUDIT_COLUMNS = {
    "candidate_id",
    "case_id",
    "timestamp_s",
    "admission_status",
    "gate_status",
    "persisted_event_id",
}
SCENARIO4_EXPECTED = {"moving", "not_moving", "not_scored"}
SCENARIO5_EXPECTED = {"positive", "negative", "not_scored"}
ADMISSION_STATUSES = {"admitted", "deduplicated", "capacity_dropped"}
GATE_STATUSES = {"confirmed", "rejected", "unverified", "not_gated", "pending"}
FINAL_FRAME_EPSILON = 1e-9


class InputError(ValueError):
    """Raised when an input cannot support an honest score."""


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise InputError(f"{name} file does not exist: {path}")


def _read_csv(path: Path, name: str, required: set[str]) -> list[dict[str, str]]:
    _require_file(path, name)
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = sorted(required - fields)
            if missing:
                raise InputError(f"{name} is missing required columns: {missing}")
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise InputError(f"{name} is not valid UTF-8: {path}") from exc
    if not rows:
        raise InputError(f"{name} has no data rows: {path}")
    return rows


def _text(value: Any, source: str, field: str) -> str:
    if value is None:
        raise InputError(f"{source}: {field} is required")
    result = str(value).strip()
    if not result:
        raise InputError(f"{source}: {field} is required")
    return result


def _number(value: Any, source: str, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise InputError(f"{source}: {field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{source}: {field} must be a finite number") from exc
    if not math.isfinite(result) or result < minimum:
        raise InputError(f"{source}: {field} must be a finite number >= {minimum}")
    return result


def _boolean(value: Any, source: str, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise InputError(f"{source}: {field} must be true or false")


def _enum(value: Any, source: str, field: str, allowed: set[str]) -> str:
    result = _text(value, source, field)
    if result not in allowed:
        raise InputError(
            f"{source}: {field} must be one of {sorted(allowed)}, got {result!r}"
        )
    return result


def load_labels(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path, "labels", LABEL_COLUMNS)
    labels: list[dict[str, Any]] = []
    durations: dict[str, float] = {}
    for line, raw in enumerate(rows, 2):
        source = f"labels row {line}"
        case_id = _text(raw.get("case_id"), source, "case_id")
        duration = _number(
            raw.get("clip_duration_s"), source, "clip_duration_s", minimum=0.0
        )
        if duration <= 0:
            raise InputError(f"{source}: clip_duration_s must be greater than zero")
        previous = durations.setdefault(case_id, duration)
        if not math.isclose(previous, duration):
            raise InputError(f"{source}: clip_duration_s conflicts within case {case_id}")
        start = _number(raw.get("start_s"), source, "start_s", minimum=0.0)
        end = _number(raw.get("end_s"), source, "end_s", minimum=0.0)
        if end <= start:
            raise InputError(f"{source}: interval must satisfy start_s < end_s")
        if end > duration and not math.isclose(end, duration):
            raise InputError(f"{source}: end_s exceeds clip_duration_s")
        person_ref = _text(raw.get("person_ref"), source, "person_ref")
        refs = tuple(ref.strip() for ref in person_ref.split("|") if ref.strip())
        if not refs or len(set(refs)) != len(refs):
            raise InputError(f"{source}: person_ref must contain unique non-empty IDs")
        labels.append({
            "case_id": case_id,
            "clip_duration_s": duration,
            "start_s": start,
            "end_s": end,
            "label": _text(raw.get("label"), source, "label"),
            "person_ref": person_ref,
            "person_refs": refs,
            "expected_zone": str(raw.get("expected_zone") or "").strip(),
            "scenario4_expected": _enum(
                raw.get("scenario4_expected"),
                source,
                "scenario4_expected",
                SCENARIO4_EXPECTED,
            ),
            "scenario5_expected": _enum(
                raw.get("scenario5_expected"),
                source,
                "scenario5_expected",
                SCENARIO5_EXPECTED,
            ),
            "incident_id": _text(raw.get("incident_id"), source, "incident_id"),
        })
    _validate_label_intervals(labels)
    return labels


def _validate_label_intervals(labels: list[dict[str, Any]]) -> None:
    by_person: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    incidents: dict[tuple[str, str], tuple[float, float, str]] = {}
    for label in labels:
        for person_ref in label["person_refs"]:
            by_person[(label["case_id"], person_ref)].append(label)
        incident_key = (label["case_id"], label["incident_id"])
        signature = (
            label["start_s"],
            label["end_s"],
            label["scenario5_expected"],
        )
        previous = incidents.setdefault(incident_key, signature)
        if previous != signature:
            raise InputError(
                "labels: rows sharing a case_id and incident_id must have identical "
                "intervals and scenario5_expected"
            )
    for key, rows in by_person.items():
        ordered = sorted(rows, key=lambda row: (row["start_s"], row["end_s"]))
        for left, right in zip(ordered, ordered[1:]):
            if right["start_s"] < left["end_s"] and not math.isclose(
                right["start_s"], left["end_s"]
            ):
                raise InputError(
                    f"labels: overlapping intervals for {key[0]}:{key[1]}"
                )


def load_observations(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path, "observations", OBSERVATION_COLUMNS)
    observations = []
    for line, raw in enumerate(rows, 2):
        source = f"observations row {line}"
        visible = _boolean(raw.get("visible"), source, "visible")
        detected = _boolean(raw.get("detected"), source, "detected")
        track_id = str(raw.get("track_id") or "").strip()
        if detected and not track_id:
            raise InputError(f"{source}: track_id is required when detected=true")
        if not detected and track_id:
            raise InputError(f"{source}: track_id must be empty when detected=false")
        observations.append({
            "case_id": _text(raw.get("case_id"), source, "case_id"),
            "timestamp_s": _number(
                raw.get("timestamp_s"), source, "timestamp_s", minimum=0.0
            ),
            "person_ref": _text(raw.get("person_ref"), source, "person_ref"),
            "visible": visible,
            "detected": detected,
            "track_id": track_id,
            "moving": _boolean(raw.get("moving"), source, "moving"),
            "zone": str(raw.get("zone") or "").strip(),
            "boxed": _boolean(raw.get("boxed"), source, "boxed"),
        })
    return observations


def _validate_audit_rows(raw_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(raw_rows)
    candidates = []
    seen_ids: set[str] = set()
    for line, raw in enumerate(rows, 1):
        source = f"audit row {line}"
        if not isinstance(raw, dict):
            raise InputError(f"{source} must be an object")
        missing = sorted(AUDIT_COLUMNS - set(raw))
        if missing:
            raise InputError(f"{source}: missing required fields: {missing}")
        candidate_id = _text(raw.get("candidate_id"), source, "candidate_id")
        if candidate_id in seen_ids:
            raise InputError(f"{source}: duplicate candidate_id {candidate_id!r}")
        seen_ids.add(candidate_id)
        admission_status = _enum(
            raw.get("admission_status"),
            source,
            "admission_status",
            ADMISSION_STATUSES,
        )
        gate_status = _enum(
            raw.get("gate_status"), source, "gate_status", GATE_STATUSES
        )
        persisted_event_id = str(raw.get("persisted_event_id") or "").strip()
        if admission_status != "admitted" and gate_status != "not_gated":
            raise InputError(
                f"{source}: {admission_status} rows must use gate_status=not_gated"
            )
        if admission_status != "admitted" and persisted_event_id:
            raise InputError(
                f"{source}: {admission_status} rows must not have persisted_event_id"
            )
        if gate_status != "confirmed" and persisted_event_id:
            raise InputError(
                f"{source}: only confirmed rows may have persisted_event_id"
            )
        if admission_status == "admitted" and gate_status not in {
            "confirmed", "rejected", "unverified"
        }:
            raise InputError(
                f"{source}: admitted rows require a completed gate status"
            )
        candidates.append({
            "candidate_id": candidate_id,
            "case_id": _text(raw.get("case_id"), source, "case_id"),
            "timestamp_s": _number(
                raw.get("timestamp_s"), source, "timestamp_s", minimum=0.0
            ),
            "admission_status": admission_status,
            "gate_status": gate_status,
            "persisted_event_id": persisted_event_id,
        })
    return candidates


def load_audit_rows(path: Path) -> list[dict[str, Any]]:
    _require_file(path, "audit")
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            try:
                columns = {
                    row["name"]
                    for row in con.execute("PRAGMA table_info(motion_candidate_audit)")
                }
                missing = sorted(AUDIT_COLUMNS - columns)
                if missing:
                    raise InputError(
                        "SQLite audit table motion_candidate_audit is missing "
                        f"required columns: {missing}"
                    )
                rows = [
                    dict(row)
                    for row in con.execute(
                        "SELECT candidate_id, case_id, timestamp_s, "
                        "admission_status, gate_status, persisted_event_id "
                        "FROM motion_candidate_audit ORDER BY rowid"
                    )
                ]
            finally:
                con.close()
        except sqlite3.Error as exc:
            raise InputError(f"could not read SQLite audit {path}: {exc}") from exc
        return _validate_audit_rows(rows)
    try:
        document = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"audit is not valid JSON: {path}: {exc}") from exc
    if isinstance(document, list):
        rows = document
    elif isinstance(document, dict):
        rows = document.get("rows")
    else:
        raise InputError("audit JSON must be a list or an object with a rows list")
    if not isinstance(rows, list):
        raise InputError("audit JSON must be a list or an object with a rows list")
    return _validate_audit_rows(rows)


def _matches_timestamp(label: dict[str, Any], timestamp: float) -> bool:
    if label["start_s"] <= timestamp < label["end_s"]:
        return True
    return (
        math.isclose(
            timestamp,
            label["clip_duration_s"],
            rel_tol=0.0,
            abs_tol=FINAL_FRAME_EPSILON,
        )
        and math.isclose(
            label["end_s"], label["clip_duration_s"], rel_tol=0.0,
            abs_tol=FINAL_FRAME_EPSILON,
        )
    )


def _observation_label(
    labels: list[dict[str, Any]], observation: dict[str, Any]
) -> dict[str, Any]:
    matches = [
        label
        for label in labels
        if label["case_id"] == observation["case_id"]
        and observation["person_ref"] in label["person_refs"]
        and _matches_timestamp(label, observation["timestamp_s"])
    ]
    if len(matches) != 1:
        raise InputError(
            "observations: each row must match exactly one labeled half-open "
            f"interval; {observation['case_id']}:{observation['person_ref']} at "
            f"{observation['timestamp_s']} matched {len(matches)}"
        )
    return matches[0]


def _candidate_label(
    labels: list[dict[str, Any]], candidate: dict[str, Any]
) -> dict[str, Any]:
    matches: dict[str, dict[str, Any]] = {}
    for label in labels:
        if (
            label["case_id"] == candidate["case_id"]
            and label["scenario5_expected"] != "not_scored"
            and _matches_timestamp(label, candidate["timestamp_s"])
        ):
            matches[label["incident_id"]] = label
    if len(matches) != 1:
        raise InputError(
            "audit: each candidate must match exactly one scored scenario-5 "
            f"incident; {candidate['candidate_id']} matched {len(matches)}"
        )
    return next(iter(matches.values()))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _load_perf_rate(path: Path, mode: str, index: int) -> float:
    name = f"{mode} performance report {index}"
    _require_file(path, name)
    try:
        report = json.loads(path.read_text())
        value = report["stages"]["detect_batch"]["engine"]["rate_per_s"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise InputError(
            f"{name}: missing stages.detect_batch.engine.rate_per_s"
        ) from exc
    try:
        rate = _number(value, name, "detect_batch.engine.rate_per_s", minimum=0.0)
    except InputError as exc:
        raise InputError(
            f"{name}: detect_batch.engine.rate_per_s must be a positive number"
        ) from exc
    if rate <= 0:
        raise InputError(
            f"{name}: detect_batch.engine.rate_per_s must be a positive number"
        )
    return rate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_inputs(
    labels_path: Path,
    observations_path: Path,
    audit_path: Path,
    hidden_perf_paths: list[Path],
    shown_perf_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    paths = (
        [Path(labels_path), Path(observations_path), Path(audit_path)]
        + [Path(path) for path in hidden_perf_paths]
        + [Path(path) for path in shown_perf_paths]
    )
    if len(hidden_perf_paths) != 3 or len(shown_perf_paths) != 3:
        raise InputError(
            "exactly 3 hidden and 3 shown performance reports are required"
        )
    performance_paths = [
        Path(path).resolve()
        for path in [*hidden_perf_paths, *shown_perf_paths]
    ]
    if len(set(performance_paths)) != 6:
        raise InputError("six distinct performance reports are required")

    labels = load_labels(Path(labels_path))
    observations = load_observations(Path(observations_path))
    candidates = load_audit_rows(Path(audit_path))
    labeled_observations = [
        (observation, _observation_label(labels, observation))
        for observation in observations
    ]
    labeled_candidates = [
        (candidate, _candidate_label(labels, candidate))
        for candidate in candidates
    ]

    visible = [row for row, _label in labeled_observations if row["visible"]]
    detected_visible = [row for row in visible if row["detected"]]
    visible_by_case = Counter(row["case_id"] for row in visible)
    detected_by_case = Counter(row["case_id"] for row in detected_visible)
    recall_by_case = {
        case_id: {
            "detected_visible_person_frames": detected_by_case[case_id],
            "visible_person_frames": count,
            "value": _ratio(detected_by_case[case_id], count),
        }
        for case_id, count in sorted(visible_by_case.items())
    }

    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in detected_visible:
        by_person[f"{observation['case_id']}:{observation['person_ref']}"].append(
            observation
        )
    switch_counts: dict[str, int] = {}
    for key, rows in by_person.items():
        track_ids = [
            row["track_id"]
            for row in sorted(rows, key=lambda row: row["timestamp_s"])
        ]
        count = sum(left != right for left, right in zip(track_ids, track_ids[1:]))
        if count:
            switch_counts[key] = count

    scenario4_rows = []
    for observation, label in labeled_observations:
        expected = label["scenario4_expected"]
        if expected == "not_scored":
            continue
        expected_zone = label["expected_zone"]
        correct = observation["detected"] and (
            (
                expected == "moving"
                and observation["moving"]
                and (not expected_zone or observation["zone"] == expected_zone)
                and observation["boxed"]
            )
            or (
                expected == "not_moving"
                and not observation["moving"]
                and not observation["boxed"]
            )
        )
        scenario4_rows.append({
            "case_id": observation["case_id"],
            "timestamp_s": observation["timestamp_s"],
            "person_ref": observation["person_ref"],
            "scenario4_expected": expected,
            "observed_detected": observation["detected"],
            "observed_moving": observation["moving"],
            "observed_zone": observation["zone"],
            "observed_boxed": observation["boxed"],
            "correct": correct,
        })

    incidents: dict[tuple[str, str], dict[str, Any]] = {}
    for label in labels:
        if label["scenario5_expected"] == "not_scored":
            continue
        key = (label["case_id"], label["incident_id"])
        incidents.setdefault(key, label)
    positive_incidents = {
        key: label
        for key, label in incidents.items()
        if label["scenario5_expected"] == "positive"
    }
    matched_positive = [
        (candidate, label)
        for candidate, label in labeled_candidates
        if label["scenario5_expected"] == "positive"
    ]
    candidate_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate, label in matched_positive:
        candidate_groups[(label["case_id"], label["incident_id"])].append(candidate)

    delay_by_incident: dict[str, float | None] = {}
    duplicate_candidates: dict[str, int] = {}
    duplicate_alerts: dict[str, int] = {}
    for key, label in positive_incidents.items():
        incident_candidates = candidate_groups.get(key, [])
        name = f"{label['case_id']}:{label['incident_id']}"
        delay_by_incident[name] = (
            min(row["timestamp_s"] for row in incident_candidates) - label["start_s"]
            if incident_candidates else None
        )
        candidate_duplicates = max(0, len(incident_candidates) - 1)
        if candidate_duplicates:
            duplicate_candidates[name] = candidate_duplicates
        persisted = {
            row["persisted_event_id"]
            for row in incident_candidates
            if row["persisted_event_id"]
        }
        duplicates = max(0, len(persisted) - 1)
        if duplicates:
            duplicate_alerts[name] = duplicates

    delays = [value for value in delay_by_incident.values() if value is not None]
    hidden_rates = [
        _load_perf_rate(Path(path), "hidden", index)
        for index, path in enumerate(hidden_perf_paths, 1)
    ]
    shown_rates = [
        _load_perf_rate(Path(path), "shown", index)
        for index, path in enumerate(shown_perf_paths, 1)
    ]
    hidden_median = median(hidden_rates)
    shown_median = median(shown_rates)
    delta = shown_median - hidden_median

    result = {
        "schema_version": 1,
        "interval_semantics": (
            "[start_s,end_s), except timestamp_s == clip_duration_s belongs "
            "to the final interval ending at clip_duration_s"
        ),
        "inputs": {
            "sha256": {str(path): _sha256(path) for path in paths},
        },
        "person_detection_recall": {
            "detected_visible_person_frames": len(detected_visible),
            "visible_person_frames": len(visible),
            "value": _ratio(len(detected_visible), len(visible)),
            "by_case": recall_by_case,
        },
        "id_switches": {
            "count": sum(switch_counts.values()),
            "by_person": dict(sorted(switch_counts.items())),
        },
        "scenario4": {
            "correct_observations": sum(row["correct"] for row in scenario4_rows),
            "scored_observations": len(scenario4_rows),
            "observations": scenario4_rows,
        },
        "scenario5": {
            "precision": {
                "matched_candidates": len(matched_positive),
                "generated_candidates": len(candidates),
                "value": _ratio(len(matched_positive), len(candidates)),
            },
            "recall": {
                "hit_positive_incidents": sum(
                    bool(candidate_groups.get(key)) for key in positive_incidents
                ),
                "positive_incidents": len(positive_incidents),
                "value": _ratio(
                    sum(bool(candidate_groups.get(key)) for key in positive_incidents),
                    len(positive_incidents),
                ),
            },
            "admission_status_counts": dict(
                sorted(Counter(row["admission_status"] for row in candidates).items())
            ),
            "gate_status_counts": dict(
                sorted(Counter(row["gate_status"] for row in candidates).items())
            ),
            "duplicate_candidates": {
                "total": sum(duplicate_candidates.values()),
                "by_incident": dict(sorted(duplicate_candidates.items())),
            },
        },
        "duplicate_alerts": {
            "total": sum(duplicate_alerts.values()),
            "by_incident": dict(sorted(duplicate_alerts.items())),
        },
        "detection_delay_s": {
            "by_incident": dict(sorted(delay_by_incident.items())),
            "median": median(delays) if delays else None,
        },
        "tracking_visibility_fps": {
            "hidden_runs": hidden_rates,
            "shown_runs": shown_rates,
            "hidden_median": hidden_median,
            "shown_median": shown_median,
            "delta_fps": delta,
            "impact_percent": delta / hidden_median * 100.0,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument(
        "--audit",
        type=Path,
        required=True,
        help="Exported JSON rows or SQLite with motion_candidate_audit table.",
    )
    parser.add_argument("--hidden-perf", type=Path, nargs=3, required=True)
    parser.add_argument("--shown-perf", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = score_inputs(
            args.labels,
            args.observations,
            args.audit,
            args.hidden_perf,
            args.shown_perf,
            args.output,
        )
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
