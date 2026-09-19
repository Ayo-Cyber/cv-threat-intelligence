"""Score labeled Chi KPI 3 object-watch scenarios."""

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
from typing import Any, Iterable


LABEL_COLUMNS = {
    "case_id",
    "clip_sha256",
    "clip_duration_s",
    "target_fps",
    "start_s",
    "end_s",
    "label",
    "object_ref",
    "expected_zone",
    "expected_state",
    "expected_object_id",
    "incident_id",
}
OBSERVATION_COLUMNS = {
    "case_id",
    "timestamp_s",
    "object_ref",
    "visible",
    "detected",
    "track_id",
    "object_id",
    "similarity",
    "zone",
    "bbox",
}
AUDIT_COLUMNS = {
    "candidate_id",
    "case_id",
    "timestamp_s",
    "object_id",
    "object_label",
    "state",
    "admission_status",
    "gate_status",
    "persisted_event_id",
}
ADMISSION_STATUSES = {"admitted", "deduplicated", "capacity_dropped"}
GATE_STATUSES = {"confirmed", "rejected", "unverified", "not_gated", "pending"}
NEGATIVE_STATES = {"negative", "none", "not_scored"}
FINAL_FRAME_EPSILON = 1e-9
MAX_OBSERVATION_TOLERANCE_S = 0.001


class InputError(ValueError):
    """Raised when labels, observations, or audit rows cannot be scored."""


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise InputError(f"{name} file does not exist: {path}")


def _read_csv(path: Path, name: str, required: set[str]) -> list[dict[str, str]]:
    _require_file(path, name)
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(required - set(reader.fieldnames or ()))
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


def _sha256_field(value: Any, source: str, field: str) -> str:
    result = _text(value, source, field).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise InputError(f"{source}: {field} must be a 64-character SHA-256")
    return result


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labels(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path, "labels", LABEL_COLUMNS)
    labels: list[dict[str, Any]] = []
    durations: dict[str, float] = {}
    target_rates: dict[str, float] = {}
    clip_hashes: dict[str, str] = {}
    incident_bounds: dict[tuple[str, str], tuple[float, float, str]] = {}
    for line, raw in enumerate(rows, 2):
        source = f"labels row {line}"
        case_id = _text(raw.get("case_id"), source, "case_id")
        clip_sha256 = _sha256_field(raw.get("clip_sha256"), source, "clip_sha256")
        previous_hash = clip_hashes.setdefault(case_id, clip_sha256)
        if previous_hash != clip_sha256:
            raise InputError(f"{source}: clip_sha256 conflicts within case {case_id}")
        duration = _number(raw.get("clip_duration_s"), source, "clip_duration_s")
        if duration <= 0:
            raise InputError(f"{source}: clip_duration_s must be greater than zero")
        previous_duration = durations.setdefault(case_id, duration)
        if not math.isclose(previous_duration, duration):
            raise InputError(f"{source}: clip_duration_s conflicts within case {case_id}")
        target_fps = _number(raw.get("target_fps"), source, "target_fps")
        if target_fps <= 0:
            raise InputError(f"{source}: target_fps must be greater than zero")
        previous_rate = target_rates.setdefault(case_id, target_fps)
        if not math.isclose(previous_rate, target_fps):
            raise InputError(f"{source}: target_fps conflicts within case {case_id}")
        start = _number(raw.get("start_s"), source, "start_s")
        end = _number(raw.get("end_s"), source, "end_s")
        if end <= start:
            raise InputError(f"{source}: interval must satisfy start_s < end_s")
        if end > duration and not math.isclose(
            end, duration, rel_tol=0.0, abs_tol=FINAL_FRAME_EPSILON
        ):
            raise InputError(f"{source}: end_s exceeds clip_duration_s")
        expected_state = _text(raw.get("expected_state"), source, "expected_state")
        label = {
            "case_id": case_id,
            "clip_sha256": clip_sha256,
            "clip_duration_s": duration,
            "target_fps": target_fps,
            "start_s": start,
            "end_s": end,
            "label": _text(raw.get("label"), source, "label"),
            "object_ref": _text(raw.get("object_ref"), source, "object_ref"),
            "expected_zone": str(raw.get("expected_zone") or "").strip(),
            "expected_state": expected_state,
            "expected_object_id": _text(
                raw.get("expected_object_id"), source, "expected_object_id"
            ),
            "incident_id": _text(raw.get("incident_id"), source, "incident_id"),
        }
        key = (case_id, label["incident_id"])
        signature = (start, end, expected_state)
        previous_signature = incident_bounds.setdefault(key, signature)
        if previous_signature != signature:
            raise InputError(
                "labels: rows sharing a case_id and incident_id must have "
                "identical intervals and expected_state"
            )
        labels.append(label)
    _validate_label_intervals(labels)
    return labels


def _validate_label_intervals(labels: list[dict[str, Any]]) -> None:
    by_object: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    positive_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        by_object[(label["case_id"], label["object_ref"])].append(label)
        if label["expected_state"] not in NEGATIVE_STATES:
            positive_by_case[label["case_id"]].append(label)
    for key, rows in by_object.items():
        ordered = sorted(rows, key=lambda row: (row["start_s"], row["end_s"]))
        for left, right in zip(ordered, ordered[1:]):
            if right["start_s"] < left["end_s"] and not math.isclose(
                right["start_s"], left["end_s"], rel_tol=0.0,
                abs_tol=FINAL_FRAME_EPSILON,
            ):
                raise InputError(
                    f"labels: overlapping intervals for {key[0]}:{key[1]}"
                )
    for case_id, rows in positive_by_case.items():
        ordered = sorted(rows, key=lambda row: (row["start_s"], row["end_s"]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                if right["start_s"] >= left["end_s"]:
                    break
                raise InputError(
                    f"labels: ambiguous positive object intervals for {case_id}: "
                    f"{left['incident_id']} overlaps {right['incident_id']}"
                )


def load_observations(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path, "observations", OBSERVATION_COLUMNS)
    observations = []
    for line, raw in enumerate(rows, 2):
        source = f"observations row {line}"
        visible = _boolean(raw.get("visible"), source, "visible")
        detected = _boolean(raw.get("detected"), source, "detected")
        track_id = str(raw.get("track_id") or "").strip()
        object_id = str(raw.get("object_id") or "").strip()
        if detected and not object_id:
            raise InputError(f"{source}: object_id is required when detected=true")
        if not detected and (track_id or object_id):
            raise InputError(
                f"{source}: track_id and object_id must be empty when detected=false"
            )
        observations.append({
            "case_id": _text(raw.get("case_id"), source, "case_id"),
            "timestamp_s": _number(raw.get("timestamp_s"), source, "timestamp_s"),
            "object_ref": _text(raw.get("object_ref"), source, "object_ref"),
            "visible": visible,
            "detected": detected,
            "track_id": track_id,
            "object_id": object_id,
            "similarity": _number(raw.get("similarity") or 0, source, "similarity"),
            "zone": str(raw.get("zone") or "").strip(),
            "bbox": str(raw.get("bbox") or "").strip(),
        })
    return observations


def _expected_observation_slots(
    labels: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    slots: list[dict[str, Any]] = []
    tolerances: dict[str, float] = {}
    for label in labels:
        fps = label["target_fps"]
        tolerance = min(MAX_OBSERVATION_TOLERANCE_S, 0.25 / fps)
        tolerances[label["case_id"]] = tolerance
        index = math.ceil((label["start_s"] - FINAL_FRAME_EPSILON) * fps)
        while True:
            timestamp = index / fps
            if timestamp >= label["end_s"] - FINAL_FRAME_EPSILON:
                break
            slots.append({
                "case_id": label["case_id"],
                "object_ref": label["object_ref"],
                "timestamp_s": timestamp,
                "label": label,
            })
            index += 1
        if math.isclose(
            label["end_s"], label["clip_duration_s"], rel_tol=0.0,
            abs_tol=FINAL_FRAME_EPSILON,
        ):
            slots.append({
                "case_id": label["case_id"],
                "object_ref": label["object_ref"],
                "timestamp_s": label["clip_duration_s"],
                "label": label,
            })
    return slots, tolerances


def _validate_observation_coverage(
    labels: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    slots, tolerances = _expected_observation_slots(labels)
    matched_slots: dict[int, dict[str, Any]] = {}
    labeled = []
    for observation in observations:
        tolerance = tolerances.get(observation["case_id"])
        matches = [] if tolerance is None else [
            (index, slot)
            for index, slot in enumerate(slots)
            if slot["case_id"] == observation["case_id"]
            and slot["object_ref"] == observation["object_ref"]
            and abs(slot["timestamp_s"] - observation["timestamp_s"]) <= tolerance
        ]
        if not matches:
            raise InputError(
                "observations: row does not match an expected object sample: "
                f"{observation['case_id']}:{observation['object_ref']} at "
                f"{observation['timestamp_s']}"
            )
        if len(matches) > 1:
            raise InputError(
                "observations: row ambiguously matches expected object samples: "
                f"{observation['case_id']}:{observation['object_ref']} at "
                f"{observation['timestamp_s']}"
            )
        slot_index, slot = matches[0]
        if slot_index in matched_slots:
            raise InputError(
                "observations: duplicate observation for expected object sample "
                f"{slot['case_id']}:{slot['object_ref']} at {slot['timestamp_s']}"
            )
        matched_slots[slot_index] = observation
        labeled.append((observation, slot["label"]))
    missing = [slot for index, slot in enumerate(slots) if index not in matched_slots]
    if missing:
        slot = missing[0]
        raise InputError(
            "observations: missing expected object sample "
            f"{slot['case_id']}:{slot['object_ref']} at {slot['timestamp_s']}"
        )
    return labeled, {
        "expected_object_samples": len(slots),
        "observed_object_samples": len(observations),
        "timestamp_tolerance_s": dict(sorted(tolerances.items())),
    }


def _validate_audit_rows(raw_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    seen_ids: set[str] = set()
    for line, raw in enumerate(raw_rows, 1):
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
            raise InputError(f"{source}: only confirmed rows may have persisted_event_id")
        if admission_status == "admitted" and gate_status not in {
            "confirmed", "rejected", "unverified"
        }:
            raise InputError(f"{source}: admitted rows require a completed gate status")
        candidates.append({
            "candidate_id": candidate_id,
            "case_id": _text(raw.get("case_id"), source, "case_id"),
            "timestamp_s": _number(raw.get("timestamp_s"), source, "timestamp_s"),
            "object_id": _text(raw.get("object_id"), source, "object_id"),
            "object_label": _text(raw.get("object_label"), source, "object_label"),
            "state": _text(raw.get("state"), source, "state"),
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
                    for row in con.execute("PRAGMA table_info(object_watch_audit)")
                }
                missing = sorted(AUDIT_COLUMNS - columns)
                if missing:
                    raise InputError(
                        "SQLite audit table object_watch_audit is missing "
                        f"required columns: {missing}"
                    )
                rows = [
                    dict(row)
                    for row in con.execute(
                        "SELECT candidate_id, case_id, timestamp_s, object_id, "
                        "object_label, state, admission_status, gate_status, "
                        "persisted_event_id FROM object_watch_audit ORDER BY rowid"
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
        math.isclose(timestamp, label["clip_duration_s"], rel_tol=0.0,
                     abs_tol=FINAL_FRAME_EPSILON)
        and math.isclose(label["end_s"], label["clip_duration_s"], rel_tol=0.0,
                         abs_tol=FINAL_FRAME_EPSILON)
    )


def _candidate_match(
    labels: list[dict[str, Any]], candidate: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    case_labels = [
        label for label in labels if label["case_id"] == candidate["case_id"]
    ]
    if not case_labels:
        raise InputError(
            f"audit: candidate {candidate['candidate_id']} has unknown case_id "
            f"{candidate['case_id']!r}"
        )
    duration = case_labels[0]["clip_duration_s"]
    if candidate["timestamp_s"] > duration and not math.isclose(
        candidate["timestamp_s"], duration, rel_tol=0.0,
        abs_tol=FINAL_FRAME_EPSILON,
    ):
        raise InputError(
            f"audit: candidate {candidate['candidate_id']} timestamp is outside "
            f"clip bounds [0,{duration}]"
        )
    matches = [
        label
        for label in case_labels
        if label["expected_state"] not in NEGATIVE_STATES
        and label["expected_state"] == candidate["state"]
        and label["expected_object_id"] == candidate["object_id"]
        and _matches_timestamp(label, candidate["timestamp_s"])
    ]
    if len(matches) > 1:
        raise InputError(
            "audit: candidate matched ambiguous object intervals; "
            f"{candidate['candidate_id']} matched {len(matches)}"
        )
    if not matches:
        return None, "false_match"
    return matches[0], "positive_interval"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_inputs(labels_path: Path, observations_path: Path, audit_path: Path) -> dict[str, Any]:
    labels = load_labels(Path(labels_path))
    observations = load_observations(Path(observations_path))
    candidates = load_audit_rows(Path(audit_path))
    labeled_observations, coverage = _validate_observation_coverage(
        labels, observations
    )

    visible = [
        (observation, label)
        for observation, label in labeled_observations
        if observation["visible"]
    ]
    correct = [
        observation for observation, label in visible
        if observation["detected"]
        and observation["object_id"] == label["expected_object_id"]
        and (not label["expected_zone"] or observation["zone"] == label["expected_zone"])
    ]
    positive_labels = [
        label for label in labels if label["expected_state"] not in NEGATIVE_STATES
    ]
    expected_by_state = Counter(label["expected_state"] for label in positive_labels)
    hits_by_incident: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    matched_by_state: Counter[str] = Counter()
    false_by_state: Counter[str] = Counter()
    candidate_rows = []
    for candidate in candidates:
        label, classification = _candidate_match(labels, candidate)
        row = {**candidate, "classification": classification}
        if label:
            row["incident_id"] = label["incident_id"]
            row["detection_delay_s"] = candidate["timestamp_s"] - label["start_s"]
            hits_by_incident[(label["case_id"], label["incident_id"])].append(row)
            matched_by_state[candidate["state"]] += 1
        else:
            row["incident_id"] = ""
            row["detection_delay_s"] = None
            false_by_state[candidate["state"]] += 1
        candidate_rows.append(row)

    hit_incidents = {
        key for key, rows in hits_by_incident.items() if rows
    }
    recall_by_state = {}
    precision_by_state = {}
    for state in sorted(set(expected_by_state) | set(matched_by_state) | set(false_by_state)):
        expected = expected_by_state[state]
        matched_incidents = len({
            key for key, rows in hits_by_incident.items()
            if rows and rows[0]["state"] == state
        })
        recall_by_state[state] = _ratio(matched_incidents, expected)
        precision_by_state[state] = _ratio(
            matched_by_state[state], matched_by_state[state] + false_by_state[state]
        )

    duplicate_candidates = sum(max(0, len(rows) - 1) for rows in hits_by_incident.values())
    duplicate_persisted_alerts = 0
    for rows in hits_by_incident.values():
        persisted = [row for row in rows if row["persisted_event_id"]]
        duplicate_persisted_alerts += max(0, len(persisted) - 1)
    delays = [
        min(row["detection_delay_s"] for row in rows if row["detection_delay_s"] is not None)
        for rows in hits_by_incident.values()
        if rows
    ]
    metrics = {
        "object_match_recall": _ratio(len(correct), len(visible)),
        "false_match_count": sum(false_by_state.values()),
        "event_recall_by_state": recall_by_state,
        "event_precision_by_state": precision_by_state,
        "duplicate_candidates": duplicate_candidates,
        "duplicate_persisted_alerts": duplicate_persisted_alerts,
        "detection_delay_s": delays,
        "gate_status_counts": dict(Counter(row["gate_status"] for row in candidates)),
    }
    return {
        "schema_version": 1,
        "input_hashes": {
            "labels_sha256": _sha256(Path(labels_path)),
            "observations_sha256": _sha256(Path(observations_path)),
            "audit_sha256": _sha256(Path(audit_path)),
        },
        "coverage": coverage,
        "metrics": metrics,
        "candidate_rows": candidate_rows,
        "positive_intervals": len(positive_labels),
        "positive_intervals_hit": len(hit_incidents),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = score_inputs(args.labels, args.observations, args.audit)
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
