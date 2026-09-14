from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.score_chi_objects import InputError, score_inputs


LABEL_FIELDS = [
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
]
OBSERVATION_FIELDS = [
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
]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    labels = _write_csv(tmp_path / "labels.csv", LABEL_FIELDS, [
        {
            "case_id": "OBJ-P03",
            "clip_sha256": "a" * 64,
            "clip_duration_s": "4.0",
            "target_fps": "1.0",
            "start_s": "1.0",
            "end_s": "3.0",
            "label": "Chi product removed from storage",
            "object_ref": "carton-a",
            "expected_zone": "storage",
            "expected_state": "object_removed",
            "expected_object_id": "chi-carton",
            "incident_id": "removed-1",
        },
        {
            "case_id": "OBJ-N02",
            "clip_sha256": "b" * 64,
            "clip_duration_s": "3.0",
            "target_fps": "1.0",
            "start_s": "0.0",
            "end_s": "3.0",
            "label": "Chi product remains allowed in storage",
            "object_ref": "carton-b",
            "expected_zone": "storage",
            "expected_state": "negative",
            "expected_object_id": "chi-carton",
            "incident_id": "allowed-1",
        },
    ])
    observations = _write_csv(tmp_path / "observations.csv", OBSERVATION_FIELDS, [
        {
            "case_id": "OBJ-P03",
            "timestamp_s": "1.0",
            "object_ref": "carton-a",
            "visible": "true",
            "detected": "true",
            "track_id": "10",
            "object_id": "chi-carton",
            "similarity": "0.82",
            "zone": "storage",
            "bbox": "0,0,10,10",
        },
        {
            "case_id": "OBJ-P03",
            "timestamp_s": "2.0",
            "object_ref": "carton-a",
            "visible": "true",
            "detected": "true",
            "track_id": "10",
            "object_id": "chi-carton",
            "similarity": "0.81",
            "zone": "storage",
            "bbox": "0,0,10,10",
        },
        {
            "case_id": "OBJ-N02",
            "timestamp_s": "0.0",
            "object_ref": "carton-b",
            "visible": "true",
            "detected": "true",
            "track_id": "20",
            "object_id": "chi-carton",
            "similarity": "0.77",
            "zone": "storage",
            "bbox": "0,0,10,10",
        },
        {
            "case_id": "OBJ-N02",
            "timestamp_s": "1.0",
            "object_ref": "carton-b",
            "visible": "true",
            "detected": "true",
            "track_id": "20",
            "object_id": "chi-carton",
            "similarity": "0.77",
            "zone": "storage",
            "bbox": "0,0,10,10",
        },
        {
            "case_id": "OBJ-N02",
            "timestamp_s": "2.0",
            "object_ref": "carton-b",
            "visible": "true",
            "detected": "true",
            "track_id": "20",
            "object_id": "chi-carton",
            "similarity": "0.77",
            "zone": "storage",
            "bbox": "0,0,10,10",
        },
        {
            "case_id": "OBJ-N02",
            "timestamp_s": "3.0",
            "object_ref": "carton-b",
            "visible": "true",
            "detected": "true",
            "track_id": "20",
            "object_id": "chi-carton",
            "similarity": "0.77",
            "zone": "storage",
            "bbox": "0,0,10,10",
        },
    ])
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "rows": [
            {
                "candidate_id": "c1",
                "case_id": "OBJ-P03",
                "timestamp_s": 1.5,
                "object_id": "chi-carton",
                "object_label": "Chi carton",
                "state": "object_removed",
                "admission_status": "admitted",
                "gate_status": "confirmed",
                "persisted_event_id": "evt-1",
            }
        ]
    }))
    return labels, observations, audit


def test_missing_visible_observation_invalidates_score(tmp_path):
    labels, observations, audit = _valid_inputs(tmp_path)
    rows = list(csv.DictReader(observations.open()))
    _write_csv(observations, OBSERVATION_FIELDS, rows[:-1])

    with pytest.raises(InputError, match="missing expected object sample"):
        score_inputs(labels, observations, audit)


def test_candidate_outside_positive_interval_is_false_positive(tmp_path):
    labels, observations, audit = _valid_inputs(tmp_path)
    audit.write_text(json.dumps({
        "rows": [
            {
                "candidate_id": "c1",
                "case_id": "OBJ-P03",
                "timestamp_s": 3.5,
                "object_id": "chi-carton",
                "object_label": "Chi carton",
                "state": "object_removed",
                "admission_status": "admitted",
                "gate_status": "rejected",
                "persisted_event_id": "",
            }
        ]
    }))

    score = score_inputs(labels, observations, audit)

    assert score["metrics"]["false_match_count"] == 1
    assert score["metrics"]["event_precision_by_state"]["object_removed"] == 0.0


def test_duplicate_persisted_alert_is_reported(tmp_path):
    labels, observations, audit = _valid_inputs(tmp_path)
    audit.write_text(json.dumps({
        "rows": [
            {
                "candidate_id": "c1",
                "case_id": "OBJ-P03",
                "timestamp_s": 1.5,
                "object_id": "chi-carton",
                "object_label": "Chi carton",
                "state": "object_removed",
                "admission_status": "admitted",
                "gate_status": "confirmed",
                "persisted_event_id": "evt-1",
            },
            {
                "candidate_id": "c2",
                "case_id": "OBJ-P03",
                "timestamp_s": 2.0,
                "object_id": "chi-carton",
                "object_label": "Chi carton",
                "state": "object_removed",
                "admission_status": "admitted",
                "gate_status": "confirmed",
                "persisted_event_id": "evt-2",
            },
        ]
    }))

    score = score_inputs(labels, observations, audit)

    assert score["metrics"]["duplicate_candidates"] == 1
    assert score["metrics"]["duplicate_persisted_alerts"] == 1


def test_unknown_case_in_audit_is_malformed(tmp_path):
    labels, observations, audit = _valid_inputs(tmp_path)
    audit.write_text(json.dumps({
        "rows": [
            {
                "candidate_id": "bad",
                "case_id": "OBJ-MISSING",
                "timestamp_s": 1.0,
                "object_id": "chi-carton",
                "object_label": "Chi carton",
                "state": "object_removed",
                "admission_status": "admitted",
                "gate_status": "confirmed",
                "persisted_event_id": "evt-1",
            }
        ]
    }))

    with pytest.raises(InputError, match="unknown case_id"):
        score_inputs(labels, observations, audit)
