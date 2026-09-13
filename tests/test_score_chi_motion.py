from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tools.score_chi_motion import InputError, load_audit_rows, score_inputs


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "score_chi_motion.py"

LABEL_FIELDS = [
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
]
OBSERVATION_FIELDS = [
    "case_id",
    "timestamp_s",
    "person_ref",
    "visible",
    "detected",
    "track_id",
    "moving",
    "zone",
    "boxed",
]
AUDIT_FIELDS = [
    "candidate_id",
    "case_id",
    "timestamp_s",
    "admission_status",
    "gate_status",
    "persisted_event_id",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_perf(path: Path, rate: float) -> Path:
    path.write_text(json.dumps({
        "stages": {
            "detect_batch": {
                "engine": {
                    "rate_per_s": rate,
                    "count": 100,
                    "span_s": 20.0,
                }
            }
        }
    }))
    return path


def _valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path, list[Path], list[Path]]:
    labels = _write_csv(tmp_path / "labels.csv", LABEL_FIELDS, [
        {
            "case_id": "S4-P01",
            "clip_duration_s": "2.0",
            "start_s": "0.0",
            "end_s": "1.0",
            "label": "stationary",
            "person_ref": "p1",
            "expected_zone": "movement_permitted",
            "scenario4_expected": "not_moving",
            "scenario5_expected": "negative",
            "incident_id": "s4-stationary",
        },
        {
            "case_id": "S4-P01",
            "clip_duration_s": "2.0",
            "start_s": "1.0",
            "end_s": "2.0",
            "label": "moving",
            "person_ref": "p1",
            "expected_zone": "movement_permitted",
            "scenario4_expected": "moving",
            "scenario5_expected": "negative",
            "incident_id": "s4-moving",
        },
        {
            "case_id": "S5-P01",
            "clip_duration_s": "4.0",
            "start_s": "1.0",
            "end_s": "4.0",
            "label": "simultaneous_movement",
            "person_ref": "p1|p2",
            "expected_zone": "movement_permitted",
            "scenario4_expected": "not_scored",
            "scenario5_expected": "positive",
            "incident_id": "multi-1",
        },
        {
            "case_id": "S5-N01",
            "clip_duration_s": "4.0",
            "start_s": "0.0",
            "end_s": "4.0",
            "label": "stationary",
            "person_ref": "p3",
            "expected_zone": "movement_permitted",
            "scenario4_expected": "not_moving",
            "scenario5_expected": "negative",
            "incident_id": "crowd-1",
        },
    ])
    observations = _write_csv(tmp_path / "observations.csv", OBSERVATION_FIELDS, [
        {
            "case_id": "S4-P01",
            "timestamp_s": "0.5",
            "person_ref": "p1",
            "visible": "true",
            "detected": "true",
            "track_id": "10",
            "moving": "false",
            "zone": "movement_permitted",
            "boxed": "false",
        },
        {
            "case_id": "S4-P01",
            "timestamp_s": "1.0",
            "person_ref": "p1",
            "visible": "true",
            "detected": "true",
            "track_id": "10",
            "moving": "true",
            "zone": "movement_permitted",
            "boxed": "true",
        },
        {
            "case_id": "S4-P01",
            "timestamp_s": "2.0",
            "person_ref": "p1",
            "visible": "true",
            "detected": "true",
            "track_id": "11",
            "moving": "true",
            "zone": "movement_permitted",
            "boxed": "true",
        },
        {
            "case_id": "S5-P01",
            "timestamp_s": "2.0",
            "person_ref": "p1",
            "visible": "true",
            "detected": "false",
            "track_id": "",
            "moving": "false",
            "zone": "",
            "boxed": "false",
        },
    ])
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"rows": [
        {
            "candidate_id": "c1",
            "case_id": "S5-P01",
            "timestamp_s": 2.0,
            "admission_status": "admitted",
            "gate_status": "confirmed",
            "persisted_event_id": "e1",
        },
        {
            "candidate_id": "c2",
            "case_id": "S5-P01",
            "timestamp_s": 3.0,
            "admission_status": "admitted",
            "gate_status": "confirmed",
            "persisted_event_id": "e2",
        },
        {
            "candidate_id": "c3",
            "case_id": "S5-N01",
            "timestamp_s": 2.0,
            "admission_status": "admitted",
            "gate_status": "confirmed",
            "persisted_event_id": "e3",
        },
    ]}))
    hidden = [_write_perf(tmp_path / f"hidden-{index}.json", rate)
              for index, rate in enumerate((10.0, 12.0, 11.0), 1)]
    shown = [_write_perf(tmp_path / f"shown-{index}.json", rate)
             for index, rate in enumerate((9.0, 10.0, 8.0), 1)]
    return labels, observations, audit, hidden, shown


def test_scores_motion_acceptance_and_writes_a_retained_artifact(tmp_path: Path) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    output = tmp_path / "motion-score.json"

    result = score_inputs(labels, observations, audit, hidden, shown, output)

    assert result["person_detection_recall"] == {
        "detected_visible_person_frames": 3,
        "visible_person_frames": 4,
        "value": 0.75,
        "by_case": {
            "S4-P01": {
                "detected_visible_person_frames": 3,
                "visible_person_frames": 3,
                "value": 1.0,
            },
            "S5-P01": {
                "detected_visible_person_frames": 0,
                "visible_person_frames": 1,
                "value": 0.0,
            },
        },
    }
    assert result["id_switches"]["count"] == 1
    assert result["id_switches"]["by_person"] == {"S4-P01:p1": 1}
    assert result["scenario5"]["precision"] == {
        "matched_candidates": 2,
        "generated_candidates": 3,
        "value": pytest.approx(2 / 3),
    }
    assert result["scenario5"]["recall"] == {
        "hit_positive_incidents": 1,
        "positive_incidents": 1,
        "value": 1.0,
    }
    assert result["scenario5"]["duplicate_candidates"] == {
        "total": 1,
        "by_incident": {"S5-P01:multi-1": 1},
    }
    assert result["duplicate_alerts"]["total"] == 1
    assert result["duplicate_alerts"]["by_incident"] == {"S5-P01:multi-1": 1}
    assert result["detection_delay_s"]["by_incident"] == {
        "S5-P01:multi-1": 1.0
    }
    assert result["detection_delay_s"]["median"] == 1.0
    assert result["tracking_visibility_fps"] == {
        "hidden_runs": [10.0, 12.0, 11.0],
        "shown_runs": [9.0, 10.0, 8.0],
        "hidden_median": 11.0,
        "shown_median": 9.0,
        "delta_fps": -2.0,
        "impact_percent": pytest.approx(-18.1818181818),
    }
    assert result["scenario4"]["correct_observations"] == 3
    assert result["scenario4"]["scored_observations"] == 3
    assert set(result["inputs"]["sha256"]) == {
        str(labels),
        str(observations),
        str(audit),
        *(str(path) for path in hidden),
        *(str(path) for path in shown),
    }
    assert json.loads(output.read_text()) == result


def test_intervals_are_half_open_except_the_exact_final_frame(tmp_path: Path) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)

    result = score_inputs(
        labels, observations, audit, hidden, shown, tmp_path / "result.json"
    )

    scenario4_rows = result["scenario4"]["observations"]
    assert scenario4_rows[1]["timestamp_s"] == 1.0
    assert scenario4_rows[1]["scenario4_expected"] == "moving"
    assert scenario4_rows[2]["timestamp_s"] == 2.0
    assert scenario4_rows[2]["scenario4_expected"] == "moving"


def test_scenario4_requires_the_expected_box_state(tmp_path: Path) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    rows = list(csv.DictReader(observations.open()))
    rows[1]["boxed"] = "false"
    _write_csv(observations, OBSERVATION_FIELDS, rows)

    result = score_inputs(
        labels, observations, audit, hidden, shown, tmp_path / "result.json"
    )

    assert result["scenario4"]["correct_observations"] == 2
    assert result["scenario4"]["observations"][1]["observed_boxed"] is False


def test_loads_candidate_and_gate_audit_rows_from_sqlite(tmp_path: Path) -> None:
    database = tmp_path / "events.db"
    con = sqlite3.connect(database)
    con.execute(
        "CREATE TABLE motion_candidate_audit ("
        "candidate_id TEXT, case_id TEXT, timestamp_s REAL, "
        "admission_status TEXT, gate_status TEXT, persisted_event_id TEXT)"
    )
    con.execute(
        "INSERT INTO motion_candidate_audit VALUES (?, ?, ?, ?, ?, ?)",
        ("c1", "S5-P01", 2.0, "admitted", "confirmed", "e1"),
    )
    con.commit()
    con.close()

    assert load_audit_rows(database) == [{
        "candidate_id": "c1",
        "case_id": "S5-P01",
        "timestamp_s": 2.0,
        "admission_status": "admitted",
        "gate_status": "confirmed",
        "persisted_event_id": "e1",
    }]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_labels", "labels file does not exist"),
        ("bad_boolean", "observations row 2: visible must be true or false"),
        ("missing_column", "labels is missing required columns"),
        ("bad_perf", "detect_batch.engine.rate_per_s must be a positive number"),
        ("wrong_pair_count", "exactly 3 hidden and 3 shown performance reports"),
        ("duplicate_perf", "six distinct performance reports"),
    ],
)
def test_rejects_missing_or_malformed_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    if mutation == "missing_labels":
        labels = tmp_path / "absent.csv"
    elif mutation == "bad_boolean":
        rows = list(csv.DictReader(observations.open()))
        rows[0]["visible"] = "yes"
        _write_csv(observations, OBSERVATION_FIELDS, rows)
    elif mutation == "missing_column":
        rows = list(csv.DictReader(labels.open()))
        fields = LABEL_FIELDS[:-1]
        _write_csv(
            labels,
            fields,
            [{key: value for key, value in row.items() if key in fields} for row in rows],
        )
    elif mutation == "bad_perf":
        hidden[0].write_text(json.dumps({
            "stages": {"detect_batch": {"engine": {"rate_per_s": None}}}
        }))
    elif mutation == "wrong_pair_count":
        hidden = hidden[:2]
    elif mutation == "duplicate_perf":
        shown[0] = hidden[0]

    with pytest.raises(InputError, match=message):
        score_inputs(
            labels, observations, audit, hidden, shown, tmp_path / "result.json"
        )


def test_an_empty_candidate_audit_scores_zero_recall_without_inventing_precision(
    tmp_path: Path,
) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    audit.write_text(json.dumps({"rows": []}))

    result = score_inputs(
        labels, observations, audit, hidden, shown, tmp_path / "result.json"
    )

    assert result["scenario5"]["precision"] == {
        "matched_candidates": 0,
        "generated_candidates": 0,
        "value": None,
    }
    assert result["scenario5"]["recall"] == {
        "hit_positive_incidents": 0,
        "positive_incidents": 1,
        "value": 0.0,
    }
    assert result["detection_delay_s"]["by_incident"] == {
        "S5-P01:multi-1": None
    }


@pytest.mark.parametrize("document", [42, {"rows": ["not-an-object"]}])
def test_rejects_malformed_audit_json_structures(
    tmp_path: Path, document: object
) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(document))

    with pytest.raises(InputError, match="audit JSON|audit row 1 must be an object"):
        load_audit_rows(audit)


@pytest.mark.parametrize(
    ("admission_status", "gate_status", "persisted_event_id", "message"),
    [
        ("deduplicated", "confirmed", "", "must use gate_status=not_gated"),
        ("capacity_dropped", "not_gated", "e1", "must not have persisted_event_id"),
        ("admitted", "rejected", "e1", "only confirmed rows may have"),
        ("admitted", "pending", "", "admitted rows require a completed gate status"),
    ],
)
def test_rejects_impossible_candidate_audit_states(
    tmp_path: Path,
    admission_status: str,
    gate_status: str,
    persisted_event_id: str,
    message: str,
) -> None:
    _labels, _observations, audit, _hidden, _shown = _valid_inputs(tmp_path)
    audit.write_text(json.dumps({"rows": [{
        "candidate_id": "bad-1",
        "case_id": "S5-P01",
        "timestamp_s": 2.0,
        "admission_status": admission_status,
        "gate_status": gate_status,
        "persisted_event_id": persisted_event_id,
    }]}))

    with pytest.raises(InputError, match=message):
        load_audit_rows(audit)


def test_cli_fails_clearly_for_an_absent_input(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--labels",
            str(tmp_path / "missing.csv"),
            "--observations",
            str(tmp_path / "missing-observations.csv"),
            "--audit",
            str(tmp_path / "missing-audit.json"),
            "--hidden-perf",
            "a.json",
            "b.json",
            "c.json",
            "--shown-perf",
            "d.json",
            "e.json",
            "f.json",
            "--output",
            str(tmp_path / "result.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "labels file does not exist" in result.stderr
