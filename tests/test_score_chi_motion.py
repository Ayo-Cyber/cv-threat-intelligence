from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tools.score_chi_motion import (
    InputError,
    _expected_observation_slots,
    load_audit_rows,
    score_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "score_chi_motion.py"

LABEL_FIELDS = [
    "case_id",
    "clip_sha256",
    "clip_duration_s",
    "target_fps",
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


def _write_perf(path: Path, rate: float, *, mode: str, pair_id: str) -> Path:
    capture = path.with_suffix(".mjpeg")
    capture.write_bytes(f"{pair_id}:{mode}:captured-mjpeg".encode())
    capture_sha256 = hashlib.sha256(capture.read_bytes()).hexdigest()
    path.write_text(json.dumps({
        "chi_motion_performance": {
            "schema_version": 1,
            "case_id": "S5-P01",
            "clip_sha256": "a" * 64,
            "pair_id": pair_id,
            "run_id": f"{pair_id}-{mode}",
            "tracking_mode": mode,
            "tracking_query": 0 if mode == "hidden" else 1,
            "config_sha256": "b" * 64,
            "sample_duration_s": 20.0,
            "sample_count": 100,
            "capture_path": capture.name,
            "capture_sha256": capture_sha256,
        },
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


def _mutate_perf(path: Path, **metadata: object) -> None:
    report = json.loads(path.read_text())
    report["chi_motion_performance"].update(metadata)
    path.write_text(json.dumps(report))


def _valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path, list[Path], list[Path]]:
    labels = _write_csv(tmp_path / "labels.csv", LABEL_FIELDS, [
        {
            "case_id": "S4-P01",
            "clip_sha256": "c" * 64,
            "clip_duration_s": "2.0",
            "target_fps": "1.0",
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
            "clip_sha256": "c" * 64,
            "clip_duration_s": "2.0",
            "target_fps": "1.0",
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
            "clip_sha256": "a" * 64,
            "clip_duration_s": "4.0",
            "target_fps": "1.0",
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
            "clip_sha256": "d" * 64,
            "clip_duration_s": "4.0",
            "target_fps": "1.0",
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
    observations_rows = [
        {
            "case_id": "S4-P01",
            "timestamp_s": "0.0",
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
    ]
    for timestamp in (1.0, 2.0, 3.0, 4.0):
        for person_ref, track_id in (("p1", "20"), ("p2", "21")):
            if timestamp == 2.0 and person_ref == "p1":
                continue
            observations_rows.append({
                "case_id": "S5-P01",
                "timestamp_s": str(timestamp),
                "person_ref": person_ref,
                "visible": "true",
                "detected": "true",
                "track_id": track_id,
                "moving": "true",
                "zone": "movement_permitted",
                "boxed": "true",
            })
    for timestamp in (0.0, 1.0, 2.0, 3.0, 4.0):
        observations_rows.append({
            "case_id": "S5-N01",
            "timestamp_s": str(timestamp),
            "person_ref": "p3",
            "visible": "true",
            "detected": "true",
            "track_id": "30",
            "moving": "false",
            "zone": "movement_permitted",
            "boxed": "false",
        })
    observations = _write_csv(
        tmp_path / "observations.csv", OBSERVATION_FIELDS, observations_rows
    )
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
    hidden = [
        _write_perf(
            tmp_path / f"hidden-{index}.json", rate,
            mode="hidden", pair_id=f"pair-{index}",
        )
        for index, rate in enumerate((10.0, 12.0, 11.0), 1)
    ]
    shown = [
        _write_perf(
            tmp_path / f"shown-{index}.json", rate,
            mode="shown", pair_id=f"pair-{index}",
        )
        for index, rate in enumerate((9.0, 10.0, 8.0), 1)
    ]
    return labels, observations, audit, hidden, shown


def test_scores_motion_acceptance_and_writes_a_retained_artifact(tmp_path: Path) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    output = tmp_path / "motion-score.json"

    result = score_inputs(labels, observations, audit, hidden, shown, output)

    assert result["person_detection_recall"] == {
        "detected_visible_person_frames": 15,
        "visible_person_frames": 16,
        "value": 15 / 16,
        "by_case": {
            "S4-P01": {
                "detected_visible_person_frames": 3,
                "visible_person_frames": 3,
                "value": 1.0,
            },
            "S5-P01": {
                "detected_visible_person_frames": 7,
                "visible_person_frames": 8,
                "value": 7 / 8,
            },
            "S5-N01": {
                "detected_visible_person_frames": 5,
                "visible_person_frames": 5,
                "value": 1.0,
            },
        },
    }
    assert result["observation_coverage"] == {
        "expected_person_samples": 16,
        "observed_person_samples": 16,
        "timestamp_tolerance_s": {"S4-P01": 0.001, "S5-N01": 0.001,
                                  "S5-P01": 0.001},
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
    performance = result["tracking_visibility_fps"]
    assert performance["hidden_runs"] == [10.0, 12.0, 11.0]
    assert performance["shown_runs"] == [9.0, 10.0, 8.0]
    assert performance["hidden_median"] == 11.0
    assert performance["shown_median"] == 9.0
    assert performance["delta_fps"] == -2.0
    assert performance["impact_percent"] == pytest.approx(-18.1818181818)
    assert [pair["pair_id"] for pair in performance["pairs"]] == [
        "pair-1", "pair-2", "pair-3"
    ]
    assert performance["pairs"][0]["hidden"]["tracking_mode"] == "hidden"
    assert performance["pairs"][0]["shown"]["tracking_mode"] == "shown"
    assert result["scenario4"]["correct_observations"] == 8
    assert result["scenario4"]["scored_observations"] == 8
    captures = [path.with_suffix(".mjpeg") for path in [*hidden, *shown]]
    assert set(result["inputs"]["sha256"]) == {
        str(labels),
        str(observations),
        str(audit),
        *(str(path) for path in hidden),
        *(str(path) for path in shown),
        *(str(path) for path in captures),
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

    assert result["scenario4"]["correct_observations"] == 7
    assert result["scenario4"]["observations"][1]["observed_boxed"] is False


def test_out_of_window_candidate_is_a_false_positive_not_an_input_error(
    tmp_path: Path,
) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    document = json.loads(audit.read_text())
    document["rows"].append({
        "candidate_id": "outside-positive-window",
        "case_id": "S5-P01",
        "timestamp_s": 0.5,
        "admission_status": "admitted",
        "gate_status": "rejected",
        "persisted_event_id": "",
    })
    audit.write_text(json.dumps(document))

    result = score_inputs(
        labels, observations, audit, hidden, shown, tmp_path / "result.json"
    )

    assert result["scenario5"]["precision"] == {
        "matched_candidates": 2,
        "generated_candidates": 4,
        "value": 0.5,
    }
    row = next(
        row for row in result["scenario5"]["candidate_results"]
        if row["candidate_id"] == "outside-positive-window"
    )
    assert row["classification"] == "false_positive"
    assert row["match"] == "out_of_window"
    assert row["incident_id"] is None


def test_candidate_case_must_exist_and_timestamp_must_be_inside_clip(
    tmp_path: Path,
) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    document = json.loads(audit.read_text())
    document["rows"][0]["case_id"] = "UNKNOWN"
    audit.write_text(json.dumps(document))
    with pytest.raises(InputError, match="unknown case_id"):
        score_inputs(
            labels, observations, audit, hidden, shown, tmp_path / "result.json"
        )

    document["rows"][0]["case_id"] = "S5-P01"
    document["rows"][0]["timestamp_s"] = 4.1
    audit.write_text(json.dumps(document))
    with pytest.raises(InputError, match="outside clip bounds"):
        score_inputs(
            labels, observations, audit, hidden, shown, tmp_path / "result.json"
        )


def test_ambiguous_scenario5_overlap_remains_an_error(tmp_path: Path) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    label_rows = list(csv.DictReader(labels.open()))
    label_rows.append({
        **label_rows[2],
        "person_ref": "p4",
        "incident_id": "multi-overlap",
    })
    _write_csv(labels, LABEL_FIELDS, label_rows)
    observation_rows = list(csv.DictReader(observations.open()))
    for timestamp in (1.0, 2.0, 3.0, 4.0):
        observation_rows.append({
            **observation_rows[3],
            "timestamp_s": str(timestamp),
            "person_ref": "p4",
            "track_id": "44",
            "detected": "true",
        })
    _write_csv(observations, OBSERVATION_FIELDS, observation_rows)

    with pytest.raises(InputError, match="ambiguous scenario-5 intervals"):
        score_inputs(
            labels, observations, audit, hidden, shown, tmp_path / "result.json"
        )


def test_missing_expected_person_sample_is_rejected(tmp_path: Path) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    rows = list(csv.DictReader(observations.open()))
    rows = [
        row for row in rows
        if not (
            row["case_id"] == "S5-P01"
            and row["person_ref"] == "p2"
            and row["timestamp_s"] == "3.0"
        )
    ]
    _write_csv(observations, OBSERVATION_FIELDS, rows)

    with pytest.raises(InputError, match="missing expected person sample.*S5-P01:p2"):
        score_inputs(
            labels, observations, audit, hidden, shown, tmp_path / "result.json"
        )


def test_observation_timestamp_tolerance_is_one_millisecond(tmp_path: Path) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    rows = list(csv.DictReader(observations.open()))
    rows[1]["timestamp_s"] = "1.0009"
    _write_csv(observations, OBSERVATION_FIELDS, rows)
    score_inputs(labels, observations, audit, hidden, shown, tmp_path / "ok.json")

    rows[1]["timestamp_s"] = "1.0011"
    _write_csv(observations, OBSERVATION_FIELDS, rows)
    with pytest.raises(InputError, match="does not match an expected sample"):
        score_inputs(
            labels, observations, audit, hidden, shown, tmp_path / "bad.json"
        )


def test_expected_samples_are_anchored_to_the_clip_timeline() -> None:
    label = {
        "case_id": "S5-P01",
        "clip_duration_s": 1.0,
        "target_fps": 5.0,
        "start_s": 0.15,
        "end_s": 0.65,
        "person_refs": ("p1",),
    }

    slots, _tolerances = _expected_observation_slots([label])

    assert [slot["timestamp_s"] for slot in slots] == pytest.approx([0.2, 0.4, 0.6])


def test_duplicate_observation_for_expected_sample_is_rejected(tmp_path: Path) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    rows = list(csv.DictReader(observations.open()))
    rows.append(dict(rows[0]))
    _write_csv(observations, OBSERVATION_FIELDS, rows)

    with pytest.raises(InputError, match="duplicate observation for expected sample"):
        score_inputs(
            labels, observations, audit, hidden, shown, tmp_path / "result.json"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("swapped_mode", "declares tracking_mode=shown.*expected hidden"),
        ("swapped_query", "tracking_query does not prove hidden mode"),
        ("pair_id", "hidden/shown pair IDs do not match"),
        ("run_id", "run_id values must be unique"),
        ("case_id", "performance reports must share case_id"),
        ("clip", "performance reports must share clip_sha256"),
        ("config", "performance reports must share config_sha256"),
        ("label_clip", "clip_sha256 does not match labels"),
        ("sample_count", "sample_count does not match"),
        ("sample_duration", "sample_duration_s does not match"),
        ("paired_sample_count", "paired reports must have equal sample_count"),
        ("paired_sample_duration", "paired reports must have equal sample_duration_s"),
        ("schema_version", "schema_version must be 1"),
        ("capture_hash", "capture_sha256 does not match"),
    ],
)
def test_rejects_unpaired_or_unproven_performance_reports(
    tmp_path: Path, mutation: str, message: str
) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)
    if mutation == "swapped_mode":
        _mutate_perf(hidden[0], tracking_mode="shown")
    elif mutation == "swapped_query":
        _mutate_perf(hidden[0], tracking_query=1)
    elif mutation == "pair_id":
        _mutate_perf(shown[0], pair_id="pair-other")
    elif mutation == "run_id":
        first = json.loads(hidden[0].read_text())["chi_motion_performance"]["run_id"]
        _mutate_perf(shown[0], run_id=first)
    elif mutation == "case_id":
        _mutate_perf(shown[0], case_id="S5-N01")
    elif mutation == "clip":
        _mutate_perf(shown[0], clip_sha256="c" * 64)
    elif mutation == "config":
        _mutate_perf(shown[0], config_sha256="c" * 64)
    elif mutation == "label_clip":
        for report in [*hidden, *shown]:
            _mutate_perf(report, clip_sha256="e" * 64)
    elif mutation == "sample_count":
        _mutate_perf(shown[0], sample_count=99)
    elif mutation == "sample_duration":
        _mutate_perf(shown[0], sample_duration_s=19.0)
    elif mutation == "paired_sample_count":
        report = json.loads(shown[0].read_text())
        report["chi_motion_performance"]["sample_count"] = 99
        report["stages"]["detect_batch"]["engine"]["count"] = 99
        shown[0].write_text(json.dumps(report))
    elif mutation == "paired_sample_duration":
        report = json.loads(shown[0].read_text())
        report["chi_motion_performance"]["sample_duration_s"] = 19.0
        report["stages"]["detect_batch"]["engine"]["span_s"] = 19.0
        shown[0].write_text(json.dumps(report))
    elif mutation == "schema_version":
        _mutate_perf(shown[0], schema_version=2)
    elif mutation == "capture_hash":
        shown[0].with_suffix(".mjpeg").write_bytes(b"tampered")

    with pytest.raises(InputError, match=message):
        score_inputs(
            labels, observations, audit, hidden, shown, tmp_path / "result.json"
        )


def test_performance_pairs_are_matched_by_pair_id_not_argument_order(
    tmp_path: Path,
) -> None:
    labels, observations, audit, hidden, shown = _valid_inputs(tmp_path)

    result = score_inputs(
        labels,
        observations,
        audit,
        list(reversed(hidden)),
        [shown[1], shown[2], shown[0]],
        tmp_path / "result.json",
    )

    assert [pair["pair_id"] for pair in result["tracking_visibility_fps"]["pairs"]] == [
        "pair-1", "pair-2", "pair-3"
    ]
    assert result["tracking_visibility_fps"]["hidden_runs"] == [10.0, 12.0, 11.0]
    assert result["tracking_visibility_fps"]["shown_runs"] == [9.0, 10.0, 8.0]


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
        report = json.loads(hidden[0].read_text())
        report["stages"]["detect_batch"]["engine"]["rate_per_s"] = None
        hidden[0].write_text(json.dumps(report))
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
