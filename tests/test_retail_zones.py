"""Validates the zone/dwell logic in retail_zones.py with synthetic detections.

No torch / ultralytics needed — we hand-build sv.Detections, so this runs anywhere
supervision is installed. Run:  python tests/test_retail_zones.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.retail.zones import (  # noqa: E402
    RetailZoneMonitor,
    ZoneSpec,
    filter_person_detections,
    load_zone_config,
)


def _person(xyxy: list[float], tracker_id: int) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        tracker_id=np.array([tracker_id]),
    )


def _shelf_zone() -> ZoneSpec:
    # A shelf covering the left half of a 1000x1000 frame, 5s loiter threshold.
    return ZoneSpec(
        name="shelf",
        polygon=np.array([[0, 0], [500, 0], [500, 1000], [0, 1000]]),
        anchors=(sv.Position.BOTTOM_CENTER,),
        kind="shelf",
        dwell_alert_seconds=5.0,
    )


def test_presence_detection() -> None:
    monitor = RetailZoneMonitor([_shelf_zone()])
    # Person whose bottom-center (x=100) is inside the left shelf.
    inside = monitor.update(_person([80, 100, 120, 400], tracker_id=1), timestamp=0.0)
    assert inside[0].zones == ["shelf"], inside[0].zones
    # Person on the right half — bottom-center x=800, outside.
    outside = monitor.update(_person([780, 100, 820, 400], tracker_id=2), timestamp=0.0)
    assert outside[0].zones == [], outside[0].zones
    print("PASS presence detection")


def test_dwell_accumulates_and_alerts() -> None:
    monitor = RetailZoneMonitor([_shelf_zone()])
    det = _person([80, 100, 120, 400], tracker_id=1)
    s0 = monitor.update(det, timestamp=0.0)[0]
    assert abs(s0.dwell_seconds["shelf"] - 0.0) < 1e-6
    assert not s0.loitering
    s_mid = monitor.update(det, timestamp=3.0)[0]
    assert abs(s_mid.dwell_seconds["shelf"] - 3.0) < 1e-6
    assert not s_mid.loitering, "3s < 5s threshold should not loiter-alert"
    s_late = monitor.update(det, timestamp=6.0)[0]
    assert abs(s_late.dwell_seconds["shelf"] - 6.0) < 1e-6
    assert s_late.loitering, "6s > 5s threshold should loiter-alert"
    print("PASS dwell accumulates and crosses loiter threshold")


def test_dwell_resets_on_leave() -> None:
    monitor = RetailZoneMonitor([_shelf_zone()])
    inside = _person([80, 100, 120, 400], tracker_id=1)
    outside = _person([780, 100, 820, 400], tracker_id=1)
    monitor.update(inside, timestamp=0.0)
    monitor.update(inside, timestamp=4.0)            # dwell = 4s
    monitor.update(outside, timestamp=5.0)           # left the zone -> forget
    back = monitor.update(inside, timestamp=6.0)[0]  # re-entered -> dwell restarts at 0
    assert abs(back.dwell_seconds["shelf"] - 0.0) < 1e-6, back.dwell_seconds
    print("PASS dwell resets when a track leaves and re-enters")


def test_untracked_detection_has_no_dwell() -> None:
    monitor = RetailZoneMonitor([_shelf_zone()])
    det = sv.Detections(
        xyxy=np.array([[80, 100, 120, 400]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )  # tracker_id is None
    s = monitor.update(det, timestamp=10.0)[0]
    assert s.zones == ["shelf"]
    assert s.tracker_id is None
    assert s.dwell_seconds["shelf"] == 0.0
    print("PASS untracked detection reports presence but no dwell")


def test_example_config_loads() -> None:
    cfg = Path(__file__).resolve().parents[1] / "configs" / "retail_zones.example.json"
    zones = load_zone_config(cfg)
    names = {z.name for z in zones}
    assert {"shelf_left", "shelf_right", "exit"} <= names, names
    monitor = RetailZoneMonitor(zones)
    # A person standing in the left shelf of the example (feet around x=280,y=650).
    det = _person([240, 300, 320, 660], tracker_id=7)
    s = monitor.update(det, timestamp=0.0)[0]
    assert "shelf_left" in s.zones, s.zones
    print("PASS example config loads and triggers")


def test_sticky_dwell_bridges_brief_gap() -> None:
    # grace=2.0s: a 1s flicker out of the zone must NOT reset dwell.
    monitor = RetailZoneMonitor([_shelf_zone()], dwell_grace_seconds=2.0)
    inside = _person([80, 100, 120, 400], tracker_id=1)   # bottom-center x=100 -> in zone
    outside = _person([780, 100, 820, 400], tracker_id=1)  # bottom-center x=800 -> out
    monitor.update(inside, timestamp=0.0)
    monitor.update(inside, timestamp=1.0)                  # dwell = 1s
    monitor.update(outside, timestamp=2.0)                 # 1s gap, within 2s grace -> keep
    back = monitor.update(inside, timestamp=3.0)[0]
    assert abs(back.dwell_seconds["shelf"] - 3.0) < 1e-6, back.dwell_seconds
    print("PASS sticky dwell bridges a brief gap within grace")


def test_sticky_dwell_resets_after_grace() -> None:
    monitor = RetailZoneMonitor([_shelf_zone()], dwell_grace_seconds=0.5)
    inside = _person([80, 100, 120, 400], tracker_id=1)
    outside = _person([780, 100, 820, 400], tracker_id=1)
    monitor.update(inside, timestamp=0.0)
    monitor.update(inside, timestamp=2.0)                  # dwell = 2s
    monitor.update(outside, timestamp=3.0)                 # 1s gap > 0.5s grace -> forget
    back = monitor.update(inside, timestamp=4.0)[0]
    assert abs(back.dwell_seconds["shelf"] - 0.0) < 1e-6, back.dwell_seconds
    print("PASS dwell resets once the gap exceeds grace")


def test_person_filter_drops_mannequin_keeps_person() -> None:
    det = sv.Detections(
        xyxy=np.array([
            [300, 50, 340, 90],     # small square 'mannequin head' -> drop
            [80, 100, 180, 500],    # tall large 'person' -> keep
        ], dtype=float),
        confidence=np.array([0.9, 0.9]),
        class_id=np.array([0, 0]),
        tracker_id=np.array([1, 2]),
    )
    kept = filter_person_detections(det, frame_hw=(1000, 1000))
    assert len(kept) == 1, len(kept)
    assert int(kept.tracker_id[0]) == 2, "the person, not the mannequin, should survive"
    print("PASS person filter drops mannequin, keeps person")


def test_annotate_runs() -> None:
    monitor = RetailZoneMonitor([_shelf_zone()])
    det = _person([80, 100, 120, 400], tracker_id=1)
    states = monitor.update(det, timestamp=2.0)
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    out = monitor.annotate(frame, det, states)
    assert out.shape == frame.shape
    print("PASS annotate produces a same-shape frame")


if __name__ == "__main__":
    test_presence_detection()
    test_dwell_accumulates_and_alerts()
    test_dwell_resets_on_leave()
    test_untracked_detection_has_no_dwell()
    test_sticky_dwell_bridges_brief_gap()
    test_sticky_dwell_resets_after_grace()
    test_person_filter_drops_mannequin_keeps_person()
    test_example_config_loads()
    test_annotate_runs()
    print("\nAll retail_zones tests passed.")
