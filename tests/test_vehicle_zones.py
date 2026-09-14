"""Vehicle entering/exiting a zone (KPI #2).

Vehicles reuse the person zone state machine (entry-confirm, grace, edge-vs-lost
exits) but run on their OWN tracker + monitor, filtered to COCO vehicle classes
(car/motorcycle/bus/truck), and emit vehicle_* events that auto-confirm as
deterministic geometry. Proven end-to-end on packaging/demo_data/clips/entrance.mp4
(2 vehicle_entered + 1 vehicle_left, delivered to Telegram, 14 Sep).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.event_adapters import vehicle_states_to_events, vehicle_exits_to_events
from cvti.retail.zones import RetailZoneMonitor, ZoneSpec
from cvti.serving.gate_pool import BYPASS_DETECTORS


def _gate_zone() -> ZoneSpec:
    return ZoneSpec(
        name="gate",
        polygon=np.array([[300, 300], [700, 300], [700, 700], [300, 700]]),
        anchors=(sv.Position.CENTER,),
        kind="restricted",
        dwell_alert_seconds=None,
    )


def _vehicle(cx: float, tid: int) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[cx - 40, 450, cx + 40, 550]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([2]),          # COCO car
        tracker_id=np.array([tid]),
    )


def test_vehicle_events_are_deterministic_bypass():
    # Vehicle crossings are geometry, not a model guess — they must auto-confirm.
    for d in ("vehicle_entry", "vehicle_exit", "vehicle_presence"):
        assert d in BYPASS_DETECTORS, d


def test_vehicle_entry_then_exit_fires_the_right_events():
    mon = RetailZoneMonitor([_gate_zone()], dwell_grace_seconds=1.0)
    # outside the gate (cx=100) -> nothing
    evs = vehicle_states_to_events(mon.update(_vehicle(100, 1), 0.0))
    assert [e.detector for e in evs] == []
    # first frame INSIDE the gate (cx=500) -> entry + presence (confirm=0 default)
    evs = vehicle_states_to_events(mon.update(_vehicle(500, 1), 0.5))
    kinds = [e.detector for e in evs]
    assert "vehicle_entry" in kinds and "vehicle_presence" in kinds
    entry = next(e for e in evs if e.detector == "vehicle_entry")
    assert entry.level == "high" and entry.extra["zone"] == "gate"
    assert entry.object_label == "vehicle"
    # drives back OUT of the gate (still visible at cx=100) -> exit after grace
    mon.update(_vehicle(100, 1), 1.0)
    assert vehicle_exits_to_events(mon.drain_exits()) == []   # within grace, not yet
    mon.update(_vehicle(100, 1), 2.0)                          # absent from zone > grace
    exits = vehicle_exits_to_events(mon.drain_exits(), timestamp=2.0)
    assert [e.detector for e in exits] == ["vehicle_exit"]
    assert exits[0].object_label == "vehicle" and exits[0].extra["how"] == "walked_out"
    assert "LEFT ZONE GATE" in exits[0].title


def test_two_vehicles_each_get_their_own_entry():
    mon = RetailZoneMonitor([_gate_zone()], dwell_grace_seconds=1.0)
    dets = sv.Detections(
        xyxy=np.array([[460, 450, 540, 550], [560, 450, 640, 550]], dtype=float),
        confidence=np.array([0.9, 0.9]),
        class_id=np.array([2, 7]),       # car + truck
        tracker_id=np.array([1, 2]),
    )
    # both start inside -> both get an entry on their first in-zone frame
    evs = vehicle_states_to_events(mon.update(dets, 0.0))
    entries = [e for e in evs if e.detector == "vehicle_entry"]
    assert {e.person_id for e in entries} == {1, 2}
    # staying in-zone does not re-fire entry
    evs = vehicle_states_to_events(mon.update(dets, 0.6))
    assert [e for e in evs if e.detector == "vehicle_entry"] == []
