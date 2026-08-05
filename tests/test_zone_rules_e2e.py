"""End-to-end: zone -> dwell -> presence event -> loitering RULE fires.

Proves the whole rules+zoning chain is wired, using the REAL config files the app
ships with (configs/zones/live_watch.json + configs/rules/live_watch.json):

  RetailZoneMonitor (polygon + dwell)
    -> zone_states_to_events  (presence RawEvent, extra={zone, dwell_seconds})
    -> CustomizationEngine     (loitering rule: presence AND zone AND dwell>=1.5)
    -> CandidateAlert
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.event_adapters import zone_states_to_events
from cvti.retail.zones import RetailZoneMonitor, load_zone_config
from cvti.rules.customization import CustomizationEngine

ZONE_CFG = ROOT / "configs" / "zones" / "live_watch.json"
RULES_CFG = ROOT / "configs" / "rules" / "live_watch.json"


def _person(xyxy, tracker_id=1) -> sv.Detections:
    return sv.Detections(xyxy=np.array([xyxy], dtype=float),
                         class_id=np.array([0]), tracker_id=np.array([tracker_id]))


def _fire_names(engine, monitor, det, t):
    states = monitor.update(det, timestamp=t)
    events = zone_states_to_events(states, timestamp=t)
    alerts = engine.evaluate(events)
    return {a.rule_name for a in alerts}, events


def test_configs_exist_and_load():
    assert ZONE_CFG.exists() and RULES_CFG.exists()
    zones = load_zone_config(str(ZONE_CFG))
    assert zones, "watch zone must load"
    eng = CustomizationEngine(str(RULES_CFG))
    names = {r["name"] for r in eng.rules}
    assert "loitering_watch" in names


def test_loitering_fires_only_after_dwell_threshold():
    monitor = RetailZoneMonitor(load_zone_config(str(ZONE_CFG)))
    engine = CustomizationEngine(str(RULES_CFG))
    person = _person([600, 300, 700, 700], tracker_id=1)   # bottom-centre (650,700) inside zone

    names0, ev0 = _fire_names(engine, monitor, person, t=0.0)
    assert ev0, "a person in the zone must produce a presence event"
    assert ev0[0].extra.get("zone") == "watch"
    assert "loitering_watch" not in names0            # dwell 0 -> no loiter yet

    names_half, ev_half = _fire_names(engine, monitor, person, t=0.5)
    assert ev_half[0].extra["dwell_seconds"] < 1.5
    assert "loitering_watch" not in names_half         # still under threshold

    names_late, ev_late = _fire_names(engine, monitor, person, t=2.0)
    assert ev_late[0].extra["dwell_seconds"] >= 1.5
    assert "loitering_watch" in names_late             # crossed 1.5s -> loitering fires


def test_person_outside_zone_never_loiters():
    monitor = RetailZoneMonitor(load_zone_config(str(ZONE_CFG)))
    engine = CustomizationEngine(str(RULES_CFG))
    # bottom-centre (2000,2000) is outside the ~1280x720 watch polygon
    outside = _person([1950, 1900, 2050, 2000], tracker_id=9)
    for t in (0.0, 2.0, 4.0):
        names, _ = _fire_names(engine, monitor, outside, t=t)
        assert "loitering_watch" not in names


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
