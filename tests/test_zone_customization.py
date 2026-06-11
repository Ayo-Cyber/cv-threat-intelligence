"""Proves the zone -> Customization Engine wiring: zone + time + dwell rules fire.

This is the "anyone in the vault zone after 8pm is a threat" example, end to end at the
logic level (no video needed). Run:  python tests/test_zone_customization.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from customization import CustomizationEngine, zone_states_to_events  # noqa: E402
from retail_zones import PersonZoneState  # noqa: E402

CFG = str(Path(__file__).resolve().parents[1] / "configs" / "banking_zones_v1.json")
NIGHT = datetime(2026, 6, 11, 21, 0)   # 9pm
DAY = datetime(2026, 6, 11, 12, 0)     # noon


def _state(zones, dwell, tid=7):
    return PersonZoneState(tracker_id=tid, bbox=(0, 0, 1, 1), zones=zones, dwell_seconds=dwell)


def test_presence_events_shape() -> None:
    events = zone_states_to_events([_state(["vault", "atm"], {"vault": 1.0, "atm": 2.0})], timestamp=5.0)
    assert len(events) == 2
    assert all(e.detector == "presence" and e.active for e in events)
    zones = {e.extra["zone"] for e in events}
    assert zones == {"vault", "atm"}, zones
    print("PASS zone_states_to_events emits one presence event per occupied zone")


def test_vault_after_hours_fires_only_at_night() -> None:
    engine = CustomizationEngine(CFG)
    events = zone_states_to_events([_state(["vault"], {"vault": 3.0})])
    night = engine.evaluate(events, now=NIGHT)
    assert any(a.rule_name == "vault_after_hours" and a.priority == "critical" for a in night), night
    day = engine.evaluate(events, now=DAY)
    assert not any(a.rule_name == "vault_after_hours" for a in day), "must NOT fire during the day"
    print("PASS vault_after_hours fires at 9pm, stays silent at noon (time filter works)")


def test_atm_loitering_needs_dwell() -> None:
    engine = CustomizationEngine(CFG)
    long_dwell = zone_states_to_events([_state(["atm"], {"atm": 75.0})])
    assert any(a.rule_name == "atm_loitering" for a in engine.evaluate(long_dwell, now=DAY)), "75s should fire"
    short_dwell = zone_states_to_events([_state(["atm"], {"atm": 20.0})])
    assert not any(a.rule_name == "atm_loitering" for a in engine.evaluate(short_dwell, now=DAY)), "20s should not"
    print("PASS atm_loitering fires at 75s dwell, not at 20s (dwell context filter works)")


def test_wrong_zone_does_not_fire() -> None:
    engine = CustomizationEngine(CFG)
    # Person in the lobby (not vault, not atm) at night -> no rule should match.
    events = zone_states_to_events([_state(["lobby"], {"lobby": 100.0})])
    assert engine.evaluate(events, now=NIGHT) == [], "lobby presence matches no configured zone rule"
    print("PASS presence in an unconfigured zone fires nothing")


if __name__ == "__main__":
    test_presence_events_shape()
    test_vault_after_hours_fires_only_at_night()
    test_atm_loitering_needs_dwell()
    test_wrong_zone_does_not_fire()
    print("\nAll zone-customization tests passed.")
