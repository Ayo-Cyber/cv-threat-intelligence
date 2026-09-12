"""Validates the zone/dwell logic in retail_zones.py with synthetic detections.

No torch / ultralytics needed — we hand-build sv.Detections, so this runs anywhere
supervision is installed. Run:  python tests/test_retail_zones.py
"""

from __future__ import annotations

import json
import sys
import tempfile
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


def _wide_zone() -> ZoneSpec:
    # The whole 1000x1000 frame: room for two people to stand FAR apart while
    # both inside — the inheritance-distance tests need that separation.
    return ZoneSpec(
        name="wide",
        polygon=np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]]),
        anchors=(sv.Position.BOTTOM_CENTER,),
        kind="shelf",
        dwell_alert_seconds=60.0,
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
    # An absence LONGER than the production grace is a genuine exit: the
    # dwell restarts. (Sub-grace absences are bridged — the sticky tests.)
    monitor = RetailZoneMonitor([_shelf_zone()])
    grace = RetailZoneMonitor.DWELL_GRACE_DEFAULT
    inside = _person([80, 100, 120, 400], tracker_id=1)
    outside = _person([780, 100, 820, 400], tracker_id=1)
    monitor.update(inside, timestamp=0.0)
    monitor.update(inside, timestamp=4.0)            # dwell = 4s
    monitor.update(outside, timestamp=5.0)           # left the zone
    monitor.update(outside, timestamp=5.0 + grace + 0.6)   # stayed away past grace
    back = monitor.update(inside, timestamp=6.0 + grace)[0]  # -> dwell restarts
    assert abs(back.dwell_seconds["shelf"] - 0.0) < 1e-6, back.dwell_seconds
    print("PASS dwell resets when a track leaves for longer than the grace")


def test_production_default_grace_is_on() -> None:
    # Every engine construction site builds RetailZoneMonitor() bare — the
    # default IS the field behaviour. 0.0 meant one dropped frame reset a
    # 60s loiter timer (the pre-10-Sep field bug).
    assert RetailZoneMonitor([_shelf_zone()]).dwell_grace_seconds == \
        RetailZoneMonitor.DWELL_GRACE_DEFAULT == 2.5
    print("PASS production default grace is 2.5s")


def test_id_switch_inherits_the_dwell_clock() -> None:
    """ByteTrack loses an occluded person and returns them under a NEW id.

    New id used to mean a fresh timer — a loiterer standing still through one
    occlusion was never reported. A new track in the same zone, at the spot a
    track just vanished from, inherits its entry time."""
    monitor = RetailZoneMonitor([_shelf_zone()])
    monitor.update(_person([80, 100, 120, 400], tracker_id=5), timestamp=0.0)
    monitor.update(_person([80, 100, 120, 400], tracker_id=5), timestamp=50.0)
    # occlusion: one absent second, then the tracker hands back id 9 SAME spot
    s = monitor.update(_person([82, 102, 122, 402], tracker_id=9),
                       timestamp=51.0)[0]
    assert s.dwell_seconds["shelf"] >= 50.0, s.dwell_seconds
    print("PASS an id switch does not restart the loiter clock")


def test_inheritance_needs_the_same_spot() -> None:
    # A DIFFERENT person arriving elsewhere in the zone must start at zero —
    # inheritance is for occlusion re-identification, not zone hand-me-downs.
    monitor = RetailZoneMonitor([_wide_zone()])
    monitor.update(_person([80, 100, 120, 400], tracker_id=5), timestamp=0.0)
    monitor.update(_person([80, 100, 120, 400], tracker_id=5), timestamp=50.0)
    s = monitor.update(_person([400, 100, 440, 400], tracker_id=9),
                       timestamp=51.0)[0]
    assert s.dwell_seconds["wide"] < 1.0, s.dwell_seconds
    print("PASS a new person far away starts their own clock")


def test_a_consumed_donor_cannot_seed_two_heirs() -> None:
    monitor = RetailZoneMonitor([_shelf_zone()])
    monitor.update(_person([80, 100, 120, 400], tracker_id=5), timestamp=0.0)
    monitor.update(_person([80, 100, 120, 400], tracker_id=5), timestamp=50.0)
    first = monitor.update(_person([80, 100, 120, 400], tracker_id=9),
                           timestamp=51.0)[0]
    assert first.dwell_seconds["shelf"] >= 50.0
    # id churns AGAIN immediately: 9 -> 12 inherits from 9 (which now holds
    # the clock), not from the long-gone 5 twice over.
    second = monitor.update(_person([80, 100, 120, 400], tracker_id=12),
                            timestamp=52.0)[0]
    assert second.dwell_seconds["shelf"] >= 51.0
    print("PASS the dwell clock survives repeated id churn")


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


def test_far_field_person_survives_a_relaxed_area_gate() -> None:
    """A wide outdoor CCTV view sees a real person at ~0.3% of the frame — the
    retail-tuned 1.2% gate dropped them, so nobody could EVER enter an outdoor
    zone (12 Sep, VIRAT campus demo). The non-retail default (0.2%) keeps them.
    """
    # 1280x720 frame; a distant standing person ~24x88px = 0.23% of frame.
    far = sv.Detections(
        xyxy=np.array([[600, 300, 624, 388]], dtype=float),
        confidence=np.array([0.6]),
        class_id=np.array([0]),
        tracker_id=np.array([1]),
    )
    dropped = filter_person_detections(far, (720, 1280))          # retail default 1.2%
    assert len(dropped) == 0, "retail gate is expected to drop the far-field box"
    kept = filter_person_detections(far, (720, 1280), min_area_ratio=0.002)
    assert len(kept) == 1, "the relaxed far-field gate must keep a real distant person"


def test_zone_entry_fires_once_then_presence_continues() -> None:
    """Crossing INTO a zone is a distinct one-shot event; presence continues
    every frame after (12 Sep: 'entering' must be separable from 'loitering')."""
    mon = RetailZoneMonitor([_wide_zone()])
    s0 = mon.update(_person([100, 100, 200, 400], tracker_id=1), 0.0)[0]
    assert "wide" in s0.entered_zones, "first frame in a zone is an entry"
    s1 = mon.update(_person([100, 100, 200, 400], tracker_id=1), 0.5)[0]
    assert "wide" not in s1.entered_zones, "a staying person does not re-enter"
    assert "wide" in s1.zones, "but presence continues"


def test_zone_exit_is_grace_debounced() -> None:
    """Leaving is reported only after the grace window — a one-frame gap
    (occlusion / boundary jitter) must NOT read as an exit. The person was
    last seen at the frame edge, i.e. somewhere one can actually leave from."""
    mon = RetailZoneMonitor([_wide_zone()], dwell_grace_seconds=2.0)
    mon.update(_person([0, 100, 60, 400], tracker_id=1), 0.0)     # centre x=30: at the edge
    # gone for less than grace -> no exit yet
    mon.update(sv.Detections.empty(), 1.0)
    assert mon.drain_exits() == [], "a brief gap is not an exit"
    # gone past grace -> exit fires, once
    mon.update(sv.Detections.empty(), 3.5)
    assert mon.drain_exits() == [(1, "wide")]
    assert mon.drain_exits() == [], "exits drain — not re-reported"


def test_entry_is_confirmed_not_instant_when_the_zone_asks() -> None:
    """A single false 'person' box used to be a HIGH 'entered' alert. With a
    confirmation window the person must be seen inside for that long; dwell
    still counts from first sight so the loiter clock is unchanged."""
    z = _wide_zone()
    z.entry_confirm_seconds = 0.6
    mon = RetailZoneMonitor([z])
    box = _person([400, 400, 500, 700], tracker_id=1)
    assert mon.update(box, 0.0)[0].entered_zones == [], "first sight is not yet an entry"
    assert mon.update(box, 0.25)[0].entered_zones == []
    s = mon.update(box, 0.7)[0]
    assert s.entered_zones == ["wide"], "confirmed after the window"
    assert abs(s.dwell_seconds["wide"] - 0.7) < 1e-6, "dwell counted from first sight"
    assert mon.update(box, 1.0)[0].entered_zones == [], "fires once"


def test_a_flicker_never_becomes_an_entry_or_an_exit() -> None:
    z = _wide_zone()
    z.entry_confirm_seconds = 0.6
    mon = RetailZoneMonitor([z], dwell_grace_seconds=1.0)
    assert mon.update(_person([400, 400, 500, 700], tracker_id=1), 0.0)[0].entered_zones == []
    for t in (0.25, 3.0, 20.0):                       # never seen again
        mon.update(sv.Detections.empty(), t)
        assert mon.drain_exits() == [], "an entry that never fired cannot be left"


def test_vanishing_mid_zone_is_not_leaving_until_the_lost_window() -> None:
    """Turning away from a close webcam, an occlusion, a detector dropout: the
    track vanishes in the MIDDLE of the floor. That is not a doorway (12 Sep:
    phantom left/entered pairs). It stays present through the lost window,
    then leaves as 'lost' — an honest label, not a claimed departure."""
    mon = RetailZoneMonitor([_wide_zone()], dwell_grace_seconds=2.0)
    mon.update(_person([450, 400, 550, 700], tracker_id=1), 0.0)   # centre (500, 550): interior
    mon.update(sv.Detections.empty(), 3.5)                          # past grace
    assert mon.drain_exits() == [], "an occlusion mid-floor is not an exit"
    mon.update(sv.Detections.empty(), RetailZoneMonitor.LOST_GRACE_DEFAULT + 0.5)
    ex = mon.drain_exits()
    assert len(ex) == 1 and ex[0].how == "lost" and ex[0].tracker_id == 1, ex


def test_reappearing_mid_zone_inherits_across_a_long_occlusion() -> None:
    mon = RetailZoneMonitor([_wide_zone()], dwell_grace_seconds=2.0)
    mon.update(_person([450, 400, 550, 700], tracker_id=1), 0.0)
    mon.update(sv.Detections.empty(), 5.0)                          # gone 5s: past grace, within lost
    s = mon.update(_person([455, 400, 555, 700], tracker_id=2), 5.5)[0]   # new id, same spot
    assert s.entered_zones == [], "same person under a new id: no second entry"
    assert abs(s.dwell_seconds["wide"] - 5.5) < 1e-6, "and the clock is kept"
    assert mon.drain_exits() == [], "and nobody left"


def test_seen_outside_the_zone_is_a_real_exit_after_grace() -> None:
    mon = RetailZoneMonitor([_shelf_zone()], dwell_grace_seconds=1.0)
    mon.update(_person([80, 100, 120, 400], tracker_id=1), 0.0)     # in the shelf zone
    mon.update(_person([780, 100, 820, 400], tracker_id=1), 4.0)    # visibly walked out of it
    mon.update(_person([780, 100, 820, 400], tracker_id=1), 5.5)    # still outside, past grace
    ex = mon.drain_exits()
    assert len(ex) == 1 and ex[0].how == "walked_out", ex
    assert ex[0].dwell_seconds == 0.0, "one sighting inside: no dwell to report"


def test_leaving_through_the_edge_carries_the_dwell() -> None:
    mon = RetailZoneMonitor([_wide_zone()], dwell_grace_seconds=2.0)
    mon.update(_person([450, 400, 550, 700], tracker_id=1), 0.0)
    mon.update(_person([0, 400, 60, 700], tracker_id=1), 10.0)      # at the left edge
    mon.update(sv.Detections.empty(), 12.5)
    ex = mon.drain_exits()
    assert len(ex) == 1 and ex[0].how == "walked_out", ex
    assert abs(ex[0].dwell_seconds - 10.0) < 1e-6, "left after 10s inside"
    assert ex[0] == (1, "wide"), "still unpacks/compares as the old tuple"
    from cvti.event_adapters import zone_exits_to_events
    ev = zone_exits_to_events(ex, timestamp=12.5)[0]
    assert ev.title == "PERSON LEFT ZONE WIDE AFTER 10S"
    assert ev.extra["dwell_seconds"] == 10.0 and ev.extra["how"] == "walked_out"


def test_normalized_polygon_follows_the_frame() -> None:
    """A doorway drawn in frame fractions is right on any camera resolution."""
    cfg = {"zones": [{"name": "door",
                      "polygon": [[0.4, 0.5], [0.6, 0.5], [0.6, 1.0], [0.4, 1.0]]}]}
    p = Path(tempfile.mkdtemp()) / "zones.json"
    p.write_text(json.dumps(cfg))
    specs = load_zone_config(p)
    assert specs[0].normalized and specs[0].entry_confirm_seconds == 0.5
    mon = RetailZoneMonitor(specs)
    # 640x360 frame: the door spans x 256..384, y 180..360. Feet at x=320 -> inside.
    s = mon.update(_person([300, 200, 340, 350], tracker_id=1), 0.0, frame_hw=(360, 640))[0]
    assert s.zones == ["door"], s.zones
    s2 = mon.update(_person([100, 200, 140, 350], tracker_id=2), 0.0, frame_hw=(360, 640))[0]
    assert s2.zones == [], s2.zones


def test_zone_entry_and_exit_events_from_adapter() -> None:
    from cvti.event_adapters import zone_states_to_events, zone_exits_to_events
    mon = RetailZoneMonitor([_wide_zone()], dwell_grace_seconds=1.0)
    states = mon.update(_person([0, 100, 60, 400], tracker_id=7), 0.0)    # at the frame edge
    evs = zone_states_to_events(states, timestamp=0.0)
    kinds = [e.detector for e in evs]
    assert "zone_entry" in kinds and "presence" in kinds
    mon.update(sv.Detections.empty(), 2.0)
    exits = zone_exits_to_events(mon.drain_exits(), timestamp=2.0)
    assert [e.detector for e in exits] == ["zone_exit"]
    assert exits[0].person_id == 7
    assert exits[0].extra["how"] == "walked_out"
