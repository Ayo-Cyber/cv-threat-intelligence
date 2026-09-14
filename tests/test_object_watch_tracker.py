from __future__ import annotations

from cvti.object_watch.matcher import ObjectMatch
from cvti.object_watch.tracker import ObjectStateTracker


def match(
    *,
    ts: float = 1.0,
    bbox: tuple[int, int, int, int] = (1, 1, 10, 10),
    zone: str | None = "storage",
    track_id: int | None = 7,
    object_id: str = "chi-carton",
) -> ObjectMatch:
    return ObjectMatch(
        camera_id="cam1",
        object_id=object_id,
        object_label="Chi carton",
        category="product",
        bbox=bbox,
        similarity=0.82,
        timestamp=ts,
        track_id=track_id,
        zone_id=zone,
    )


def test_entered_and_exited_zone_are_edges_not_continuous():
    tracker = ObjectStateTracker(left_behind_seconds=120)

    first = tracker.update([match(ts=1.0, zone="storage")], 1.0)
    second = tracker.update([match(ts=2.0, bbox=(2, 2, 11, 11), zone="storage")], 2.0)
    third = tracker.update([match(ts=3.0, bbox=(20, 20, 30, 30), zone="loading_bay")], 3.0)

    assert [e.state for e in first] == ["object_seen", "object_entered_zone"]
    assert [e.state for e in second] == ["object_seen"]
    assert "object_exited_zone" in [e.state for e in third]
    assert "object_entered_zone" in [e.state for e in third]


def test_removed_fires_when_stable_object_disappears_after_grace():
    tracker = ObjectStateTracker(removed_grace_seconds=2.0, stable_seen_seconds=1.0)

    assert tracker.update([match(ts=1.0)], 1.0)
    assert not any(e.state == "object_removed" for e in tracker.update([], 2.0))
    removed = tracker.update([], 4.1)

    assert [e.state for e in removed] == ["object_removed"]
    assert removed[0].object_id == "chi-carton"


def test_left_behind_requires_stationary_dwell():
    tracker = ObjectStateTracker(left_behind_seconds=3.0, stationary_pixel_tolerance=3.0)

    tracker.update([match(ts=1.0, bbox=(10, 10, 20, 20))], 1.0)
    tracker.update([match(ts=2.0, bbox=(11, 10, 21, 20))], 2.0)
    events = tracker.update([match(ts=4.2, bbox=(10, 11, 20, 21))], 4.2)

    assert "object_left_behind" in [e.state for e in events]
    assert next(e for e in events if e.state == "object_left_behind").dwell_seconds >= 3.0


def test_left_behind_resets_when_object_moves_materially():
    tracker = ObjectStateTracker(left_behind_seconds=3.0, stationary_pixel_tolerance=3.0)

    tracker.update([match(ts=1.0, bbox=(10, 10, 20, 20))], 1.0)
    tracker.update([match(ts=2.0, bbox=(80, 80, 90, 90))], 2.0)
    events = tracker.update([match(ts=4.2, bbox=(81, 80, 91, 90))], 4.2)

    assert "object_left_behind" not in [e.state for e in events]


def test_loaded_near_vehicle_requires_product_and_vehicle_overlap():
    tracker = ObjectStateTracker()

    events = tracker.update(
        [match(ts=1.0, bbox=(10, 10, 30, 30), zone="loading_bay")],
        1.0,
        vehicles=[(20, 20, 60, 60)],
    )

    loaded = [e for e in events if e.state == "object_loaded_near_vehicle"]
    assert len(loaded) == 1
    assert loaded[0].zone_id == "loading_bay"


def test_loaded_near_vehicle_ignores_non_product():
    tracker = ObjectStateTracker()
    vehicle_match = ObjectMatch(
        camera_id="cam1",
        object_id="truck",
        object_label="Truck",
        category="vehicle",
        bbox=(10, 10, 30, 30),
        similarity=0.9,
        timestamp=1.0,
        track_id=9,
        zone_id="loading_bay",
    )

    events = tracker.update([vehicle_match], 1.0, vehicles=[(20, 20, 60, 60)])

    assert "object_loaded_near_vehicle" not in [e.state for e in events]
