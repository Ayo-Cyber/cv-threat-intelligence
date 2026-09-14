"""Adapters that convert pipeline-specific state into shared RawEvents."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from cvti.contracts import RawEvent

if TYPE_CHECKING:
    from cvti.detector.core import ThreatAssessment


def zone_states_to_events(zone_states: list[Any], timestamp: float = 0.0) -> list[RawEvent]:
    """Bridge RetailZoneMonitor output into presence + zone_entry RawEvents.

    Three distinct signals, not one (12 Sep, field feedback that 'entering'
    was indistinguishable from 'loitering'):
    - `zone_entry`: fires ONCE, the frame a person crosses into the zone — the
      instant 'someone entered the area' alert.
    - `presence`: continuous while they are inside; carries dwell_seconds and
      the `loitering` flag (dwell past the zone's threshold) that the
      'remained beyond the configured time' rule keys on.
    (`zone_exit` is emitted separately from the monitor's drained exits.)
    """
    events: list[RawEvent] = []
    for state in zone_states:
        tid = getattr(state, "tracker_id", None)
        zones = getattr(state, "zones", []) or []
        dwell_map = getattr(state, "dwell_seconds", {}) or {}
        loitering = bool(getattr(state, "loitering", False))
        entered = set(getattr(state, "entered_zones", []) or [])
        for zone in zones:
            dwell = float(dwell_map.get(zone, 0.0))
            if zone in entered:
                events.append(
                    RawEvent(
                        detector="zone_entry",
                        active=True,
                        title=f"PERSON ENTERED ZONE {zone.upper()}",
                        level="medium",
                        person_id=tid,
                        timestamp=timestamp,
                        extra={"zone": zone, "dwell_seconds": dwell},
                    )
                )
            events.append(
                RawEvent(
                    detector="presence",
                    active=True,
                    title=f"PERSON IN ZONE {zone.upper()}",
                    level="low",
                    person_id=tid,
                    timestamp=timestamp,
                    extra={"zone": zone, "dwell_seconds": dwell, "loitering": loitering},
                )
            )
    return events


def zone_exits_to_events(exits: list[Any], timestamp: float = 0.0) -> list[RawEvent]:
    """Bridge the monitor's drained exits (ZoneExit, or bare (tracker_id, zone)
    tuples) into zone_exit events.

    The title carries how long they were inside and how they went — "left
    after 17s" is what an operator acts on; "left" alone is not. A `lost`
    exit (vanished mid-zone past the lost window) says so, rather than
    claiming a departure nobody saw."""
    events: list[RawEvent] = []
    for ex in exits or []:
        tid, zone = tuple(ex)[:2]
        dwell = getattr(ex, "dwell_seconds", None)
        how = getattr(ex, "how", "walked_out")
        after = f" AFTER {dwell:.0f}S" if dwell is not None else ""
        title = (f"PERSON LOST FROM VIEW IN ZONE {zone.upper()}{after}" if how == "lost"
                 else f"PERSON LEFT ZONE {zone.upper()}{after}")
        extra: dict[str, Any] = {"zone": zone, "how": how}
        if dwell is not None:
            extra["dwell_seconds"] = float(dwell)
        events.append(
            RawEvent(
                detector="zone_exit",
                active=True,
                title=title,
                level="low",
                person_id=tid,
                timestamp=timestamp,
                extra=extra,
            )
        )
    return events


def concealment_to_events(assessments: list[Any], timestamp: float = 0.0) -> list[RawEvent]:
    """Bridge ConcealmentDetector output into concealment RawEvents."""
    events: list[RawEvent] = []
    for assessment in assessments:
        destination = getattr(assessment, "destination", None)
        title = "POSSIBLE CONCEALMENT"
        if destination:
            title += f" ({destination})"
        candidate = bool(getattr(assessment, "candidate", False))
        events.append(
            RawEvent(
                detector="concealment",
                active=candidate,
                title=title,
                level="high" if candidate else "none",
                person_id=getattr(assessment, "track_id", None),
                timestamp=timestamp,
                extra={
                    "destination": destination,
                    "score": float(getattr(assessment, "score", 0.0)),
                    "components": deepcopy(getattr(assessment, "components", {}) or {}),
                    "reasons": deepcopy(getattr(assessment, "reasons", []) or []),
                    "limited": bool(getattr(assessment, "limited", False)),
                    "associated_bag": getattr(assessment, "associated_bag", None),
                },
            )
        )
    return events


def simultaneous_movement_to_event(event: dict, timestamp: float = 0.0) -> RawEvent:
    """Bridge aggregate tracked movement into one scenario-5 RawEvent."""
    people_count = int(event.get("people_count", len(event.get("track_ids", []))))
    return RawEvent(
        detector="multiple_people_moving",
        active=True,
        title="MULTIPLE PEOPLE MOVING",
        level="high",
        timestamp=timestamp,
        extra={
            "confidence": float(event["confidence"]),
            "people_count": people_count,
            "track_ids": deepcopy(event["track_ids"]),
            "group_bbox": tuple(event["group_bbox"]),
            "motions": deepcopy(event["motions"]),
            "reasons": [f"{people_count} tracked people moving simultaneously"],
        },
    )


def object_observations_to_events(events: list[Any], timestamp: float = 0.0) -> list[RawEvent]:
    """Bridge object-watch state transitions into RawEvents."""
    raw_events: list[RawEvent] = []
    for event in events:
        event_timestamp = float(getattr(event, "timestamp", timestamp))
        state = str(getattr(event, "state", ""))
        object_label = getattr(event, "object_label", None)
        title_label = str(object_label or "WATCHED OBJECT").upper()
        zone = getattr(event, "zone_id", None)
        title = f"{title_label} {state.replace('_', ' ').upper()}"
        raw_events.append(
            RawEvent(
                detector="object_watch",
                active=True,
                title=title,
                level="medium",
                state=state,
                object_label=object_label,
                timestamp=event_timestamp,
                extra={
                    "object_id": getattr(event, "object_id", None),
                    "object_category": getattr(event, "category", None),
                    "zone": zone,
                    "track_id": getattr(event, "track_id", None),
                    "bbox": getattr(event, "bbox", None),
                    "similarity": float(getattr(event, "similarity", 0.0)),
                    "dwell_seconds": float(getattr(event, "dwell_seconds", 0.0)),
                    "reasons": deepcopy(getattr(event, "reasons", ()) or ()),
                },
            )
        )
    return raw_events


def assessments_to_events(
    object_assessment: ThreatAssessment | None,
    violence_assessment: ThreatAssessment | None,
    theft_assessment: ThreatAssessment | None,
    timestamp: float = 0.0,
    theft_detector: Any = None,
) -> list[RawEvent]:
    """Bridge detector.py ThreatAssessments into shared RawEvents."""
    events: list[RawEvent] = []

    if object_assessment is not None:
        events.append(
            RawEvent(
                detector="weapons",
                active=object_assessment.active,
                title=object_assessment.title,
                level=object_assessment.level,
                timestamp=timestamp,
                extra={"weapon_labels": object_assessment.weapon_labels},
            )
        )

    if violence_assessment is not None:
        events.append(
            RawEvent(
                detector="violence",
                active=violence_assessment.active,
                title=violence_assessment.title,
                level=violence_assessment.level,
                timestamp=timestamp,
            )
        )

    if theft_assessment is not None:
        active_state = ""
        obj_label = None
        if theft_detector is not None:
            state_priority = {"DEPART": 3, "ACQUIRE": 2, "APPROACH": 1, "IDLE": 0}
            best = -1
            for person_state in theft_detector.person_states.values():
                priority = state_priority.get(person_state.state, 0)
                if priority > best:
                    best = priority
                    active_state = person_state.state
        if not obj_label:
            obj_label = (
                theft_assessment.explicit_labels[0]
                if theft_assessment.explicit_labels
                else None
            )
        events.append(
            RawEvent(
                detector="theft",
                active=theft_assessment.active,
                title=theft_assessment.title,
                level=theft_assessment.level,
                state=active_state,
                timestamp=timestamp,
                object_label=obj_label,
            )
        )

    return events
