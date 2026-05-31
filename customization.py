"""Customization Engine — evaluates user_config.json rules against Detection Core raw events."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from detector import ThreatAssessment


@dataclass
class RawEvent:
    detector: str           # "weapons" | "violence" | "theft"
    active: bool
    title: str              # e.g. "POSSIBLE THEFT", "VIOLENCE SUSPECTED", "DANGEROUS OBJECT"
    level: str              # "none" | "low" | "high" | "critical"
    state: str = ""         # internal state name, e.g. "DEPART", "ACQUIRE", "APPROACH"
    person_id: int | None = None
    object_label: str | None = None
    timestamp: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class CandidateAlert:
    rule_name: str
    priority: str           # "low" | "medium" | "high" | "critical"
    detector: str
    title: str
    person_id: int | None
    object_label: str | None
    timestamp: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "priority": self.priority,
            "detector": self.detector,
            "title": self.title,
            "person_id": self.person_id,
            "object_label": self.object_label,
            "timestamp": self.timestamp,
            "reasons": self.reasons,
        }


PRIORITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}


class CustomizationEngine:
    """Evaluates a list of RawEvents against user-defined rules and returns CandidateAlerts."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.rules: list[dict] = []
        self.use_case_id: str = "default"
        if config_path:
            self.load(config_path)

    def load(self, config_path: str | Path) -> None:
        path = Path(config_path)
        if not path.exists():
            print(f"[CustomizationEngine] Config not found: {path}")
            return
        data = json.loads(path.read_text())
        self.use_case_id = data.get("use_case_id", "default")
        self.rules = data.get("rules", [])
        print(f"[CustomizationEngine] Loaded {len(self.rules)} rules for use-case '{self.use_case_id}'")

    def evaluate(
        self,
        events: list[RawEvent],
        scene_context: dict | None = None,
        now: datetime | None = None,
    ) -> list[CandidateAlert]:
        """Return all rules that match the current events, sorted highest priority first."""
        now = now or datetime.now()
        context = scene_context or {}
        alerts: list[CandidateAlert] = []

        for rule in self.rules:
            trigger = rule.get("trigger", {})
            for event in events:
                if not event.active:
                    continue
                if not _match_trigger(event, trigger):
                    continue
                if not _match_time_filter(rule.get("time_filter"), now):
                    continue
                if not _match_context_filter(rule.get("context_filter"), event, context):
                    continue
                alerts.append(
                    CandidateAlert(
                        rule_name=rule["name"],
                        priority=rule.get("priority", "medium"),
                        detector=event.detector,
                        title=event.title,
                        person_id=event.person_id,
                        object_label=event.object_label,
                        timestamp=event.timestamp,
                    )
                )
                break  # one alert per rule per frame is enough

        alerts.sort(key=lambda a: PRIORITY_ORDER.get(a.priority, 0), reverse=True)
        return alerts

    def top_alert(
        self,
        events: list[RawEvent],
        scene_context: dict | None = None,
        now: datetime | None = None,
    ) -> CandidateAlert | None:
        alerts = self.evaluate(events, scene_context, now)
        return alerts[0] if alerts else None


# ---------------------------------------------------------------------------
# Trigger matching
# ---------------------------------------------------------------------------

def _match_trigger(event: RawEvent, trigger: dict) -> bool:
    detector = trigger.get("detector")
    if detector and event.detector != detector:
        return False

    state = trigger.get("state")
    if state:
        # match against explicit state field first, fall back to title substring
        if event.state:
            if state.upper() != event.state.upper():
                return False
        elif state.upper() not in event.title.upper():
            return False

    level = trigger.get("level")
    if level and event.level != level:
        return False

    return True


# ---------------------------------------------------------------------------
# Time filter  "HH:MM-HH:MM"  (supports overnight ranges like 22:00-06:00)
# ---------------------------------------------------------------------------

def _match_time_filter(time_filter: str | None, now: datetime) -> bool:
    if not time_filter:
        return True
    m = re.match(r"(\d{1,2}:\d{2})-(\d{1,2}:\d{2})", time_filter)
    if not m:
        return True
    start = _hhmm_to_minutes(m.group(1))
    end = _hhmm_to_minutes(m.group(2))
    current = now.hour * 60 + now.minute
    if start <= end:
        return start <= current < end
    # overnight: e.g. 22:00–06:00
    return current >= start or current < end


def _hhmm_to_minutes(t: str) -> int:
    h, mn = t.split(":")
    return int(h) * 60 + int(mn)


# ---------------------------------------------------------------------------
# Context filter  — tiny safe expression evaluator
# Supports: ==, !=, >, <, >=, <=, and, or, not, string literals, None
# ---------------------------------------------------------------------------

_ALLOWED_NAMES = {"True", "False", "None", "and", "or", "not", "in"}


def _match_context_filter(
    expr: str | None,
    event: RawEvent,
    context: dict,
) -> bool:
    if not expr:
        return True

    ns: dict[str, Any] = {**context}
    ns.update(
        {
            "detector": event.detector,
            "title": event.title,
            "level": event.level,
            "person_id": event.person_id,
            "object_label": event.object_label,
        }
    )
    ns.update(event.extra)

    try:
        return bool(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
    except Exception:
        return True  # don't block alert if expression is malformed


# ---------------------------------------------------------------------------
# Converter: ThreatAssessment → RawEvent (used by detector.py)
# ---------------------------------------------------------------------------

def assessments_to_events(
    object_assessment: ThreatAssessment | None,
    violence_assessment: ThreatAssessment | None,
    theft_assessment: ThreatAssessment | None,
    timestamp: float = 0.0,
    theft_detector: Any = None,
) -> list[RawEvent]:
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
        # Derive the highest-priority active state across all tracked persons
        active_state = ""
        obj_label = None
        if theft_detector is not None:
            state_priority = {"DEPART": 3, "ACQUIRE": 2, "APPROACH": 1, "IDLE": 0}
            best = -1
            for ps in theft_detector.person_states.values():
                p = state_priority.get(ps.state, 0)
                if p > best:
                    best = p
                    active_state = ps.state
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
