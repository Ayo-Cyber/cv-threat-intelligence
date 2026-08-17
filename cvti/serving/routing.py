"""Alert routing — get the right alert to the right person, and chase it if ignored.

Until now every confirmed alert went to one site-wide channel. Real sites need:

    critical           -> the guard's phone, now
    medium/low         -> a log or a digest
    a specific camera  -> the person responsible for that area
    out of hours       -> a different responder
    nobody acknowledged-> escalate

A rule matches on priority / camera / rule-name / time-of-day and names the
channels to use. First match wins (put specific rules above general ones); if
nothing matches, the site default is used, so routing can never silently drop an
alert. Escalation is separate: if an alert is still unacknowledged after N
minutes, it is re-sent to the escalation channels.

Config (configs/routing.json):

    {
      "default": "console",
      "rules": [
        {"name": "critical-to-guard", "when": {"priority": ["critical"]},
         "notify": "telegram:<token>:<chat>", "escalate_after_minutes": 5,
         "escalate_to": "whatsapp"},
        {"name": "night-shift", "when": {"between": ["18:00", "06:00"]},
         "notify": "console,webhook:https://..."},
        {"name": "quiet-lows", "when": {"priority": ["low"]}, "notify": "console"}
      ]
    }
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _hhmm(s: str) -> int:
    h, _, m = s.partition(":")
    return int(h) * 60 + int(m or 0)


def _in_window(now_min: int, start: str, end: str) -> bool:
    a, b = _hhmm(start), _hhmm(end)
    return a <= now_min < b if a <= b else (now_min >= a or now_min < b)   # wraps midnight


@dataclass
class RoutingRule:
    name: str
    when: dict = field(default_factory=dict)
    notify: str = "console"
    escalate_after_minutes: float = 0.0
    escalate_to: str = ""

    def matches(self, event: dict, now: datetime | None = None) -> bool:
        w = self.when or {}
        if not w:
            return True                       # a catch-all rule
        pr = w.get("priority")
        if pr and str(event.get("priority", "")).lower() not in [p.lower() for p in pr]:
            return False
        cams = w.get("camera")
        if cams and event.get("camera_id") not in cams:
            return False
        rules = w.get("rule")
        if rules and event.get("rule") not in rules:
            return False
        zones = w.get("zone")
        if zones and event.get("zone") not in zones:
            return False
        window = w.get("between")
        if window:
            now = now or datetime.now()
            if not _in_window(now.hour * 60 + now.minute, window[0], window[1]):
                return False
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "when": self.when, "notify": self.notify,
                "escalate_after_minutes": self.escalate_after_minutes,
                "escalate_to": self.escalate_to}


@dataclass
class RoutingPolicy:
    default: str = "console"
    rules: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path, default: str = "console") -> "RoutingPolicy":
        p = Path(path)
        if not p.exists():
            return cls(default=default)
        try:
            data = json.loads(p.read_text())
        except Exception:  # noqa: BLE001 - a broken policy must not stop alerting
            print(f"[routing] could not parse {p}; using default channel only")
            return cls(default=default)
        rules = [RoutingRule(r.get("name", f"rule{i}"), r.get("when", {}),
                             r.get("notify", default),
                             float(r.get("escalate_after_minutes", 0) or 0),
                             r.get("escalate_to", ""))
                 for i, r in enumerate(data.get("rules", []), 1)]
        return cls(default=data.get("default", default), rules=rules)

    def match(self, event: dict, now: datetime | None = None) -> RoutingRule | None:
        """First matching rule wins; None means 'use the default channel'."""
        for r in self.rules:
            if r.matches(event, now):
                return r
        return None

    def channels_for(self, event: dict, now: datetime | None = None) -> tuple[str, str]:
        """-> (channel spec to notify, name of the rule that decided) """
        r = self.match(event, now)
        return (r.notify, r.name) if r else (self.default, "default")

    def to_dict(self) -> dict:
        return {"default": self.default, "rules": [r.to_dict() for r in self.rules]}


class EscalationTracker:
    """Re-notify alerts nobody acknowledged.

    The sink registers each routed alert that has an escalation deadline; `due()`
    returns the ones now overdue AND still unacknowledged (asked of the DB via the
    `is_acknowledged` callback), each exactly once.
    """

    def __init__(self, is_acknowledged: Any = None) -> None:
        self._pending: dict[Any, dict] = {}
        self.is_acknowledged = is_acknowledged or (lambda _id: False)

    def register(self, event_id: Any, event: dict, rule: RoutingRule, now: float | None = None) -> None:
        if not rule.escalate_after_minutes or not rule.escalate_to:
            return
        # `is None`, not `or`: a caller-supplied now=0.0 is falsy and would
        # silently fall back to wall-clock, pushing the deadline far into the future.
        base = time.time() if now is None else now
        self._pending[event_id] = {
            "event": event, "to": rule.escalate_to, "rule": rule.name,
            "due_at": base + rule.escalate_after_minutes * 60.0}

    def due(self, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        out = []
        for eid, item in list(self._pending.items()):
            if now < item["due_at"]:
                continue
            self._pending.pop(eid, None)          # fire once, either way
            try:
                if self.is_acknowledged(eid):
                    continue                       # handled in time — nothing to do
            except Exception:  # noqa: BLE001
                pass
            out.append({"event_id": eid, **item})
        return out

    @property
    def pending_count(self) -> int:
        return len(self._pending)
