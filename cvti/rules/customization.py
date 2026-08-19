"""Customization Engine — evaluates user_config.json rules against Detection Core raw events."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from cvti.contracts import CandidateAlert, RawEvent
from cvti.logging_setup import get_logger

log = get_logger(__name__)

PRIORITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}


class CustomizationEngine:
    """Evaluates a list of RawEvents against user-defined rules and returns CandidateAlerts."""

    def __init__(self, config_path: str | Path | None = None,
                 baseline_path: str | Path | None = None) -> None:
        self.rules: list[dict] = []
        # Always-on critical safety rules (weapon, violence, person-down, ...).
        # Merged into every evaluation and NOT disableable via the customer config,
        # so a narrow user policy can never hide a universal critical threat.
        self.baseline_rules: list[dict] = []
        self.use_case_id: str = "default"
        if baseline_path:
            self.load_baseline(baseline_path)
        if config_path:
            self.load(config_path)

    def load(self, config_path: str | Path) -> None:
        path = Path(config_path)
        if not path.exists():
            log.warning(f"[CustomizationEngine] Config not found: {path}")
            return
        data = json.loads(path.read_text())
        self.use_case_id = data.get("use_case_id", "default")
        self.rules = data.get("rules", [])
        log.info(f"[CustomizationEngine] Loaded {len(self.rules)} rules for use-case '{self.use_case_id}'")

    def load_baseline(self, baseline_path: str | Path) -> None:
        path = Path(baseline_path)
        if not path.exists():
            log.warning(f"[CustomizationEngine] Baseline config not found: {path}")
            return
        self.baseline_rules = json.loads(path.read_text()).get("rules", [])
        log.info(f"[CustomizationEngine] Loaded {len(self.baseline_rules)} always-on baseline rule(s)")

    def has_rules(self) -> bool:
        return bool(self.rules or self.baseline_rules)

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

        # The baseline is a SAFETY NET, not a second opinion: if the customer's own
        # config already handles a detector, its rule wins and the baseline stays
        # out of the way. Without this a camera with a panic_running rule also got
        # baseline_panic_running, so one incident reached the operator twice under
        # two names.
        covered = {r.get("trigger", {}).get("detector")
                   for r in self.rules if r.get("trigger", {}).get("detector")}
        baseline = [r for r in self.baseline_rules
                    if r.get("trigger", {}).get("detector") not in covered]

        # Baseline first so critical safety rules are always evaluated, whatever
        # the customer config says.
        for rule in baseline + self.rules:
            # Compound recipe (Phase 3): several signals combined by a logic op.
            if "signals" in rule:
                compound = _eval_compound(rule, events, now)
                if compound is not None:
                    alerts.append(compound)
                continue
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
# Compound threat recipes (Phase 3)
# A recipe combines several weak/strong signals into one high-level threat,
# e.g. armed_robbery = weapon_candidate + violence + running + person_down.
# ---------------------------------------------------------------------------

# Signal name -> the detector it usually corresponds to (soft aliases).
_SIGNAL_ALIASES = {
    "weapon_candidate": "weapons", "weapon": "weapons", "gun": "weapons", "knife": "weapons",
    "violence": "violence", "assault": "violence", "fight": "violence",
    "person_down": "person_down", "fall": "person_down",
    "concealment": "concealment", "theft": "theft",
    "running": "running", "panic": "running",
    "video_action": "video_action",
}


def _match_signal(event: RawEvent, spec: Any) -> bool:
    if isinstance(spec, dict):
        return _match_trigger(event, spec)
    target = _SIGNAL_ALIASES.get(spec, spec)
    sig_type = str(event.extra.get("signal_type") or "")
    return (event.detector == target or event.detector == spec
            or spec in sig_type or target in sig_type)


def _logic_satisfied(logic: str, severities: list[int], total_signals: int) -> bool:
    count = len(severities)
    logic = (logic or "any").lower()
    if logic == "all":
        return count >= total_signals and total_signals > 0
    if logic == "any":
        return count >= 1
    if logic.startswith("at_least_"):
        try:
            return count >= int(logic.rsplit("_", 1)[1])
        except ValueError:
            return count >= 1
    if logic == "one_high_or_two_medium":
        highs = sum(1 for s in severities if s >= PRIORITY_ORDER["high"])
        mediums = sum(1 for s in severities if s >= PRIORITY_ORDER["medium"])
        return highs >= 1 or mediums >= 2
    return count >= 1


def _eval_compound(rule: dict, events: list[RawEvent], now: datetime) -> CandidateAlert | None:
    if not _match_time_filter(rule.get("time_filter"), now):
        return None
    specs = rule.get("signals", [])
    present: dict[str, int] = {}   # signal key -> best severity seen
    latest_ts = 0.0
    for event in events:
        if not event.active:
            continue
        for spec in specs:
            if _match_signal(event, spec):
                key = spec if isinstance(spec, str) else spec.get("name", str(spec))
                sev = PRIORITY_ORDER.get(event.level, PRIORITY_ORDER["medium"])
                present[key] = max(present.get(key, 0), sev)
                latest_ts = max(latest_ts, event.timestamp)
    if not present or not _logic_satisfied(rule.get("logic", "any"),
                                           list(present.values()), len(specs)):
        return None
    reasons = [f"{k}={_rank_name(v)}" for k, v in present.items()]
    return CandidateAlert(
        rule_name=rule["name"],
        priority=rule.get("priority", "high"),
        detector="compound",
        title=rule.get("title", rule["name"].replace("_", " ").upper()),
        person_id=None,
        object_label=None,
        timestamp=latest_ts,
        reasons=reasons,
        question=rule.get("gate_question"),
    )


def _rank_name(rank: int) -> str:
    for name, r in PRIORITY_ORDER.items():
        if r == rank:
            return name
    return "medium"
