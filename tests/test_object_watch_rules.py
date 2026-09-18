from __future__ import annotations

import json

import pytest

from cvti.contracts import RawEvent
from cvti.rules.customization import CustomizationEngine


def _event(*, object_id="chi-carton", zone="storage", state="object_seen"):
    extra = {"object_id": object_id, "zone": zone}
    return RawEvent(
        detector="object_watch",
        active=True,
        title="CHI CARTON OBJECT SEEN",
        level="medium",
        state=state,
        object_label="Chi carton",
        extra=extra,
    )


def _engine(trigger, context_filter=None):
    engine = CustomizationEngine()
    rule = {"name": "watch_chi", "priority": "high", "trigger": trigger}
    if context_filter is not None:
        rule["context_filter"] = context_filter
    engine.rules = [rule]
    return engine


@pytest.mark.parametrize(
    "event",
    [
        _event(object_id="other"),
        RawEvent("object_watch", True, "SEEN", "medium", state="object_seen",
                 extra={"zone": "storage"}),
        _event(zone="loading_bay"),
        RawEvent("object_watch", True, "SEEN", "medium", state="object_seen",
                 extra={"object_id": "chi-carton"}),
    ],
)
def test_target_and_zone_trigger_are_strict(event):
    engine = _engine({
        "detector": "object_watch",
        "state": "object_seen",
        "object_id": "chi-carton",
        "zone": "storage",
    })
    assert engine.evaluate([event]) == []


def test_scoped_object_seen_rule_matches_exact_target_and_zone():
    engine = _engine({
        "detector": "object_watch",
        "state": "object_seen",
        "object_id": "chi-carton",
        "zone": "storage",
    })
    assert [a.rule_name for a in engine.evaluate([_event()])] == ["watch_chi"]


def test_unscoped_object_seen_config_is_rejected(tmp_path):
    config = tmp_path / "rules.json"
    config.write_text(json.dumps({"rules": [{
        "name": "all_seen",
        "trigger": {"detector": "object_watch", "state": "object_seen"},
    }]}))
    with pytest.raises(ValueError, match="requires trigger.object_id"):
        CustomizationEngine(config)


def test_legacy_transition_rule_without_target_still_matches():
    engine = _engine({"detector": "object_watch", "state": "object_removed"})
    assert engine.evaluate([_event(state="object_removed")])


def test_malformed_object_watch_filter_fails_closed():
    engine = _engine(
        {"detector": "object_watch", "state": "object_seen", "object_id": "chi-carton"},
        "this is not valid Python !!!",
    )
    assert engine.evaluate([_event()]) == []


def test_malformed_filter_keeps_existing_fail_open_semantics_for_other_detectors():
    engine = _engine({"detector": "presence"}, "this is not valid Python !!!")
    event = RawEvent("presence", True, "PRESENT", "medium")
    assert engine.evaluate([event])
