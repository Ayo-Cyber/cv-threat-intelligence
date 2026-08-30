from __future__ import annotations

import json
from pathlib import Path

from cvti.contracts import RawEvent
from cvti.rules.context_compatibility import evaluate_context_compatibility
from cvti.rules.customization import CustomizationEngine


def _scene(environment: str) -> dict:
    return {
        "environment_type": environment,
        "scene_description": "A monitored area.",
        "zones": [
            {
                "id": "suggested_checkout",
                "label": "Suggested checkout",
                "role": "checkout",
                "bbox": [0, 0, 10, 10],
            }
        ],
    }


def _shoplifting_rule(mode: str = "enforce") -> dict:
    return {
        "name": "shoplifting",
        "trigger": {"detector": "concealment"},
        "priority": "high",
        "context_requirements": {
            "environment_types": ["retail_shop", "mall_corridor"],
            "zone_roles_any": ["merchandise", "checkout"],
            "mode": mode,
        },
    }


def _event(detector: str, title: str = "Candidate") -> RawEvent:
    return RawEvent(
        detector=detector,
        active=True,
        title=title,
        level="high",
        timestamp=1.0,
    )


def _engine(tmp_path, rules: list[dict], baseline: list[dict] | None = None):
    config = tmp_path / "rules.json"
    config.write_text(json.dumps({"use_case_id": "test", "rules": rules}))
    baseline_path = None
    if baseline is not None:
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"rules": baseline}))
    return CustomizationEngine(config, baseline_path=baseline_path)


def test_shoplifting_is_suppressed_in_parking_lot() -> None:
    decision = evaluate_context_compatibility(
        _shoplifting_rule(), _scene("parking_lot")
    )

    assert decision.allowed is False
    assert decision.mode == "context_incompatible"
    assert "retail_shop" in decision.reason


def test_matching_environment_or_accepted_zone_role_allows_rule() -> None:
    assert evaluate_context_compatibility(
        _shoplifting_rule(), _scene("retail_shop")
    ).allowed
    assert evaluate_context_compatibility(
        _shoplifting_rule(),
        _scene("parking_lot"),
        active_zone_roles={"checkout"},
    ).allowed


def test_mapper_suggested_zone_does_not_count_as_accepted() -> None:
    decision = evaluate_context_compatibility(
        _shoplifting_rule(), _scene("parking_lot"), active_zone_roles=set()
    )

    assert decision.allowed is False


def test_missing_requirements_preserve_legacy_allow() -> None:
    decision = evaluate_context_compatibility({"name": "legacy"}, None)

    assert decision.allowed is True
    assert decision.mode == "allowed"


def test_override_allows_and_records_override() -> None:
    decision = evaluate_context_compatibility(
        _shoplifting_rule(mode="override"), _scene("parking_lot")
    )

    assert decision.allowed is True
    assert decision.mode == "explicit_override"


def test_unknown_context_fails_enforced_requirement() -> None:
    assert not evaluate_context_compatibility(
        _shoplifting_rule(), _scene("unknown")
    ).allowed


def test_critical_baseline_bypasses_requirements() -> None:
    decision = evaluate_context_compatibility(
        _shoplifting_rule(), _scene("parking_lot"), baseline=True
    )

    assert decision.allowed is True
    assert decision.mode == "critical_baseline"


def test_engine_does_not_emit_context_incompatible_candidate(tmp_path) -> None:
    engine = _engine(tmp_path, [_shoplifting_rule()])

    alerts = engine.evaluate(
        [_event("concealment")], scene_context=_scene("parking_lot")
    )

    assert alerts == []
    assert engine.context_decisions == [
        {
            "rule": "shoplifting",
            "environment": "parking_lot",
            "decision": "context_incompatible",
            "reason": engine.context_decisions[0]["reason"],
        }
    ]


def test_plain_english_gate_question_is_explicit_override(tmp_path) -> None:
    rule = {
        "name": "customer_question",
        "trigger": {"detector": "presence"},
        "priority": "high",
        "gate_question": "Is someone taking stock from a delivery vehicle?",
    }
    engine = _engine(tmp_path, [rule])

    alerts = engine.evaluate(
        [_event("presence")], scene_context=_scene("parking_lot")
    )

    assert alerts[0].rule_name == "customer_question"
    assert engine.context_decisions[0]["decision"] == "explicit_override"


def test_baseline_fire_still_emits_with_parking_context(tmp_path) -> None:
    fire = {
        "name": "baseline_fire_smoke",
        "trigger": {"detector": "fire"},
        "priority": "critical",
        "critical_baseline": True,
    }
    engine = _engine(tmp_path, [], baseline=[fire])

    alerts = engine.evaluate([_event("fire")], scene_context=_scene("parking_lot"))

    assert alerts[0].rule_name == "baseline_fire_smoke"
    assert engine.context_decisions[0]["decision"] == "critical_baseline"


def test_engine_uses_only_separately_accepted_zone_roles(tmp_path) -> None:
    engine = _engine(tmp_path, [_shoplifting_rule()])

    alerts = engine.evaluate(
        [_event("concealment")],
        scene_context=_scene("parking_lot"),
        active_zone_roles={"checkout"},
    )

    assert alerts[0].rule_name == "shoplifting"


def test_always_on_baseline_rules_are_explicitly_context_exempt() -> None:
    rules = json.loads(Path("configs/baseline_critical_v1.json").read_text())["rules"]

    assert rules
    assert all(rule.get("critical_baseline") is True for rule in rules)


def test_shipped_shoplifting_rules_declare_retail_context() -> None:
    paths = (
        "configs/retail_pipeline_v1.json",
        "configs/retail_v1.json",
        "configs/all_threats_v1.json",
        "configs/all_threats_video_v1.json",
    )
    found = 0
    for path in paths:
        rules = json.loads(Path(path).read_text())["rules"]
        for rule in rules:
            if rule.get("name") not in {"shoplifting", "theft_depart"}:
                continue
            found += 1
            requirements = rule["context_requirements"]
            assert requirements["mode"] == "enforce"
            assert "retail_shop" in requirements["environment_types"]
            assert "checkout" in requirements["zone_roles_any"]
    assert found == 6
