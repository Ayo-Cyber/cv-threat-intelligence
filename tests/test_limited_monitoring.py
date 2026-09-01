from __future__ import annotations

import json

from cvti.contracts import RawEvent
from cvti.rules.customization import CustomizationEngine
from cvti.serving.camera import PerCameraState
from cvti.serving.pipeline import activate_reviewed_scene_contexts
from _scene_hierarchy_fixtures import camera_context


def test_unreviewed_camera_runs_critical_baseline_but_not_shoplifting(tmp_path) -> None:
    config = {
        "use_case_id": "limited-test",
        "rules": [{
            "name": "shoplifting",
            "priority": "high",
            "trigger": {"detector": "concealment", "state": "suspected"},
        }],
    }
    baseline = {
        "use_case_id": "baseline-test",
        "rules": [{
            "name": "baseline_fire_smoke",
            "priority": "critical",
            "critical_baseline": True,
            "trigger": {"detector": "fire_smoke", "state": "suspected"},
        }],
    }
    config_path = tmp_path / "customer.json"
    baseline_path = tmp_path / "baseline.json"
    config_path.write_text(json.dumps(config))
    baseline_path.write_text(json.dumps(baseline))
    engine = CustomizationEngine(config_path, baseline_path)

    alerts = engine.evaluate(
        [
            RawEvent("fire_smoke", True, "POSSIBLE FIRE", "critical", state="suspected"),
            RawEvent("concealment", True, "CONCEALMENT", "high", state="suspected"),
        ],
        monitoring_scope="critical_only",
    )

    assert [alert.rule_name for alert in alerts] == ["baseline_fire_smoke"]
    assert any(
        row["decision"] == "awaiting_review" and row["rule"] == "shoplifting"
        for row in engine.context_decisions
    )


def test_full_scope_preserves_existing_rule_behavior(tmp_path) -> None:
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({
        "rules": [{
            "name": "shoplifting",
            "trigger": {"detector": "concealment", "state": "suspected"},
        }],
    }))
    engine = CustomizationEngine(path)

    alerts = engine.evaluate([
        RawEvent("concealment", True, "CONCEALMENT", "high", state="suspected")
    ])

    assert [alert.rule_name for alert in alerts] == ["shoplifting"]


def test_approval_activates_full_rules_without_rebuilding_state() -> None:
    state = object.__new__(PerCameraState)
    state.camera_id = "cam1"
    state.engine = CustomizationEngine()
    state.scene_context = None
    state.monitoring_scope = "critical_only"
    identity = id(state)
    reviewed = camera_context(
        "cam1", site_type="supermarket", area_id="sales", area_type="retail_floor"
    )

    state.activate_scene_context(reviewed)

    assert id(state) == identity
    assert state.monitoring_scope == "full"
    assert state.scene_context == reviewed


def test_running_process_picks_up_reviewed_camera_artifact(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"clip")
    state = object.__new__(PerCameraState)
    state.camera_id = "cam1"
    state.scene_context = None
    state.monitoring_scope = "critical_only"
    from cvti.scene.context_store import SceneContextStore

    SceneContextStore(tmp_path / "context", "cam1").approve(
        camera_context("cam1"), "owner", source
    )

    changed = activate_reviewed_scene_contexts({"cam1": state}, tmp_path)

    assert changed == ["cam1"]
    assert state.monitoring_scope == "full"
