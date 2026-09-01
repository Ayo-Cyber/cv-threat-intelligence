from __future__ import annotations

import inspect
import json

from cvti.scene.context_store import render_scene_context
from cvti.serving import pipeline
from cvti.serving.custom_rules import CustomRuleScanner
from cvti.serving.scene_map import SceneMappingPreflight


def test_run_site_inspects_context_before_states_then_maps_in_background() -> None:
    source = inspect.getsource(pipeline.run_site)

    assert source.index("mapping_service.inspect(") < source.index("build_camera_states(")
    assert source.index("build_camera_states(") < source.index("SceneMappingCoordinator(")
    assert source.index("build_camera_states(") < source.index("pipe.start()")


def test_post_start_background_mapper_is_removed() -> None:
    source = inspect.getsource(pipeline.run_site)

    assert "map_cameras_async" not in source


def test_mapper_defaults_to_local_gate_settings() -> None:
    settings = pipeline.resolve_mapper_settings(
        gate_provider="ollama",
        gate_model="gemma3:4b",
        gate_base_url="",
        mapper_provider="",
        mapper_model="",
        mapper_base_url="",
    )

    assert settings == (
        "ollama",
        "gemma3:4b",
        "http://localhost:11434/v1",
    )


def test_explicit_mapper_settings_override_gate() -> None:
    settings = pipeline.resolve_mapper_settings(
        gate_provider="mock",
        gate_model="",
        gate_base_url="",
        mapper_provider="ollama",
        mapper_model="gemma3:4b-it-qat",
        mapper_base_url="http://127.0.0.1:11434/v1",
    )

    assert settings == (
        "ollama",
        "gemma3:4b-it-qat",
        "http://127.0.0.1:11434/v1",
    )


def test_unreviewed_camera_remains_active_for_critical_only_monitoring() -> None:
    cameras = [{"id": "cam_1"}, {"id": "cam_2"}]
    preflight = SceneMappingPreflight(blocked_camera_ids={"cam_2"})

    assert pipeline.active_cameras_after_preflight(cameras, preflight) == cameras


def test_preflight_statuses_select_monitoring_scope() -> None:
    preflight = SceneMappingPreflight(
        statuses={
            "reviewed": {"status": "ready_reviewed"},
            "pending": {"status": "ready_unreviewed"},
            "failed": {"status": "failed"},
        }
    )

    assert pipeline.monitoring_scopes_from_preflight(preflight) == {
        "reviewed": "full",
        "pending": "critical_only",
        "failed": "critical_only",
    }


def test_rendered_expected_actors_are_possibilities_not_identities() -> None:
    text = render_scene_context(
        {
            "environment_type": "retail_shop",
            "scene_description": "A shop floor.",
            "expected_actors": ["staff", "customers"],
        }
    )

    assert "may include staff, customers" in text
    assert "is staff" not in text


def test_custom_rule_scanner_uses_injected_context_provider() -> None:
    scanner = CustomRuleScanner(
        [],
        sink=None,
        model="gemma3:4b",
        context_provider=lambda camera_id: {
            "environment_type": "parking_lot",
            "scene_description": f"Parking area for {camera_id}.",
            "expected_actors": ["drivers"],
        },
    )

    text = scanner._scene("cam_1")

    assert "Parking area for cam_1" in text
    assert "may include drivers" in text


def test_all_blocked_preflight_writes_degraded_credential_free_health(tmp_path) -> None:
    preflight = SceneMappingPreflight(
        statuses={
            "cam_1": {
                "status": "failed",
                "error": "scene mapper unavailable",
                "provenance": "mapper",
                "usable": False,
                "review_required": False,
                "environment_type": "unknown",
            }
        },
        blocked_camera_ids={"cam_1"},
    )

    pipeline._write_mapping_only_health(
        str(tmp_path), preflight, gate_provider="ollama", gate_model="gemma3:4b"
    )

    document = json.loads((tmp_path / "gate_health.json").read_text())
    serialized = json.dumps(document)
    assert document["status"] == "degraded"
    assert "scene mapping failed" in document["reasons"][0]
    assert "rtsp://" not in serialized
    assert "password" not in serialized
