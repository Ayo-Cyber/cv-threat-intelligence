from __future__ import annotations

import inspect

from cvti.scene.context_store import normalize_scene_context_policy
from cvti.serving import scene_map


def test_serving_mapper_has_no_duplicate_prompt_parser_or_background_api() -> None:
    source = inspect.getsource(scene_map)

    assert "SCENE_PROMPT" not in source
    assert "def _parse(" not in source
    assert "def _sample_frame(" not in source
    assert "map_cameras_async" not in source


def test_legacy_policy_defaults_to_auto() -> None:
    assert normalize_scene_context_policy(None) == "auto"
    assert normalize_scene_context_policy("") == "auto"


def test_preflight_status_never_contains_camera_source() -> None:
    fields = scene_map.SceneMappingPreflight().statuses

    assert fields == {}


def test_prompt_carries_operator_hints_as_priors() -> None:
    from cvti.scene.agent_mapper import build_prompt

    prompt = build_prompt(
        template="MAP THE SCENE.",
        camera_id="Forecourt ATM",
        source_type="rtsp",
        source_frame_path="frame.jpg",
        max_zone_suggestions=3,
        operator_hints={"environment_type": "atm_area",
                        "expected_actors": ["customers", "guard"],
                        "note": "kiosk on the right"},
    )
    assert "site operator described this camera" in prompt
    assert '"atm_area"' in prompt
    assert "customers, guard" in prompt
    assert "kiosk on the right" in prompt
    # unless-contradicted framing: priors anchor, they do not dictate
    assert "unless the frame clearly contradicts" in prompt

    bare = build_prompt(template="MAP.", camera_id="c", source_type="rtsp",
                        source_frame_path="f.jpg", max_zone_suggestions=3)
    assert "site operator" not in bare


def test_prompt_requests_independent_hierarchy_evidence() -> None:
    from cvti.scene.agent_mapper import build_prompt

    prompt = build_prompt(
        template="MAP.",
        camera_id="cam1",
        source_type="rtsp",
        source_frame_path="frame.jpg",
        max_zone_suggestions=3,
        operator_hints={
            "site_type": "manufacturing_plant",
            "area_id": "loading",
            "area_type": "loading_bay",
        },
    )

    assert "independent visual observation" in prompt
    assert 'site type prior: "manufacturing_plant"' in prompt
    assert 'area type prior: "loading_bay"' in prompt
    assert "site_type_candidate" in prompt
    assert "area_type_candidate" in prompt
