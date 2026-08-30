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
