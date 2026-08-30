from __future__ import annotations

from pathlib import Path


HTML = Path("cvti/app/web/index.html")


def _function_body(source: str, function_name: str) -> str:
    start = source.index(f"function {function_name}(")
    next_function = source.find("\nfunction ", start + 1)
    return source[start:] if next_function < 0 else source[start:next_function]


def test_rules_screen_has_mapping_status_review_and_remap_controls() -> None:
    html = HTML.read_text()

    for token in (
        "mapping-status",
        "updateSceneContext",
        "approveSceneContext",
        "requestSceneRemap",
        "acceptSuggestedZone",
        "expected_actors",
    ):
        assert token in html


def test_mapper_load_does_not_auto_accept_suggested_zones() -> None:
    html = HTML.read_text()

    assert "acceptSuggestedZone" not in _function_body(html, "fillScene")


def test_mapper_review_targets_current_web_console() -> None:
    html = HTML.read_text()

    assert "renderSceneMapper" in html
    assert "cvti/app/widgets/mapper.py" not in html


def test_scene_mutation_controls_follow_camera_configuration_permission() -> None:
    html = HTML.read_text()

    assert 'sceneCanEdit()' in html
    assert 'configure_cameras' in _function_body(html, "sceneCanEdit")
