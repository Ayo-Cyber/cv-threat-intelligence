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


def test_onboarding_collects_scene_hints_and_sends_them_with_the_camera() -> None:
    """Both add-camera flows (Configure form and the first-run wizard) must
    collect the operator's scene knowledge and ship it on the camera payload —
    the frontend half of the onboarding→mapper bridge."""
    html = HTML.read_text()
    assert "sceneHintFields(\"ch\")" in html, "Configure add-camera form lost its scene hint fields"
    assert "sceneHintFields(\"wz\")" in html, "wizard lost its scene hint fields"
    assert html.count("sceneHintValues(") >= 3, "hints are not merged into addCamera payloads"
    assert "scene_hint" in html
    # while mapping is pending the panel shows the operator's own notes
    assert "sceneHintSummary" in html


def test_wizard_assigns_camera_to_selected_area() -> None:
    html = HTML.read_text()
    body = _function_body(html, "wizAdd")

    assert "area_id" in body
    assert "wzArea" in body


def test_finishing_setup_starts_monitoring_then_opens_scene_review() -> None:
    body = _function_body(HTML.read_text(), "wizFinish")

    assert body.index('call("startMonitoring"') < body.index("openSceneReview")


def test_scene_review_is_one_grouped_workspace() -> None:
    html = HTML.read_text()
    assert 'id="sceneReview"' in html
    body = _function_body(html, "renderSceneReview")
    assert "summary.areas" in body
    assert "camera_ids" in body
    assert "site_type" in body
    assert "approveSiteContext" in body
    for label in ("Confirm area", "Edit and confirm", "Reject and remap", "Keep paused"):
        assert label in html
