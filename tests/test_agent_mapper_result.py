from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cvti.scene.agent_mapper import (
    AgentMapper,
    MappingResult,
    SampledFrame,
    load_schema,
    parse_and_validate_scene_context,
)


def test_map_result_returns_context_selected_frame_and_raw_response() -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    sample = SampledFrame(image=frame, timestamp_seconds=2.0, score=1.0)
    mapper = AgentMapper(provider="mock")

    with patch(
        "cvti.scene.agent_mapper.capture_sample_frames", return_value=[sample]
    ) as capture:
        result = mapper.map_result(
            "clip.mp4",
            "cam_1",
            source_frame_path="site/context/cam_1/source_frame.jpg",
        )

    assert isinstance(result, MappingResult)
    assert result.selected_frame is frame
    assert result.raw_response.startswith("{")
    assert result.context["camera_id"] == "cam_1"
    assert (
        result.context["source_frame_path"]
        == "site/context/cam_1/source_frame.jpg"
    )
    capture.assert_called_once_with("clip.mp4", "video_file", 3, 2.0)


def test_map_compatibility_wrapper_returns_only_context() -> None:
    mapper = AgentMapper(provider="mock")
    expected = {"camera_id": "cam_1"}

    with patch.object(mapper, "map_result") as mapped:
        mapped.return_value = MappingResult(expected, object(), "{}")
        result = mapper.map("clip.mp4", "cam_1", sample_count=4)

    assert result == expected
    mapped.assert_called_once_with("clip.mp4", "cam_1", 4)


def test_mapper_returns_visual_site_and_area_candidates() -> None:
    raw = json.dumps({
        "environment_type": "warehouse_floor",
        "scene_description": "A mixing line is visible.",
        "expected_actors": ["staff"],
        "zones": [],
        "confidence": 0.86,
        "notes": "",
        "site_type_candidate": "manufacturing plant",
        "area_type_candidate": "production floor",
        "view_description": "West mixing line.",
    })

    context = parse_and_validate_scene_context(
        raw,
        load_schema(Path("schemas/scene_context.schema.json")),
        camera_id="cam_1",
        source_type="video_file",
        source_frame_path="context/cam_1/source_frame.jpg",
    )

    assert context["site_type_candidate"] == "manufacturing_plant"
    assert context["area_type_candidate"] == "production_floor"
    assert context["view_description"] == "West mixing line."


def test_invalid_hierarchy_candidates_degrade_to_unknown() -> None:
    raw = json.dumps({
        "environment_type": "warehouse_floor",
        "scene_description": "A room is visible.",
        "expected_actors": [],
        "zones": [],
        "confidence": 0.4,
        "notes": "",
        "site_type_candidate": "space_port",
        "area_type_candidate": "moon_lobby",
        "view_description": "",
    })

    context = parse_and_validate_scene_context(
        raw, {}, "cam_1", "video_file", "frame.jpg"
    )

    assert context["site_type_candidate"] == "unknown"
    assert context["area_type_candidate"] == "unknown"
