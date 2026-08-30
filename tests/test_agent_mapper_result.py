from __future__ import annotations

from unittest.mock import patch

import numpy as np

from cvti.scene.agent_mapper import AgentMapper, MappingResult, SampledFrame


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
