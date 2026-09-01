from __future__ import annotations

import json

import numpy as np

from cvti.scene.agent_mapper import MappingResult
from cvti.scene.context_store import SceneContextStore
from cvti.serving.camera import build_camera_states
from cvti.serving.scene_map import FullAgentMapperService


def _context(camera_id: str = "cam_1", environment: str = "retail_shop") -> dict:
    return {
        "camera_id": camera_id,
        "source_type": "video_file",
        "environment_type": environment,
        "scene_description": "A monitored customer area.",
        "expected_actors": ["staff"],
        "zones": [],
        "confidence": 0.8,
        "generated_at": "2026-08-30T10:00:00Z",
        "source_frame_path": "unused.jpg",
        "notes": "",
    }


class RecordingMapper:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def map_result(
        self,
        source: str,
        camera_id: str,
        sample_count: int = 3,
        source_frame_path: str = "",
        operator_hints: dict | None = None,
    ) -> MappingResult:
        self.calls.append((source, camera_id, sample_count, source_frame_path))
        self.hints_seen = getattr(self, "hints_seen", []) + [operator_hints]
        context = _context(camera_id)
        context["source_frame_path"] = source_frame_path
        return MappingResult(
            context,
            np.full((40, 60, 3), 120, dtype=np.uint8),
            '{"environment_type":"retail_shop"}',
        )


class FailingMapper:
    def map_result(self, *args, **kwargs):
        raise RuntimeError("ollama unavailable")


def _source(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"clip")
    return str(path)


def _camera(tmp_path, **overrides) -> dict:
    camera = {"id": "cam_1", "source": _source(tmp_path)}
    camera.update(overrides)
    return camera


def test_reviewed_cache_avoids_mapper_call(tmp_path) -> None:
    source = _source(tmp_path)
    store = SceneContextStore(tmp_path / "out/context", "cam_1")
    store.approve(_context(), "owner", source)
    mapper = RecordingMapper()
    service = FullAgentMapperService(tmp_path / "out", mapper)

    result = service.prepare([{"id": "cam_1", "source": source}], "require_reviewed")

    assert result.contexts["cam_1"]["camera_id"] == "cam_1"
    assert result.blocked_camera_ids == set()
    assert mapper.calls == []


def test_auto_policy_maps_missing_context_and_returns_it_usable(tmp_path) -> None:
    mapper = RecordingMapper()
    service = FullAgentMapperService(tmp_path / "out", mapper)

    result = service.prepare([_camera(tmp_path)], "auto")

    assert result.blocked_camera_ids == set()
    assert result.contexts["cam_1"]["environment_type"] == "retail_shop"
    assert len(mapper.calls) == 1
    assert mapper.calls[0][:3] == (_camera(tmp_path)["source"], "cam_1", 3)


def test_require_reviewed_maps_but_blocks_unreviewed_camera(tmp_path) -> None:
    service = FullAgentMapperService(tmp_path / "out", RecordingMapper())

    result = service.prepare([_camera(tmp_path)], "require_reviewed")

    assert result.blocked_camera_ids == {"cam_1"}
    assert result.statuses["cam_1"]["status"] == "ready_unreviewed"
    assert "cam_1" not in result.contexts


def test_manual_policy_blocks_without_calling_mapper(tmp_path) -> None:
    mapper = RecordingMapper()
    service = FullAgentMapperService(tmp_path / "out", mapper)

    result = service.prepare([_camera(tmp_path)], "manual")

    assert result.blocked_camera_ids == {"cam_1"}
    assert mapper.calls == []


def test_mapper_failure_under_auto_runs_the_camera_and_stays_loud(tmp_path) -> None:
    """Changed 30 Aug after the E2E harness proved the original semantics
    fail-CLOSED: one missing prompt file and the engine started with zero
    cameras watching. Under the default policy a mapper failure now runs the
    camera WITHOUT scene context — no fabricated context, the failure stays
    in the health rows — because 'monitoring without context' must never be
    traded for 'no monitoring at all' by default."""
    service = FullAgentMapperService(tmp_path / "out", FailingMapper())

    result = service.prepare([_camera(tmp_path)], "auto")

    assert result.blocked_camera_ids == set(), "auto policy blocked a camera"
    assert result.statuses["cam_1"]["status"] == "failed"
    assert "ollama unavailable" in result.statuses["cam_1"]["error"]
    assert "cam_1" not in result.contexts     # nothing fabricated


def test_mapper_failure_under_strict_policies_still_blocks(tmp_path) -> None:
    """require_reviewed keeps its teeth — there the operator explicitly chose
    certainty over coverage."""
    service = FullAgentMapperService(tmp_path / "out", FailingMapper())

    result = service.prepare([_camera(tmp_path)], "require_reviewed")

    assert result.blocked_camera_ids == {"cam_1"}


def test_static_camera_context_is_reviewed_and_never_calls_mapper(tmp_path) -> None:
    mapper = RecordingMapper()
    service = FullAgentMapperService(tmp_path / "out", mapper)
    camera = _camera(
        tmp_path,
        environment_type="parking_lot",
        scene_description="A monitored car park.",
        expected_actors=["drivers", "security staff"],
    )

    result = service.prepare([camera], "require_reviewed")

    assert result.blocked_camera_ids == set()
    assert result.statuses["cam_1"]["status"] == "ready_reviewed"
    assert result.contexts["cam_1"]["expected_actors"] == [
        "drivers",
        "security staff",
    ]
    assert mapper.calls == []


def test_build_camera_states_uses_pre_resolved_context_and_accepted_roles(
    tmp_path,
) -> None:
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"use_case_id": "test", "rules": []}))
    source = _source(tmp_path)
    site = {
        "cameras": [
            {
                "id": "cam_1",
                "source": source,
                "config": str(rules),
                "accepted_zone_roles": ["checkout"],
            }
        ]
    }

    states = build_camera_states(site, scene_contexts={"cam_1": _context()})
    state = states["cam_1"]["state"]

    assert state.scene_context["expected_actors"] == ["staff"]
    assert state.active_zone_roles == {"checkout"}


def test_human_named_cameras_map_instead_of_killing_the_engine(tmp_path) -> None:
    """v1.5.0 crash-looped at startup on EVERY feed with a human-named camera
    ('Dublin Street', 'Forecourt ATM' — i.e. the live and demo feeds): the
    store rejected the id, the ValueError escaped prepare(), and the engine
    died before its first frame. The id is the site's business; only the
    directory must be filesystem-safe."""
    mapper = RecordingMapper()
    # explicit legacy_root: the default is cwd-relative, and this repo's own
    # runs/context/ still holds 'Dublin Street' contexts from the pre-mapper
    # era — which is itself proof these ids were always legitimate.
    service = FullAgentMapperService(tmp_path / "out", mapper,
                                     legacy_root=tmp_path / "legacy")

    result = service.prepare(
        [_camera(tmp_path, id="Dublin Street"),
         _camera(tmp_path, id="Forecourt ATM")], "auto")

    assert result.blocked_camera_ids == set()
    assert set(result.contexts) == {"Dublin Street", "Forecourt ATM"}
    assert len(mapper.calls) == 2


def test_a_preflight_exception_degrades_to_failed_never_raises(tmp_path) -> None:
    """No exception from the mapper preflight may take down monitoring: it
    degrades to a failed mapping status — auto runs the camera generic and
    loud, the strict policies block that camera only."""
    class ExplodingService(FullAgentMapperService):
        def _prepare_camera(self, camera, policy):
            raise RuntimeError("boom from deep inside the preflight")

    service = ExplodingService(tmp_path / "out", RecordingMapper())

    result = service.prepare([_camera(tmp_path)], "auto")
    assert result.blocked_camera_ids == set()
    assert result.statuses["cam_1"]["status"] == "failed"
    assert "boom" in result.statuses["cam_1"]["error"]

    strict = ExplodingService(tmp_path / "out2", RecordingMapper())
    blocked = strict.prepare([_camera(tmp_path)], "require_reviewed")
    assert blocked.blocked_camera_ids == {"cam_1"}


def test_onboarding_hints_reach_the_mapper_as_priors(tmp_path) -> None:
    """The discord this heals (co-engineer, 1 Sep): onboarding collected
    knowledge the mapper never saw. Hints are PRIORS — the mapper still runs
    (unlike a full scene_description, which is human-authored and skips it)
    but anchored to what the operator said."""
    mapper = RecordingMapper()
    service = FullAgentMapperService(tmp_path / "out", mapper,
                                     legacy_root=tmp_path / "legacy")

    service.prepare([_camera(
        tmp_path,
        environment_type="parking_lot",
        expected_actors=["staff", "drivers"],
        scene_hint="cars park along the left fence",
    )], "auto")

    assert len(mapper.calls) == 1, "hints must not bypass the mapper"
    assert mapper.hints_seen == [{
        "environment_type": "parking_lot",
        "expected_actors": ["staff", "drivers"],
        "note": "cars park along the left fence",
    }]


def test_no_hints_means_no_priors_and_manual_still_bypasses(tmp_path) -> None:
    mapper = RecordingMapper()
    service = FullAgentMapperService(tmp_path / "out", mapper,
                                     legacy_root=tmp_path / "legacy")
    service.prepare([_camera(tmp_path)], "auto")
    assert mapper.hints_seen == [None]

    authored = FullAgentMapperService(tmp_path / "out2", mapper,
                                      legacy_root=tmp_path / "legacy")
    result = authored.prepare([_camera(
        tmp_path, id="cam_2", scene_description="A watched forecourt.",
    )], "auto")
    assert len(mapper.calls) == 1, "a full description is authored, not a hint"
    assert result.contexts["cam_2"]["scene_description"] == "A watched forecourt."


def test_shared_area_is_a_prior_and_each_camera_still_maps(tmp_path) -> None:
    mapper = RecordingMapper()
    service = FullAgentMapperService(
        tmp_path / "out", mapper, legacy_root=tmp_path / "legacy"
    )

    service.prepare([
        _camera(
            tmp_path, id="north", area_id="loading", area_type="loading_bay",
            site_type="manufacturing_plant",
        ),
        _camera(
            tmp_path, id="south", area_id="loading", area_type="loading_bay",
            site_type="manufacturing_plant",
        ),
    ], "auto")

    assert [call[1] for call in mapper.calls] == ["north", "south"]
    assert mapper.hints_seen == [
        {
            "site_type": "manufacturing_plant",
            "area_id": "loading",
            "area_type": "loading_bay",
        },
        {
            "site_type": "manufacturing_plant",
            "area_id": "loading",
            "area_type": "loading_bay",
        },
    ]


def test_inspect_existing_context_never_calls_mapper(tmp_path) -> None:
    mapper = RecordingMapper()
    service = FullAgentMapperService(
        tmp_path / "out", mapper, legacy_root=tmp_path / "legacy"
    )

    result = service.inspect([_camera(tmp_path)], "require_reviewed")

    assert mapper.calls == []
    assert result.statuses["cam_1"]["status"] == "pending"
    assert result.blocked_camera_ids == {"cam_1"}
