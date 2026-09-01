from __future__ import annotations

import json

from cvti.scene.coordinator import SceneMappingCoordinator
from cvti.serving.scene_map import FullAgentMapperService
from _scene_hierarchy_fixtures import DeterministicMapper, site_with_areas, write_site


class FailingMapper(DeterministicMapper):
    def map_result(self, source, camera_id, sample_count=3,
                   source_frame_path="", operator_hints=None):
        self.calls.append(camera_id)
        raise RuntimeError("temporary mapper outage")


def _coordinator(tmp_path, mapper):
    site_path = write_site(tmp_path, ["cam1", "cam2", "cam3"])
    output = tmp_path / "out"
    return SceneMappingCoordinator(
        FullAgentMapperService(output, mapper, legacy_root=tmp_path / "legacy"),
        site_path,
        output,
        max_workers=1,
    )


def test_mapping_queue_is_bounded_and_completes_all_cameras(tmp_path) -> None:
    mapper = DeterministicMapper(delay=0.001)
    coordinator = _coordinator(tmp_path, mapper)

    coordinator.enqueue(["cam1", "cam2", "cam3"])
    coordinator.run_until_idle()

    assert mapper.max_active == 1
    assert coordinator.progress().ready == 3
    saved = json.loads((tmp_path / "out/context/mapping_queue.json").read_text())
    assert all("source" not in job for job in saved["jobs"])


def test_interrupted_queue_resumes_without_completed_jobs(tmp_path) -> None:
    first = _coordinator(tmp_path, DeterministicMapper())
    first.enqueue(["cam1", "cam2", "cam3"])
    first.run_next()

    resumed = _coordinator(tmp_path, DeterministicMapper())
    resumed.resume()

    assert resumed.pending_camera_ids() == ["cam2", "cam3"]


def test_enqueue_deduplicates_same_camera_and_source(tmp_path) -> None:
    coordinator = _coordinator(tmp_path, DeterministicMapper())

    coordinator.enqueue(["cam1", "cam1"])

    assert coordinator.progress().total == 1


def test_camera_completion_builds_area_and_site_proposals(tmp_path) -> None:
    coordinator = _coordinator(tmp_path, DeterministicMapper())
    coordinator.enqueue(["cam1"])

    coordinator.run_until_idle()

    assert (tmp_path / "out/context/_areas/production/area_context.proposal.json").exists()
    assert (tmp_path / "out/context/_site/site_context.proposal.json").exists()


def test_area_metadata_reaches_each_camera_as_a_prior(tmp_path) -> None:
    site = write_site(tmp_path, ["cam1"])
    payload = json.loads(site.read_text())
    payload["site_type"] = "manufacturing_plant"
    payload["areas"][0]["area_type"] = "production_floor"
    site.write_text(json.dumps(payload))
    mapper = DeterministicMapper()
    output = tmp_path / "out"
    coordinator = SceneMappingCoordinator(
        FullAgentMapperService(output, mapper, legacy_root=tmp_path / "legacy"),
        site,
        output,
    )

    coordinator.enqueue(["cam1"])
    coordinator.run_until_idle()

    assert mapper.hints_seen == [{
        "site_type": "manufacturing_plant",
        "area_id": "production",
        "area_type": "production_floor",
    }]


def test_one_hundred_camera_site_maps_with_one_local_worker(tmp_path) -> None:
    site = site_with_areas(area_count=20, cameras_per_area=5)
    site_path = tmp_path / "site.json"
    site_path.write_text(json.dumps(site))
    output = tmp_path / "out"
    mapper = DeterministicMapper()
    coordinator = SceneMappingCoordinator(
        FullAgentMapperService(output, mapper, legacy_root=tmp_path / "legacy"),
        site_path,
        output,
        max_workers=1,
    )

    coordinator.enqueue([camera["id"] for camera in site["cameras"]])
    coordinator.run_until_idle()

    progress = coordinator.progress()
    assert progress.total == 100
    assert progress.ready == 100
    assert progress.failed == 0
    assert mapper.max_active == 1


def test_failed_job_can_be_requeued_with_a_bounded_attempt_limit(tmp_path) -> None:
    coordinator = _coordinator(tmp_path, FailingMapper())
    coordinator.enqueue(["cam1"])
    coordinator.run_until_idle()

    assert coordinator.progress().failed == 1
    assert coordinator.requeue_failed(max_attempts=2) == ["cam1"]
    coordinator.run_until_idle()

    assert coordinator.progress().failed == 1
    assert coordinator.requeue_failed(max_attempts=2) == []


def test_manual_policy_never_calls_the_mapper(tmp_path) -> None:
    mapper = DeterministicMapper()
    site_path = write_site(tmp_path, ["cam1"])
    output = tmp_path / "out"
    coordinator = SceneMappingCoordinator(
        FullAgentMapperService(output, mapper, legacy_root=tmp_path / "legacy"),
        site_path,
        output,
        policy="manual",
    )

    coordinator.enqueue(["cam1"])
    coordinator.run_until_idle()

    assert mapper.calls == []
    assert coordinator.progress().failed == 1
