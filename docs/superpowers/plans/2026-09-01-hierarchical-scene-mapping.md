# Hierarchical Scene Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every camera infer its own scene while Argus groups related cameras into reviewable areas, infers site/area context, and opens an immediate post-onboarding review flow that scales to 100 cameras.

**Architecture:** Preserve the current per-camera `SceneContext` and artifact paths, then add site and area contracts, deterministic aggregation, and a bounded persistent mapping coordinator. Cameras awaiting semantic review run critical-baseline-only monitoring; reviewed context enables the full rule pack. The existing Argus web console gains area assignment during onboarding and one grouped Scene Review workspace rather than one popup per camera.

**Tech Stack:** Python 3.9, dataclasses, pathlib/JSON, OpenCV, local Ollama/OpenAI-compatible API, pytest/unittest, Qt WebChannel, vanilla HTML/CSS/JavaScript.

**Spec:** `docs/superpowers/specs/2026-09-01-hierarchical-scene-mapping-design.md`

## Global Constraints

- Production base is `ayo/main@126b53a` (`v1.6.1`).
- Every camera must perform independent image-based mapping; an area assignment is only a prior.
- Existing `scene_context.json` files and `<output_dir>/context/<camera_id>/` paths remain valid.
- Existing `environment_type` values remain accepted and continue driving legacy rules.
- Site/area aggregation is deterministic; it must not make another VLM call.
- Mapper suggestions never activate threat policy or zones automatically.
- Camera-specific zone coordinates are never copied between camera angles.
- Unreviewed cameras run always-on critical baselines only; context-dependent rules remain paused.
- RTSP credentials never appear in artifacts, health, logs, or UI.
- Legacy site files are never rewritten merely by loading them.
- Mutations remain role-gated and audited.
- Do not commit intermediate tasks. Keep the design, plan, code, tests, and docs uncommitted until all verification is green, then create one implementation commit and push it.

---

## File Structure

### New files

- `schemas/site_context.schema.json`: bounded site-level context contract.
- `schemas/area_context.schema.json`: bounded shared-area context contract.
- `cvti/scene/hierarchy.py`: vocabularies, aliases, hierarchy validation, migration-safe site helpers, and artifact persistence.
- `cvti/scene/aggregation.py`: pure camera-to-area and area-to-site proposal logic.
- `cvti/scene/coordinator.py`: persistent bounded mapping queue and progress model.
- `tests/test_scene_hierarchy.py`: contracts, migration, paths, and site config behavior.
- `tests/test_scene_aggregation.py`: agreement, ambiguity, conflict, and precedence.
- `tests/test_scene_coordinator.py`: queue persistence, bounded work, retries, and recomputation.
- `tests/test_limited_monitoring.py`: critical-only behavior and live context activation.
- `tests/test_scene_review_flow.py`: backend/bridge and grouped onboarding review flow.
- `tests/_scene_hierarchy_fixtures.py`: literal hierarchy payloads and deterministic test doubles shared by the new tests.

### Modified files

- `schemas/scene_context.schema.json`: optional hierarchy candidates and view description.
- `prompts/agent_mapper_prompt.txt`: request independent site/area candidates while treating hints as priors.
- `cvti/scene/agent_mapper.py`: hierarchy vocabularies, defaults, parsing, and prompt hints.
- `cvti/scene/context_store.py`: permit additive camera hierarchy fields and compose reviewed context text.
- `cvti/scene/__init__.py`: export hierarchy, aggregation, and coordinator interfaces.
- `cvti/serving/onboarding.py`: area CRUD, camera assignment, and migration-safe hierarchy reads.
- `cvti/serving/scene_map.py`: map one camera through the coordinator and pass hierarchy priors.
- `cvti/serving/pipeline.py`: start reviewed/full and unreviewed/critical-only states, run mapping worker, and apply approved contexts live.
- `cvti/serving/camera.py`: monitoring scope and live context activation on each camera state.
- `cvti/rules/customization.py`: suppress noncritical rules while scope is `critical_only`.
- `cvti/serving/health_doc.py`: mapping totals, conflicts, queue progress, and limited-monitoring state.
- `cvti/app/console_backend.py`: hierarchy/review APIs, permissions, audit, and coordinator commands.
- `cvti/app/bridge.py`: Qt slots for hierarchy and review methods.
- `cvti/app/web/index.html`: area-aware onboarding and automatic grouped Scene Review workspace.
- `docs/PROJECT_CONTEXT.md`, `plan.md`, and `docs/AGENT_MAPPER_OPERATIONS.md`: current behavior and operator commands.

### Shared Test Fixtures

Task 1 creates `tests/_scene_hierarchy_fixtures.py`. Later tasks import these
fixtures rather than inventing undefined helpers:

```python
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from cvti.scene.agent_mapper import MappingResult

FIXED_NOW = "2026-09-01T12:00:00Z"


def camera_context(
    camera_id="cam1", site_type="manufacturing_plant",
    area_id="production", area_type="production_floor", confidence=.9,
):
    return {
        "camera_id": camera_id,
        "source_type": "video_file",
        "environment_type": area_type,
        "scene_description": f"View from {camera_id}.",
        "expected_actors": ["staff"],
        "zones": [],
        "confidence": confidence,
        "generated_at": FIXED_NOW,
        "source_frame_path": f"context/{camera_id}/source_frame.jpg",
        "notes": "",
        "area_id": area_id,
        "site_type_candidate": site_type,
        "area_type_candidate": area_type,
        "view_description": f"Independent view from {camera_id}.",
    }


def site_context(site_type="manufacturing_plant"):
    return {
        "site_id": "test_site",
        "site_type": site_type,
        "site_description": "A test deployment.",
        "confidence": .9,
        "evidence_area_ids": ["production"],
        "generated_at": FIXED_NOW,
    }


def area_context(area_type="production_floor", area_id="production"):
    return {
        "area_id": area_id,
        "name": "Production",
        "site_type": "manufacturing_plant",
        "area_type": area_type,
        "area_description": "A shared test area.",
        "expected_actors": ["staff"],
        "confidence": .9,
        "evidence_camera_ids": ["cam1"],
        "conflicts": [],
        "generated_at": FIXED_NOW,
    }


def camera_observation(camera_id, site_type, area_type, confidence):
    return camera_context(camera_id, site_type, "shared", area_type, confidence)


def area(area_type, area_id="area1"):
    value = area_context(area_type, area_id)
    value["site_type"] = "unknown"
    return value


def write_site(tmp_path: Path, camera_ids, grouped=True) -> Path:
    path = tmp_path / "site.json"
    areas = [{"id": "production", "name": "Production"}] if grouped else []
    cameras = [
        {
            "id": camera_id,
            "source": str(tmp_path / f"{camera_id}.mp4"),
            **({"area_id": "production"} if grouped else {}),
        }
        for camera_id in camera_ids
    ]
    for camera in cameras:
        Path(camera["source"]).write_bytes(b"clip")
    path.write_text(json.dumps({"areas": areas, "cameras": cameras}, indent=2))
    return path


def site_with_areas(area_count, cameras_per_area):
    areas = [{"id": f"area_{i}", "name": f"Area {i}"} for i in range(area_count)]
    cameras = [
        {"id": f"cam_{a}_{c}", "source": f"clip_{a}_{c}.mp4", "area_id": f"area_{a}"}
        for a in range(area_count) for c in range(cameras_per_area)
    ]
    return {"site_id": "scale_test", "areas": areas, "cameras": cameras}


class DeterministicMapper:
    def __init__(self, delay=0):
        self.delay = delay
        self.calls = []
        self.hints_seen = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def map_result(self, source, camera_id, sample_count=3,
                   source_frame_path="", operator_hints=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls.append(camera_id)
            self.hints_seen.append(dict(operator_hints or {}))
            context = camera_context(camera_id)
            context["source_frame_path"] = source_frame_path
            return MappingResult(context, np.zeros((40, 60, 3), dtype=np.uint8), "{}")
        finally:
            with self.lock:
                self.active -= 1


def mapper_payload(**overrides):
    payload = camera_context()
    for owned in ("camera_id", "source_type", "generated_at", "source_frame_path"):
        payload.pop(owned, None)
    payload.update(overrides)
    return json.dumps(payload)


def coordinator_for(tmp_path: Path, mapper, max_workers=1):
    # Imports stay local because coordinator.py is not created until Task 5;
    # earlier hierarchy tests must still be able to import this fixture module.
    from cvti.scene.coordinator import SceneMappingCoordinator
    from cvti.serving.scene_map import FullAgentMapperService

    camera_ids = ["cam1", "cam2", "cam3"]
    site_path = write_site(tmp_path, camera_ids)
    output_dir = tmp_path / "out"
    service = FullAgentMapperService(output_dir, mapper)
    return SceneMappingCoordinator(
        service, site_path, output_dir, max_workers=max_workers)
```

`tests/test_limited_monitoring.py` defines its engine with a literal temporary
rule config containing exactly one `critical_baseline: true` fire rule and one
ordinary shoplifting rule. It constructs a real `CustomizationEngine`; for the
isolated activation test it allocates a `PerCameraState` without calling the
model-loading `__post_init__`, then exercises the real activation method.

`tests/test_scene_review_flow.py` defines `grouped_backend(tmp_path, role)` by
calling the existing `tests/_backend_helper.py::signed_in` against a site from
`write_site(tmp_path, ["north", "south", "exit"])`. `area_payload()` is
`area_context()` with no transformation.

---

### Task 1: Site, Area, and Camera Context Contracts

**Files:**
- Create: `schemas/site_context.schema.json`
- Create: `schemas/area_context.schema.json`
- Create: `cvti/scene/hierarchy.py`
- Modify: `schemas/scene_context.schema.json`
- Modify: `cvti/scene/context_store.py`
- Modify: `cvti/scene/__init__.py`
- Create: `tests/test_scene_hierarchy.py`
- Create: `tests/_scene_hierarchy_fixtures.py`

**Interfaces:**
- Produces: `SITE_TYPES`, `AREA_TYPES`, `AREA_TYPE_ALIASES`.
- Produces: `validate_site_context(payload: dict) -> dict`.
- Produces: `validate_area_context(payload: dict) -> dict`.
- Produces: `normalize_site_type(value: Any) -> str` and `normalize_area_type(value: Any) -> str`.
- Produces: `HierarchyContextStore(context_root: Path)` with `.load_site()`, `.save_site_proposal()`, `.approve_site()`, `.load_area()`, `.save_area_proposal()`, and `.approve_area()`.
- Extends camera context with optional `area_id`, `site_type_candidate`, `area_type_candidate`, and `view_description`.

- [ ] **Step 1: Write failing vocabulary and schema tests**

```python
def test_manufacturing_and_supermarket_are_distinct_site_types():
    assert normalize_site_type("manufacturing plant") == "manufacturing_plant"
    assert normalize_site_type("supermarket") == "supermarket"


def test_camera_context_accepts_additive_hierarchy_fields():
    context = camera_context()
    context.update({
        "area_id": "production",
        "site_type_candidate": "manufacturing_plant",
        "area_type_candidate": "production_floor",
        "view_description": "Mixing line and west exit.",
    })
    assert validate_scene_context(context)["area_id"] == "production"


def test_site_and_area_contexts_reject_unknown_fields():
    site = site_context()
    site["risk_hints"] = ["theft"]
    with pytest.raises(ValueError, match="risk_hints"):
        validate_site_context(site)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_scene_hierarchy.py -q`

Expected: import failure for `cvti.scene.hierarchy`.

- [ ] **Step 3: Add bounded vocabularies and validators**

```python
SITE_TYPES = {
    "manufacturing_plant", "supermarket", "retail_store", "bank",
    "warehouse", "office_building", "residential_estate", "shopping_mall",
    "school", "hospital", "hotel", "transport_hub", "mixed_use", "unknown",
}

AREA_TYPES = {
    "production_floor", "assembly_line", "loading_bay", "storage_aisle",
    "chemical_store", "machine_room", "reception", "retail_floor",
    "checkout", "banking_hall", "vault_approach", "office_floor",
    "parking_lot", "perimeter", "entrance", "walkway", "unknown",
    *ALLOWED_ENVIRONMENT_TYPES,
}
```

Validators must enforce exact keys, unique non-empty evidence IDs/actors,
confidence in `[0, 1]`, parseable timestamps, non-empty descriptions, and
canonical enum values. Camera hierarchy fields are optional so old artifacts
remain valid.

- [ ] **Step 4: Write failing artifact-layout and approval tests**

```python
def test_hierarchy_store_preserves_existing_camera_namespace(tmp_path):
    store = HierarchyContextStore(tmp_path / "context")
    assert store.site_dir == tmp_path / "context/_site"
    assert store.area_dir("Production Floor").parent == tmp_path / "context/_areas"


def test_reviewed_area_outranks_new_proposal(tmp_path):
    store = HierarchyContextStore(tmp_path / "context")
    store.approve_area(area_context("production_floor"), "owner")
    store.save_area_proposal(area_context("warehouse_floor"))
    assert store.load_area("production")["area_type"] == "production_floor"
```

- [ ] **Step 5: Implement atomic hierarchy persistence**

Use the existing atomic JSON helper and `MappingStatus`. Save proposals and
reviewed documents separately so a new proposal never overwrites an approved
context. Slug unsafe IDs with the same collision-resistant strategy as
`SceneContextStore`.

- [ ] **Step 6: Run Task 1 tests and existing context-store tests**

Run: `python -m pytest tests/test_scene_hierarchy.py tests/test_scene_context_store.py -q`

Expected: all pass.

---

### Task 2: Migration-Safe Areas in Site Configuration

**Files:**
- Modify: `cvti/serving/onboarding.py`
- Create/modify: `tests/test_scene_hierarchy.py`
- Modify: `tests/test_onboarding.py`

**Interfaces:**
- Produces: `normalized_areas(site_path) -> list[dict]` without writing.
- Produces: `upsert_area(site_path, area: dict) -> list[dict]`.
- Produces: `remove_area(site_path, area_id: str) -> list[dict]` and moves member cameras to implicit ungrouped areas.
- Produces: `assign_camera_area(site_path, camera_id: str, area_id: str) -> dict`.
- Produces: `camera_area_id(camera: dict) -> str`, stable for legacy cameras.

- [ ] **Step 1: Write failing legacy and assignment tests**

```python
def test_loading_legacy_site_does_not_rewrite_it(tmp_path):
    site = tmp_path / "site.json"
    original = '{"configured":true,"cameras":[{"id":"Front Gate","source":"x"}]}'
    site.write_text(original)
    areas = onboarding.normalized_areas(site)
    assert len(areas) == 1
    assert areas[0]["implicit"] is True
    assert site.read_text() == original


def test_three_cameras_can_share_one_area(tmp_path):
    site = write_site(tmp_path, ["north", "south", "exit"])
    onboarding.upsert_area(site, {"id": "production", "name": "Production"})
    for camera_id in ("north", "south", "exit"):
        onboarding.assign_camera_area(site, camera_id, "production")
    assert {c["area_id"] for c in onboarding.list_cameras(site)} == {"production"}
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_onboarding.py tests/test_scene_hierarchy.py -q`

Expected: missing `normalized_areas`/`assign_camera_area`.

- [ ] **Step 3: Implement area helpers with one membership source**

Areas contain `id`, `name`, and optional hints only. Membership exists only on
`camera.area_id`. Validate area IDs and reject assignments to unknown explicit
areas. Derive implicit IDs as `camera--<safe-camera-id>` in memory.

- [ ] **Step 4: Verify migration and camera CRUD**

Run: `python -m pytest tests/test_onboarding.py tests/test_console_backend.py tests/test_scene_hierarchy.py -q`

Expected: all pass and legacy fixtures remain byte-for-byte unchanged after reads.

---

### Task 3: Independent Hierarchical Camera Inference

**Files:**
- Modify: `prompts/agent_mapper_prompt.txt`
- Modify: `cvti/scene/agent_mapper.py`
- Modify: `cvti/serving/scene_map.py`
- Modify: `tests/test_agent_mapper_result.py`
- Modify: `tests/test_scene_map.py`
- Modify: `tests/test_scene_mapping_preflight.py`

**Interfaces:**
- Extends: `AgentMapper.map_result(..., operator_hints=None)` output with hierarchy candidates.
- Consumes hierarchy hints shaped as `site_type`, `area_id`, `area_type`, `expected_actors`, and `note`.
- Preserves: one independent visual request and one representative frame per camera.

- [ ] **Step 1: Write failing mapper-contract tests**

```python
def test_mapper_returns_visual_site_and_area_candidates():
    raw = mapper_payload(
        environment_type="production_floor",
        site_type_candidate="manufacturing_plant",
        area_type_candidate="production_floor",
        view_description="West mixing line.",
    )
    context = parse_and_validate_scene_context(
        raw,
        load_schema(Path("schemas/scene_context.schema.json")),
        camera_id="cam_1",
        source_type="video_file",
        source_frame_path="context/cam_1/source_frame.jpg",
    )
    assert context["site_type_candidate"] == "manufacturing_plant"
    assert context["area_type_candidate"] == "production_floor"


def test_operator_area_hint_is_a_prior_not_a_mapper_bypass(tmp_path):
    mapper = DeterministicMapper()
    service = FullAgentMapperService(tmp_path, mapper)
    source = tmp_path / "cam1.mp4"
    source.write_bytes(b"clip")
    service.prepare([{
        "id": "cam1",
        "source": str(source),
        "area_id": "loading",
        "area_type": "loading_bay",
    }], "auto")
    assert mapper.calls == ["cam1"]
    assert mapper.hints_seen[0]["area_type"] == "loading_bay"
```

Extend `DeterministicMapper` with `hints_seen`; append a defensive copy of
`operator_hints` on every call so this test proves the prior reached the VLM
boundary without replacing the independent image request.

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_agent_mapper_result.py tests/test_scene_map.py tests/test_scene_mapping_preflight.py -q`

Expected: missing hierarchy fields/hints.

- [ ] **Step 3: Extend prompt and parser**

The prompt must state that operator hints are fallible priors and require
`unknown` when imagery is insufficient. `parse_and_validate_scene_context`
normalizes invalid candidate values to `unknown`; it must never copy hints into
the response after parsing.

- [ ] **Step 4: Verify independent mapping call count**

Add a test with three cameras in one area and assert three mapper calls, three
different `camera_id` values, and one shared `area_id` prior.

- [ ] **Step 5: Run mapper/preflight suite**

Run: `python -m pytest tests/test_agent_mapper_result.py tests/test_scene_map.py tests/test_scene_mapping_preflight.py -q`

Expected: all pass.

---

### Task 4: Deterministic Area and Site Aggregation

**Files:**
- Create: `cvti/scene/aggregation.py`
- Modify: `cvti/scene/__init__.py`
- Create: `tests/test_scene_aggregation.py`

**Interfaces:**
- Produces: `AggregationConflict(field, camera_ids, values, reason)`.
- Produces: `AreaProposal(context: dict, conflicts: list[AggregationConflict], bulk_reviewable: bool)`.
- Produces: `aggregate_area(area: dict, camera_contexts: list[dict], reviewed: dict | None = None) -> AreaProposal`.
- Produces: `aggregate_site(site: dict, area_contexts: list[dict], reviewed: dict | None = None) -> dict`.

- [ ] **Step 1: Write failing agreement/conflict tests**

```python
def test_three_agreeing_camera_views_produce_one_area_proposal():
    proposal = aggregate_area(
        {"id": "production", "name": "Production"},
        [camera_observation("cam1", "manufacturing_plant", "production_floor", .9),
         camera_observation("cam2", "manufacturing_plant", "production_floor", .8),
         camera_observation("cam3", "manufacturing_plant", "production_floor", .7)],
    )
    assert proposal.context["area_type"] == "production_floor"
    assert proposal.bulk_reviewable is True


def test_high_confidence_disagreement_is_visible_not_averaged_away():
    proposal = aggregate_area(
        {"id": "front", "name": "Front"},
        [camera_observation("cam1", "bank", "banking_hall", .9),
         camera_observation("cam2", "supermarket", "checkout", .9)],
    )
    assert proposal.context["area_type"] == "unknown"
    assert proposal.bulk_reviewable is False
    assert proposal.conflicts[0].field == "area_type"
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_scene_aggregation.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement transparent evidence aggregation**

Use literal compatibility tables mapping informative area types to possible
site types. Ignore candidate values below confidence `0.60` for bulk approval.
One high-confidence disagreement creates a conflict. Reviewed context returns
unchanged and is marked authoritative until remap.

- [ ] **Step 4: Add site inference tests**

```python
def test_production_areas_infer_manufacturing_site():
    site = aggregate_site({}, [area("production_floor"), area("assembly_line"), area("loading_bay")])
    assert site["site_type"] == "manufacturing_plant"


def test_parking_alone_does_not_claim_a_site_type():
    assert aggregate_site({}, [area("parking_lot")])["site_type"] == "unknown"
```

- [ ] **Step 5: Run aggregation tests**

Run: `python -m pytest tests/test_scene_aggregation.py -q`

Expected: all pass.

---

### Task 5: Persistent Bounded Mapping Coordinator

**Files:**
- Create: `cvti/scene/coordinator.py`
- Modify: `cvti/scene/__init__.py`
- Modify: `cvti/serving/scene_map.py`
- Create: `tests/test_scene_coordinator.py`

**Interfaces:**
- Produces: `MappingJob(camera_id, area_id, source_fingerprint, state, attempts, error)`.
- Produces: `MappingProgress(total, pending, running, ready, failed, conflicts)`.
- Produces: `SceneMappingCoordinator(service, site_path, output_dir, max_workers=1)`.
- Coordinator methods: `.enqueue(camera_ids)`, `.run_next()`, `.run_until_idle()`, `.pending_camera_ids()`, `.resume()`, `.progress()`, and `.stop()`.
- Persists: `<output_dir>/context/mapping_queue.json` atomically without source URLs.

- [ ] **Step 1: Write failing queue tests**

```python
def test_one_hundred_cameras_do_not_create_parallel_mapper_calls(tmp_path):
    mapper = DeterministicMapper(delay=.01)
    coordinator = coordinator_for(tmp_path, mapper, max_workers=1)
    coordinator.enqueue([f"cam_{i}" for i in range(100)])
    coordinator.run_until_idle()
    assert mapper.max_active == 1
    assert coordinator.progress().ready == 100


def test_interrupted_queue_resumes_without_completed_jobs(tmp_path):
    first = coordinator_for(tmp_path, DeterministicMapper())
    first.enqueue(["cam1", "cam2", "cam3"])
    first.run_next()
    resumed = coordinator_for(tmp_path, DeterministicMapper())
    resumed.resume()
    assert resumed.pending_camera_ids() == ["cam2", "cam3"]
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_scene_coordinator.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement queue persistence and bounded workers**

Persist camera ID, area ID, redacted source fingerprint, state, attempts, and
error only. Deduplicate jobs by `(camera_id, source_fingerprint)`. Reset stale
`running` jobs to `pending` on resume. Default to one mapper call at a time.

- [ ] **Step 4: Recompute only affected hierarchy**

After a camera succeeds, load all camera contexts in its area, call
`aggregate_area`, persist the proposal, then recompute the site proposal. Add a
test proving `cam1` completion does not rewrite an unrelated reviewed area.

- [ ] **Step 5: Run coordinator, aggregation, and storage tests**

Run: `python -m pytest tests/test_scene_coordinator.py tests/test_scene_aggregation.py tests/test_scene_hierarchy.py -q`

Expected: all pass.

---

### Task 6: Critical-Only Monitoring and Live Context Activation

**Files:**
- Modify: `cvti/rules/customization.py`
- Modify: `cvti/serving/camera.py`
- Modify: `cvti/serving/pipeline.py`
- Modify: `cvti/serving/health_doc.py`
- Create: `tests/test_limited_monitoring.py`
- Modify: `tests/test_startup_truth.py`

**Interfaces:**
- Adds: `CustomizationEngine.evaluate(..., monitoring_scope="full")`.
- Adds: `PerCameraState.monitoring_scope: Literal["full", "critical_only"]`.
- Adds: `PerCameraState.activate_scene_context(context: dict) -> None`.
- Pipeline runs the coordinator worker and updates states without restarting unrelated cameras.

- [ ] **Step 1: Write failing critical-only rule tests**

```python
def test_unreviewed_camera_runs_critical_baseline_but_not_shoplifting():
    config = {
        "use_case_id": "limited-test",
        "rules": [{
            "name": "shoplifting", "priority": "high",
            "trigger": {"detector": "concealment", "state": "suspected"},
        }],
    }
    baseline = {
        "use_case_id": "baseline-test",
        "rules": [{
            "name": "baseline_fire_smoke", "priority": "critical",
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
            RawEvent("fire_smoke", True, "POSSIBLE FIRE", "critical",
                     state="suspected"),
            RawEvent("concealment", True, "CONCEALMENT SUSPECTED", "high",
                     state="suspected"),
        ],
        monitoring_scope="critical_only",
    )
    assert [a.rule_name for a in alerts] == ["baseline_fire_smoke"]


def test_approval_activates_full_rules_without_rebuilding_state():
    state = object.__new__(PerCameraState)
    state.camera_id = "cam1"
    state.engine = CustomizationEngine()
    state.scene_context = None
    state.monitoring_scope = "critical_only"
    identity = id(state)
    reviewed = camera_context(
        "cam1", site_type="supermarket", area_id="sales",
        area_type="retail_floor")
    reviewed["reviewed"] = True
    state.activate_scene_context(reviewed)
    assert id(state) == identity
    assert state.monitoring_scope == "full"
    assert state.scene_context == reviewed
```

The `object.__new__` construction deliberately avoids loading tracking models;
this unit owns only the in-place activation contract. Existing camera/pipeline
tests continue to exercise normal `PerCameraState` construction.

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_limited_monitoring.py -q`

Expected: missing monitoring-scope interface.

- [ ] **Step 3: Implement rule-scope filtering**

After rule match and before candidate creation, suppress rules lacking
`critical_baseline: true` when scope is `critical_only`. Record the reason in
the existing context diagnostics as `awaiting_review`, not as a false detector
result.

- [ ] **Step 4: Replace strict all-camera blocking with limited states**

Build a state for every reachable camera. Reviewed contexts use `full`;
unreviewed/failed strict contexts use `critical_only`. Keep startup and health
wording explicit: `limited monitoring - scene review required`.

- [ ] **Step 5: Add coordinator-to-state integration**

When an approved context appears, invoke `activate_scene_context` and update
health in place. Never rebuild detector models or other camera states.

- [ ] **Step 6: Run runtime regression tests**

Run: `python -m pytest tests/test_limited_monitoring.py tests/test_startup_truth.py tests/test_scene_mapping_pipeline.py tests/test_context_compatibility.py -q`

Expected: all pass.

---

### Task 7: Authorized Hierarchy and Review Backend APIs

**Files:**
- Modify: `cvti/app/console_backend.py`
- Modify: `cvti/app/bridge.py`
- Create: `tests/test_scene_review_flow.py`
- Modify: `tests/test_scene_context_console.py`

**Interfaces:**
- Produces backend methods: `scene_review_summary`, `list_areas`, `create_area`, `assign_camera_area`, `area_context`, `approve_area_context`, `update_site_context`, `approve_site_context`, `enqueue_scene_mapping`, and `scene_mapping_progress`.
- Produces matching camelCase Qt slots with JSON serialization at the bridge boundary.

- [ ] **Step 1: Write failing authorization and shape tests**

```python
def test_scene_review_summary_groups_three_cameras_into_one_area(tmp_path):
    backend = grouped_backend(tmp_path, role="owner")
    summary = backend.scene_review_summary()
    assert summary["areas"][0]["camera_ids"] == ["north", "south", "exit"]
    assert summary["counts"]["total_cameras"] == 3


def test_operator_can_view_but_cannot_assign_or_approve(tmp_path):
    backend = grouped_backend(tmp_path, role="operator")
    assert backend.scene_review_summary()["areas"]
    with pytest.raises(PermissionDenied):
        backend.assign_camera_area("north", "production")
    with pytest.raises(PermissionDenied):
        backend.approve_area_context("production", area_payload())
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_scene_review_flow.py tests/test_scene_context_console.py -q`

Expected: missing backend methods.

- [ ] **Step 3: Implement APIs using stores/coordinator**

`scene_review_summary` must return counts, sanitized camera IDs, mapping states,
area/site proposals, conflict details, and representative-frame data URIs. It
must never return source URLs or credentials.

- [ ] **Step 4: Enforce permission and audit boundaries**

Viewing uses existing camera-view permissions. Area assignment and camera
context changes require `CONFIGURE_CAMERAS`; site approval requires
`CONFIGURE_SITE`. Audit targets are `site:scene`, `area:<id>`, and
`camera:<id>` with before/after semantic fields, not credentials.

- [ ] **Step 5: Add bridge slots and verify real backend effects**

Test the backend persistence and audit records directly; inspect bridge source
only for Qt slot availability because importing Qt is optional in CI.

- [ ] **Step 6: Run backend/bridge tests**

Run: `python -m pytest tests/test_scene_review_flow.py tests/test_scene_context_console.py tests/test_console_backend.py -q`

Expected: all pass.

---

### Task 8: Area-Aware Camera Onboarding

**Files:**
- Modify: `cvti/app/web/index.html`
- Modify: `tests/test_scene_context_web.py`
- Modify: `tests/test_scene_review_flow.py`

**Interfaces:**
- Onboarding camera payload includes `area_id` plus existing optional scene hints.
- Camera step can create/select an area without leaving the wizard.
- `wizFinish()` starts monitoring and opens Scene Review instead of silently landing on the dashboard.

- [ ] **Step 1: Write failing onboarding UI contract tests**

```python
def test_wizard_assigns_camera_to_selected_area():
    html = HTML.read_text()
    body = _function_body(html, "wizAdd")
    assert "area_id" in body
    assert "wzArea" in body


def test_finishing_setup_opens_scene_review_after_starting_monitoring():
    body = _function_body(HTML.read_text(), "wizFinish")
    assert body.index('call("startMonitoring"') < body.index("openSceneReview")
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_scene_context_web.py tests/test_scene_review_flow.py -q`

Expected: missing area selector and `openSceneReview`.

- [ ] **Step 3: Add area controls to both camera-add flows**

Use a select menu with a `Create area` command. Keep controls compact and show
the area beside each added camera. Do not expose source credentials in labels.

- [ ] **Step 4: Change finish transition**

After `markConfigured`, call `startMonitoring`, close the wizard, start the app,
then open Scene Review. Mapping failures remain in the review workspace with a
retry action.

- [ ] **Step 5: Run web syntax and onboarding tests**

Run:

```bash
python -m pytest tests/test_scene_context_web.py tests/test_scene_review_flow.py -q
node -e "const fs=require('fs');const s=fs.readFileSync('cvti/app/web/index.html','utf8');[...s.matchAll(/<script[^>]*>([\\s\\S]*?)<\\/script>/g)].forEach(m=>new Function(m[1]));"
```

Expected: tests pass and Node exits `0`.

---

### Task 9: Grouped Scene Review Workspace

**Files:**
- Modify: `cvti/app/web/index.html`
- Modify: `tests/test_scene_context_web.py`
- Modify: `tests/test_scene_review_flow.py`

**Interfaces:**
- Produces UI functions: `openSceneReview`, `pollSceneReview`, `renderSceneReview`, `confirmArea`, `editAndConfirmArea`, `rejectAndRemapCameras`, and `keepAreaPaused`.
- Consumes: `sceneReviewSummary`, current camera-level scene APIs, area/site approval APIs, and mapping progress.

- [ ] **Step 1: Write failing grouped-review UI tests**

```python
def test_scene_review_is_one_workspace_not_one_modal_per_camera():
    html = HTML.read_text()
    assert 'id="sceneReview"' in html
    body = _function_body(html, "renderSceneReview")
    assert "summary.areas" in body
    assert "camera_ids" in body


def test_review_exposes_all_required_decisions():
    html = HTML.read_text()
    for label in ("Confirm area", "Edit and confirm", "Reject and remap", "Keep paused"):
        assert label in html
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_scene_context_web.py tests/test_scene_review_flow.py -q`

Expected: missing workspace/functions.

- [ ] **Step 3: Build the responsive review workspace**

Use an unframed full-screen overlay with an area sidebar, progress header, and
camera thumbnail grid. Stable thumbnail dimensions and scrollable camera lists
must prevent layout shifts with 100 cameras. Do not nest cards inside cards.

- [ ] **Step 4: Implement decisions and conflict behavior**

Disable bulk confirm when `bulk_reviewable` is false. Editing site/area fields
then confirming writes reviewed context. Reject/remap accepts selected camera
IDs. Keep paused closes the workspace but leaves `critical_only` visible on the
Live Wall.

- [ ] **Step 5: Keep existing Rules review as the maintenance path**

The current per-camera Scene context panel stays available after onboarding.
Link it from each camera thumbnail for advanced camera-specific corrections and
zone acceptance.

- [ ] **Step 6: Verify web behavior and syntax**

Run the tests and Node command from Task 8. If Playwright is available, capture
desktop `1440x900` and mobile `390x844` screenshots with a 100-camera fixture;
otherwise record that visual automation was unavailable and manually inspect
both viewport sizes in the Qt app.

---

### Task 10: End-to-End Scale, Migration, Documentation, and Final Commit

**Files:**
- Create/modify: `tests/test_scene_review_flow.py`
- Modify: `tests/test_scene_mapping_pipeline.py`
- Modify: `docs/AGENT_MAPPER_OPERATIONS.md`
- Modify: `docs/PROJECT_CONTEXT.md`
- Modify: `plan.md`
- Include: `docs/superpowers/specs/2026-09-01-hierarchical-scene-mapping-design.md`
- Include: `docs/superpowers/plans/2026-09-01-hierarchical-scene-mapping.md`

**Interfaces:**
- Verifies the complete camera -> mapper -> area/site proposal -> review -> full monitoring flow.
- Produces one final implementation commit only after every check passes.

- [ ] **Step 1: Add the 100-camera integration test**

```python
def test_one_hundred_camera_site_maps_by_area_and_activates_progressively(tmp_path):
    site = site_with_areas(area_count=20, cameras_per_area=5)
    site_path = tmp_path / "site.json"
    site_path.write_text(json.dumps(site))
    mapper = DeterministicMapper()
    service = FullAgentMapperService(tmp_path / "out", mapper)
    coordinator = SceneMappingCoordinator(
        service, site_path, tmp_path / "out", max_workers=1)
    coordinator.enqueue([camera["id"] for camera in site["cameras"]])
    coordinator.run_until_idle()
    progress = coordinator.progress()
    assert progress.total == 100
    assert progress.ready == 100
    assert progress.failed == 0
    assert mapper.max_active == 1
```

Progressive activation after area approval is tested with real camera states in
Task 6; this scale test owns queue/aggregation scale and does not duplicate that
runtime contract.

- [ ] **Step 2: Add legacy migration E2E coverage**

Open a copy of an existing configured site with reviewed camera artifacts,
assert no file mutation, no mapper call for reviewed cameras, unchanged legacy
rule behavior, and a derived review summary with implicit areas.

- [ ] **Step 3: Update operator and project documentation**

Document area creation, grouped review, limited monitoring, queue progress,
artifact locations, conflict resolution, remapping, and commands for a small
local test plus a synthetic 100-camera test. State that site/area inference is
a reviewed proposal, not guaranteed truth.

- [ ] **Step 4: Run focused hierarchy suite**

Run:

```bash
python -m pytest \
  tests/test_scene_hierarchy.py \
  tests/test_scene_aggregation.py \
  tests/test_scene_coordinator.py \
  tests/test_limited_monitoring.py \
  tests/test_scene_review_flow.py \
  tests/test_scene_context_store.py \
  tests/test_scene_mapping_preflight.py \
  tests/test_scene_mapping_pipeline.py \
  tests/test_scene_context_console.py \
  tests/test_scene_context_web.py -q
```

Expected: all pass.

- [ ] **Step 5: Run static checks**

Run:

```bash
git diff --check
node -e "const fs=require('fs');const s=fs.readFileSync('cvti/app/web/index.html','utf8');[...s.matchAll(/<script[^>]*>([\\s\\S]*?)<\\/script>/g)].forEach(m=>new Function(m[1]));"
```

Expected: both exit `0`.

- [ ] **Step 6: Run the full suite outside restricted socket sandboxes**

Run: `MPLCONFIGDIR=/private/tmp python -m pytest -q`

Expected: all tests pass; baseline before implementation was `841 passed, 1 skipped`.

- [ ] **Step 7: Run manual local-Ollama acceptance**

Use a real manufacturing/warehouse/retail multi-camera fixture. Confirm every
camera has an independent source frame, area/site proposals use evidence from
the expected cameras, conflicts are visible, and approval activates full rules.
Do not substitute `mock` for inference-quality acceptance.

- [ ] **Step 8: Create the single final commit**

```bash
git add schemas prompts cvti tests docs plan.md
git commit -m "feat: add hierarchical scene mapping and review"
```

- [ ] **Step 9: Push the completed branch and open the PR**

```bash
git push -u ayo feat/hierarchical-scene-mapping
gh pr create --repo Ayo-Cyber/cv-threat-intelligence \
  --base main \
  --head feat/hierarchical-scene-mapping \
  --title "Add hierarchical scene mapping and review"
```

The PR body must include full-suite counts, local-Ollama evidence, migration
behavior, UI screenshots or the documented visual-test limitation, and any
remaining real-footage acceptance gaps.
