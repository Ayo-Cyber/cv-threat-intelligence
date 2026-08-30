# Full Agent Mapper Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Argus's lightweight background mapper with a schema-valid, cached, reviewable Agent Mapper that grounds downstream rules and suppresses contextually impossible built-in alerts before TrueSight verification.

**Architecture:** Extend `AgentMapper` to return the selected frame and raw response once, persist that result through a site-scoped `SceneContextStore`, and resolve every camera's context during serving preflight. Apply deterministic rule compatibility in the customization engine, expose mapping state through health, and let authorized operators review or correct context through the current web console.

**Tech Stack:** Python 3.9, dataclasses, OpenCV, pathlib/JSON, pytest/unittest, Qt WebChannel, vanilla HTML/CSS/JavaScript, local Ollama OpenAI-compatible API.

**Spec:** `docs/superpowers/specs/2026-08-30-full-agent-mapper-integration-design.md`

## Global Constraints

- Production base is `ayo/main@b32dbec`.
- Use the canonical contract in `schemas/scene_context.schema.json`; do not create a second scene vocabulary.
- Context precedence is manual config, reviewed cache, policy-allowed unreviewed cache, fresh mapping, visible failure.
- Supported policies are exactly `require_reviewed`, `auto`, and `manual`; an absent policy means `auto` for legacy sites.
- Critical baseline rules bypass contextual restrictions.
- Plain-English customer rules default to explicit override semantics.
- Mapper-suggested zones remain inactive until an authorized operator accepts them.
- Context artifacts live under `<output_dir>/context/<safe_camera_id>/`, never repository-relative CWD state.
- RTSP credentials must not appear in fingerprints, logs, status JSON, or UI.
- Existing site files, static scene descriptions, and rules without context metadata remain compatible.
- Every production behavior change follows red-green-refactor; run the focused failing test before implementation.

---

## File Structure

### New files

- `cvti/scene/context_store.py`: canonical validation, environment normalization, safe paths, source fingerprints, atomic persistence, lifecycle status, context precedence, approval.
- `cvti/rules/context_compatibility.py`: pure allow/suppress/override decisions for one rule and one scene context.
- `tests/test_scene_context_store.py`: store, fingerprint, migration, precedence, and approval behavior.
- `tests/test_context_compatibility.py`: compatibility policy behavior and customization-engine integration.
- `tests/test_scene_mapping_preflight.py`: mapper service and serving preflight behavior.
- `docs/AGENT_MAPPER_OPERATIONS.md`: operator-facing artifact, policy, recovery, and local Ollama test commands.

### Modified files

- `cvti/scene/agent_mapper.py`: return a `MappingResult` containing context, selected frame, and raw response without resampling.
- `cvti/serving/scene_map.py`: become the production adapter over `AgentMapper` and `SceneContextStore`; delete duplicate prompt/parser/background mutation.
- `cvti/serving/camera.py`: accept pre-resolved canonical context and retain structured compatibility diagnostics.
- `cvti/rules/customization.py`: apply compatibility after a rule matches and before producing a candidate.
- `cvti/serving/pipeline.py`: perform per-camera mapping preflight before model/state startup, filter blocked cameras, and publish mapping health.
- `cvti/serving/health_doc.py`: degrade health for failed, stale, or review-blocked camera mapping.
- `cvti/app/console_backend.py`: read the site-scoped store and implement edit, approve, remap-request, and suggested-zone acceptance methods.
- `cvti/app/bridge.py`: expose mapper review methods to WebChannel.
- `cvti/app/web/index.html`: add scene status, representative frame, editing, approval, remap, and suggested-zone actions to Rules.
- `cvti/serving/custom_rules.py` and `cvti/serving/watch_runner.py`: consume resolved in-memory/site-scoped context rather than CWD-relative legacy files where applicable.
- `configs/baseline_critical_v1.json` and retail rule configs: mark critical baselines and add narrow shoplifting/concealment context requirements.
- Existing tests: update obsolete lightweight-mapper assumptions and add source-contract coverage.

---

### Task 1: Artifact-Bearing Agent Mapper Result

**Files:**
- Modify: `cvti/scene/agent_mapper.py`
- Create: `tests/test_agent_mapper_result.py`
- Modify: `tests/test_scene_map.py`

**Interfaces:**
- Produces: `MappingResult(context: dict[str, Any], selected_frame: Any, raw_response: str)`.
- Produces: `AgentMapper.map_result(source_raw: str, camera_id: str = "", sample_count: int = 3, source_frame_path: str = "") -> MappingResult`.
- Preserves: `AgentMapper.map(source_raw: str, camera_id: str = "", sample_count: int = 3) -> dict[str, Any]` as a compatibility wrapper returning `.context`.

- [ ] **Step 1: Write the failing result-contract tests**

```python
from unittest.mock import patch

from cvti.scene.agent_mapper import AgentMapper, MappingResult, SampledFrame


def test_map_result_returns_context_selected_frame_and_raw_response():
    frame = make_frame(width=320, height=180)
    sample = SampledFrame(frame, timestamp_seconds=2.0, score=1.0)
    mapper = AgentMapper(provider="mock")
    with patch("cvti.scene.agent_mapper.capture_sample_frames", return_value=[sample]) as capture:
        result = mapper.map_result("clip.mp4", "cam_1", source_frame_path="site/context/cam_1/source_frame.jpg")
    assert isinstance(result, MappingResult)
    assert result.selected_frame is frame
    assert result.raw_response.startswith("{")
    assert result.context["camera_id"] == "cam_1"
    assert result.context["source_frame_path"] == "site/context/cam_1/source_frame.jpg"
    capture.assert_called_once()


def test_map_compatibility_wrapper_returns_only_context():
    mapper = AgentMapper(provider="mock")
    with patch.object(mapper, "map_result") as mapped:
        mapped.return_value = MappingResult({"camera_id": "cam_1"}, object(), "{}")
        assert mapper.map("clip.mp4", "cam_1") == {"camera_id": "cam_1"}
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_agent_mapper_result.py -q`

Expected: collection fails because `MappingResult` and `map_result` do not exist.

- [ ] **Step 3: Add the minimal result API and keep one sampling/provider path**

```python
@dataclass
class MappingResult:
    context: dict[str, Any]
    selected_frame: Any
    raw_response: str


def map_result(self, source_raw: str, camera_id: str = "", sample_count: int = 3,
               source_frame_path: str = "") -> MappingResult:
    # Existing normalize, sample, representative selection, prompt, provider,
    # and parse logic moves here unchanged. Use source_frame_path when supplied.
    return MappingResult(context=context, selected_frame=selected.image, raw_response=raw)


def map(self, source_raw: str, camera_id: str = "", sample_count: int = 3) -> dict[str, Any]:
    return self.map_result(source_raw, camera_id, sample_count).context
```

- [ ] **Step 4: Run mapper tests and confirm GREEN**

Run: `./.venv/bin/python -m pytest tests/test_agent_mapper_result.py tests/test_scene_map.py -q`

Expected: all tests pass after obsolete `_parse`/`_sample_frame` tests are replaced with compatibility-wrapper tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add cvti/scene/agent_mapper.py tests/test_agent_mapper_result.py tests/test_scene_map.py
git commit -m "refactor: expose complete agent mapper result"
```

---

### Task 2: Canonical Scene Context Store

**Files:**
- Create: `cvti/scene/context_store.py`
- Create: `tests/test_scene_context_store.py`
- Modify: `cvti/scene/__init__.py`

**Interfaces:**
- Consumes: canonical environment/source/zone constants from `cvti.scene.agent_mapper`.
- Produces: `MappingStatus`, `ContextResolution`, `SceneContextStore`.
- Produces: `normalize_environment_type(value: str) -> str`.
- Produces: `validate_scene_context(context: dict) -> dict`.
- Produces: `source_fingerprint(source: int | str) -> str`.
- Produces: `SceneContextStore.load_status() -> MappingStatus`.
- Produces: `SceneContextStore.resolve(source, policy, manual_context=None, legacy_context_path=None) -> ContextResolution`.
- Produces: `SceneContextStore.save_mapping(result, source, dump_raw_response=False) -> ContextResolution`.
- Produces: `SceneContextStore.save_unreviewed(context, source, provenance="manual_edit") -> ContextResolution`.
- Produces: `SceneContextStore.approve(context, reviewer, source) -> ContextResolution`.
- Produces: `SceneContextStore.mark_pending(source)`, `.mark_failed(source, error)`, and `.mark_stale(source)`.

- [ ] **Step 1: Write failing tests for safe paths, normalization, and validation**

```python
def test_store_rejects_camera_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="camera_id"):
        SceneContextStore(tmp_path, "../outside")


@pytest.mark.parametrize(("raw", "canonical"), [
    ("retail", "retail_shop"), ("parking", "parking_lot"),
    ("warehouse", "warehouse_floor"), ("street", "estate_street"),
])
def test_environment_aliases_normalize(raw, canonical):
    assert normalize_environment_type(raw) == canonical


def test_schema_validation_rejects_missing_required_field():
    context = canonical_context()
    context.pop("expected_actors")
    with pytest.raises(ValueError, match="expected_actors"):
        validate_scene_context(context)
```

- [ ] **Step 2: Run the first store tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_scene_context_store.py -q`

Expected: import fails because `cvti.scene.context_store` does not exist.

- [ ] **Step 3: Implement canonical validation and atomic JSON helpers**

```python
ENVIRONMENT_ALIASES = {
    "retail": "retail_shop", "shop": "retail_shop",
    "parking": "parking_lot", "warehouse": "warehouse_floor",
    "street": "estate_street", "entrance": "estate_gate",
    "office": "office_floor", "home": "residential_interior",
    "other": "unknown",
}


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
```

Validation must explicitly check exact required keys, canonical enums, unique non-empty actors, four-integer non-negative zone boxes, confidence in `[0, 1]`, and parseable ISO timestamp; reject unknown extra keys except optional `notes`.

- [ ] **Step 4: Write failing fingerprint and persistence tests**

```python
def test_rtsp_fingerprint_never_contains_credentials():
    value = source_fingerprint("rtsp://alice:secret@10.0.0.8:554/live")
    assert value.startswith("sha256:")
    assert "alice" not in value and "secret" not in value


def test_save_mapping_writes_site_scoped_artifacts_atomically(tmp_path):
    store = SceneContextStore(tmp_path, "cam_1")
    store.save_mapping(mapping_result(), source="clip.mp4")
    assert json.loads(store.context_path.read_text())["camera_id"] == "cam_1"
    assert json.loads(store.status_path.read_text())["status"] == "ready_unreviewed"
    assert store.frame_path.read_bytes().startswith(b"\xff\xd8")
    assert not list(store.directory.glob("*.tmp"))
```

- [ ] **Step 5: Implement fingerprints, status dataclasses, and persistence**

```python
@dataclass(frozen=True)
class MappingStatus:
    status: str
    source_fingerprint: str = ""
    mapped_at: str = ""
    reviewed_at: str = ""
    reviewed_by: str = ""
    error: str = ""


@dataclass(frozen=True)
class ContextResolution:
    context: dict | None
    status: MappingStatus
    provenance: str
    usable: bool
```

For local files fingerprint the resolved path, size, and `st_mtime_ns`. For RTSP parse with `urllib.parse.urlsplit`, remove userinfo, and hash only scheme/host/port/path/query. Save JPEG with `cv2.imencode` followed by an atomic byte replace. Raw responses are persisted only when `dump_raw_response=True`.

- [ ] **Step 6: Write failing precedence, stale, approval, and legacy-import tests**

```python
def test_manual_context_outranks_reviewed_cache(tmp_path):
    store = SceneContextStore(tmp_path, "cam_1")
    source = source_file(tmp_path)
    store.approve(canonical_context("parking_lot"), "owner", source)
    result = store.resolve(source, "require_reviewed",
                           manual_context=canonical_context("retail_shop"))
    assert result.provenance == "manual"
    assert result.context["environment_type"] == "retail_shop"


def test_unreviewed_cache_is_usable_only_under_auto(tmp_path):
    source = source_file(tmp_path)
    store = SceneContextStore(tmp_path, "cam_1")
    store.save_mapping(mapping_result(), source)
    assert store.resolve(source, "auto").usable is True
    assert store.resolve(source, "require_reviewed").usable is False


def test_source_change_marks_matching_cache_stale(tmp_path):
    source = source_file(tmp_path, b"first")
    store = SceneContextStore(tmp_path, "cam_1")
    store.save_mapping(mapping_result(), source)
    source.write_bytes(b"changed source identity")
    result = store.resolve(source, "auto")
    assert result.usable is False
    assert result.status.status == "stale"


def test_approve_records_reviewer_and_corrected_context(tmp_path):
    store = SceneContextStore(tmp_path, "cam_1")
    result = store.approve(canonical_context("parking_lot"), "ayo", source_file(tmp_path))
    assert result.status.reviewed_by == "ayo"
    assert result.status.reviewed_at


def test_legacy_two_field_context_imports_as_unreviewed_canonical(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text('{"environment_type":"retail","scene_description":"A shop."}')
    store = SceneContextStore(tmp_path / "site", "cam_1")
    result = store.resolve(source_file(tmp_path), "auto", legacy_context_path=legacy)
    assert result.context["environment_type"] == "retail_shop"
    assert result.status.status == "ready_unreviewed"
```

Each test constructs exact canonical input and asserts `ContextResolution.provenance`, `.usable`, and `status.status`; legacy completion must set unknown source facts from caller-owned arguments and never claim review.

- [ ] **Step 7: Implement resolution and approval**

`resolve(source, policy, manual_context=None, legacy_context_path=None)` applies the documented precedence. `approve(context, reviewer, source)` validates and atomically writes corrected context, preserves `mapped_at`, sets `ready_reviewed`, and records reviewer/time. A stale or failed context may remain on disk for diagnosis but returns `usable=False`.

- [ ] **Step 8: Run store tests and confirm GREEN**

Run: `./.venv/bin/python -m pytest tests/test_scene_context_store.py -q`

Expected: all store tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add cvti/scene/context_store.py cvti/scene/__init__.py tests/test_scene_context_store.py
git commit -m "feat: add site-scoped scene context store"
```

---

### Task 3: Deterministic Rule Compatibility

**Files:**
- Create: `cvti/rules/context_compatibility.py`
- Create: `tests/test_context_compatibility.py`
- Modify: `cvti/rules/customization.py`
- Modify: `configs/baseline_critical_v1.json`
- Modify: retail concealment/shoplifting rule config files identified by `rg -l 'shoplift|concealment' configs`

**Interfaces:**
- Produces: `CompatibilityDecision(allowed: bool, mode: str, reason: str)`.
- Produces: `evaluate_context_compatibility(rule: dict, context: dict | None, *, baseline: bool = False, active_zone_roles: set[str] | None = None) -> CompatibilityDecision`.
- Produces: `CustomizationEngine.context_decisions: list[dict]`, reset on each `evaluate()` call.

- [ ] **Step 1: Write failing pure compatibility tests**

```python
def test_shoplifting_is_suppressed_in_parking_lot():
    rule = {"name": "shoplifting", "context_requirements": {
        "environment_types": ["retail_shop", "mall_corridor"],
        "zone_roles_any": ["merchandise", "checkout"], "mode": "enforce"}}
    decision = evaluate_context_compatibility(rule, scene("parking_lot"))
    assert decision.allowed is False
    assert decision.mode == "context_incompatible"


def test_matching_environment_or_zone_allows_rule():
    assert evaluate_context_compatibility(shoplifting_rule(), scene("retail_shop")).allowed
    parking = scene("parking_lot")
    assert evaluate_context_compatibility(
        shoplifting_rule(), parking, active_zone_roles={"checkout"}).allowed


def test_missing_requirements_preserve_legacy_allow():
    assert evaluate_context_compatibility({"name": "legacy"}, None).mode == "allowed"


def test_override_allows_and_records_override():
    decision = evaluate_context_compatibility(
        shoplifting_rule(mode="override"), scene("parking_lot"))
    assert decision.allowed and decision.mode == "explicit_override"


def test_unknown_context_fails_enforced_requirement():
    assert not evaluate_context_compatibility(
        shoplifting_rule(), scene("unknown")).allowed


def test_critical_baseline_bypasses_requirements():
    decision = evaluate_context_compatibility(
        shoplifting_rule(), scene("parking_lot"), baseline=True)
    assert decision.allowed and decision.mode == "critical_baseline"
```

- [ ] **Step 2: Run compatibility tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_context_compatibility.py -q`

Expected: import fails because the module does not exist.

- [ ] **Step 3: Implement the pure decision function**

```python
@dataclass(frozen=True)
class CompatibilityDecision:
    allowed: bool
    mode: str
    reason: str
```

No requirements returns `allowed`; `baseline=True` returns `critical_baseline`; `mode=override` returns `explicit_override`; enforced requirements allow when any configured environment matches or any role in the separately supplied `active_zone_roles` matches. Mapper-suggested `context["zones"]` must never satisfy a requirement because suggestions are inactive. Empty/unknown context never satisfies enforce.

- [ ] **Step 4: Write failing customization-engine integration tests**

```python
def test_engine_does_not_emit_context_incompatible_candidate(tmp_path):
    engine = engine_with_rule(tmp_path, shoplifting_rule())
    alerts = engine.evaluate([concealment_event()], scene_context=scene("parking_lot"))
    assert alerts == []
    assert engine.context_decisions[0]["decision"] == "context_incompatible"


def test_plain_english_gate_question_is_explicit_override(tmp_path):
    rule = base_rule("customer_question", "presence")
    rule["gate_question"] = "Is someone taking stock from a delivery vehicle?"
    engine = engine_with_rules(tmp_path, [rule])
    alerts = engine.evaluate([presence_event()], scene_context=scene("parking_lot"))
    assert alerts[0].rule_name == "customer_question"
    assert engine.context_decisions[0]["decision"] == "explicit_override"


def test_baseline_fire_still_emits_with_parking_context(tmp_path):
    engine = engine_with_baseline(tmp_path, [baseline_fire_rule()])
    alerts = engine.evaluate([fire_event()], scene_context=scene("parking_lot"))
    assert alerts[0].rule_name == "baseline_fire_smoke"
    assert engine.context_decisions[0]["decision"] == "critical_baseline"
```

- [ ] **Step 5: Integrate compatibility after normal trigger/time/context-filter matching**

For each matched rule, call the compatibility function before appending `CandidateAlert`. Extend `CustomizationEngine.evaluate()` and `top_alert()` with optional `active_zone_roles: set[str] | None = None`, passed independently of mapper suggestions. Treat rules with `gate_question` and no explicit requirements as customer override. Record structured fields `rule`, `environment`, `decision`, and `reason` in `context_decisions`; log suppressions at debug level to avoid alert spam.

- [ ] **Step 6: Add narrow built-in metadata**

Add `context_requirements` only to built-in shoplifting/concealment rules:

```json
"context_requirements": {
  "environment_types": ["retail_shop", "mall_corridor"],
  "zone_roles_any": ["merchandise", "checkout"],
  "mode": "enforce"
}
```

Mark baseline entries with `"critical_baseline": true`. Do not restrict violence, weapons, fire, person-down, tamper, loitering, running, or perimeter rules.

- [ ] **Step 7: Run focused rule tests and confirm GREEN**

Run: `./.venv/bin/python -m pytest tests/test_context_compatibility.py tests/test_compound_recipes.py tests/test_zone_customization.py -q`

Expected: all tests pass and existing rules without metadata are unchanged.

- [ ] **Step 8: Commit Task 3**

```bash
git add cvti/rules/context_compatibility.py cvti/rules/customization.py configs tests/test_context_compatibility.py
git commit -m "feat: suppress context-incompatible built-in alerts"
```

---

### Task 4: Full Mapper Serving Adapter and Camera Preflight

**Files:**
- Replace: `cvti/serving/scene_map.py`
- Create: `tests/test_scene_mapping_preflight.py`
- Modify: `cvti/serving/camera.py`
- Modify: `tests/test_audit_hardening.py`

**Interfaces:**
- Consumes: `AgentMapper.map_result()` and `SceneContextStore`.
- Produces: `CameraMappingResult(camera_id, resolution, mapped: bool)`.
- Produces: `SceneMappingPreflight(contexts: dict[str, dict], statuses: dict[str, dict], blocked_camera_ids: set[str])`.
- Produces: `FullAgentMapperService.prepare(cameras: list[dict], policy: str) -> SceneMappingPreflight`.
- Changes: `build_camera_states(site_config, *, pose_model=None, weapon_model=None, video_action_model=None, baseline_config=None, scene_contexts: dict[str, dict] | None = None)`.

- [ ] **Step 1: Write failing service tests for cache, fresh map, and failures**

```python
def test_reviewed_cache_avoids_mapper_call(tmp_path):
    mapper = RecordingMapper()
    service = prepared_service(tmp_path, mapper, reviewed_cache=True)
    result = service.prepare([camera()], "require_reviewed")
    assert result.contexts["cam_1"]["camera_id"] == "cam_1"
    assert mapper.calls == []


def test_auto_policy_maps_missing_context_and_returns_it_usable(tmp_path):
    mapper = RecordingMapper()
    result = prepared_service(tmp_path, mapper).prepare([camera()], "auto")
    assert result.blocked_camera_ids == set()
    assert mapper.calls == [("clip.mp4", "cam_1", 3)]


def test_require_reviewed_maps_but_blocks_unreviewed_camera(tmp_path):
    result = prepared_service(tmp_path, RecordingMapper()).prepare(
        [camera()], "require_reviewed")
    assert result.blocked_camera_ids == {"cam_1"}
    assert result.statuses["cam_1"]["status"] == "ready_unreviewed"


def test_manual_policy_blocks_camera_without_explicit_context(tmp_path):
    mapper = RecordingMapper()
    result = prepared_service(tmp_path, mapper).prepare([camera()], "manual")
    assert result.blocked_camera_ids == {"cam_1"}
    assert mapper.calls == []


def test_mapper_failure_marks_camera_failed_without_generic_context(tmp_path):
    result = prepared_service(tmp_path, FailingMapper("ollama unavailable")).prepare(
        [camera()], "auto")
    assert "cam_1" not in result.contexts
    assert result.statuses["cam_1"]["status"] == "failed"


def test_static_camera_context_is_reviewed_and_never_calls_mapper(tmp_path):
    mapper = RecordingMapper()
    cam = camera(scene_description="A monitored car park.", environment_type="parking_lot")
    result = prepared_service(tmp_path, mapper).prepare([cam], "require_reviewed")
    assert result.statuses["cam_1"]["status"] == "ready_reviewed"
    assert mapper.calls == []
```

Use a fake mapper whose `map_result` records calls and returns a real `MappingResult`. Assert status/provenance and exact blocked IDs, not implementation internals.

- [ ] **Step 2: Run preflight tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_scene_mapping_preflight.py -q`

Expected: imports fail because `FullAgentMapperService` and preflight dataclasses do not exist.

- [ ] **Step 3: Replace duplicate mapper logic with the serving adapter**

```python
class FullAgentMapperService:
    def __init__(self, output_dir: Path, mapper: AgentMapper,
                 dump_raw_response: bool = False,
                 legacy_root: Path = Path("runs/context")) -> None:
        self.output_dir = output_dir
        self.mapper = mapper
        self.dump_raw_response = dump_raw_response
        self.legacy_root = legacy_root

    def prepare(self, cameras: list[dict], policy: str = "auto") -> SceneMappingPreflight:
        """Resolve or map each camera and return contexts, statuses and blocked IDs."""
```

The service processes cameras sequentially, resolves manual/cache first, marks pending before a VLM call, calls `map_result` once, saves exact artifacts, and never mutates `PerCameraState` from a daemon thread. Keep no `SCENE_PROMPT`, `_parse`, `_sample_frame`, or `map_cameras_async` compatibility shim.

- [ ] **Step 4: Write failing camera-state injection test**

```python
def test_build_camera_states_uses_pre_resolved_canonical_context():
    states = build_camera_states(site(), scene_contexts={"cam_1": canonical_context()})
    assert states["cam_1"]["state"].scene_context["expected_actors"] == ["staff"]
```

- [ ] **Step 5: Add `scene_contexts` injection and context diagnostics to camera state**

Use pre-resolved context by camera ID before static two-field fallback. Add `active_zone_roles: set[str]` to `PerCameraState`, populated only from accepted site/zone configuration, and pass it into `engine.evaluate`; mapper-suggested zones do not enter this set. After each engine evaluation, copy `engine.context_decisions` into `PerCameraState.context_decisions` so health can count suppressions without changing queued-alert payloads.

- [ ] **Step 6: Run focused service/camera tests and confirm GREEN**

Run: `./.venv/bin/python -m pytest tests/test_scene_mapping_preflight.py tests/test_scene_map.py tests/test_audit_hardening.py tests/test_compound_recipes.py -q`

Expected: all tests pass; source-contract tests no longer expect CWD-relative background writes.

- [ ] **Step 7: Commit Task 4**

```bash
git add cvti/serving/scene_map.py cvti/serving/camera.py tests/test_scene_mapping_preflight.py tests/test_scene_map.py tests/test_audit_hardening.py
git commit -m "feat: prepare full scene context before camera startup"
```

---

### Task 5: Pipeline Lifecycle, Health, and Downstream Context

**Files:**
- Modify: `cvti/serving/pipeline.py`
- Modify: `cvti/serving/health_doc.py`
- Modify: `cvti/serving/custom_rules.py`
- Modify: `cvti/serving/watch_runner.py`
- Create: `tests/test_scene_mapping_pipeline.py`
- Modify: `tests/test_health_endpoint.py`
- Modify: `tests/test_health.py`
- Modify: `tests/test_custom_rules_multi.py`
- Modify: `tests/test_watches.py`

**Interfaces:**
- Consumes: `FullAgentMapperService.prepare()`.
- Changes: `run_site()` gains keyword arguments `mapper_provider: str = ""`, `mapper_model: str = ""`, and `mapper_base_url: str = ""`; empty values inherit the gate settings.
- Changes: pipeline CLI gains matching `--mapper-provider`, `--mapper-model`, and `--mapper-base-url` arguments and forwards them to `run_site()`.
- Adds: `scene_mapping` array and `context_suppressions` count to health output.

- [ ] **Step 1: Write failing pipeline source-contract and behavior tests**

```python
def test_run_site_prepares_context_before_building_camera_states(monkeypatch, site_file):
    capture = install_pipeline_fakes(monkeypatch, ready_preflight("cam_1"))
    run_site(str(site_file), gate_provider="ollama", gate_model="gemma3:4b", seconds=0)
    assert capture.calls.index("prepare") < capture.calls.index("build_camera_states")
    assert capture.calls.index("build_camera_states") < capture.calls.index("pipe.start")


def test_blocked_camera_is_excluded_but_ready_camera_starts(monkeypatch, two_camera_site):
    capture = install_pipeline_fakes(
        monkeypatch, mixed_preflight(ready="cam_1", blocked="cam_2"))
    run_site(str(two_camera_site), gate_provider="ollama", seconds=0)
    assert capture.active_camera_ids == ["cam_1"]


def test_mapper_defaults_to_local_gate_model(monkeypatch, site_file):
    capture = install_pipeline_fakes(monkeypatch, ready_preflight("cam_1"))
    run_site(str(site_file), gate_provider="ollama", gate_model="gemma3:4b", seconds=0)
    assert capture.mapper_settings == (
        "ollama", "gemma3:4b", "http://localhost:11434/v1")
def test_post_start_background_mapper_is_removed():
    assert "map_cameras_async" not in inspect.getsource(run_site)
```

Patch expensive model, queue, publisher, and pipeline constructors with deterministic fakes. Record call order and assert `prepare < build_camera_states < pipe.start`.

- [ ] **Step 2: Run pipeline tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_scene_mapping_pipeline.py -q`

Expected: call-order and API tests fail because preflight is absent.

- [ ] **Step 3: Integrate preflight before model and state construction**

After loading the site and before pose/weapon/video models, validate policy, build `FullAgentMapperService`, and prepare contexts. Filter only blocked cameras into a copied site dict; never mutate the loaded caller object in place. If every camera is blocked, write health with explicit mapping reasons and return without loading YOLO. Pass resolved contexts into `build_camera_states` and remove the post-`pipe.start()` mapper block.

- [ ] **Step 4: Write failing health tests**

```python
def test_failed_scene_mapping_degrades_health_with_camera_reason():
    doc = health_doc(scene_mapping=[
        mapping_health("cam_1", "failed", error="ollama unavailable")])
    assert doc["status"] == "degraded"
    assert any("cam_1" in reason and "mapping failed" in reason
               for reason in doc["reasons"])


def test_review_required_mapping_degrades_health_without_claiming_camera_offline():
    doc = health_doc(scene_mapping=[
        mapping_health("cam_1", "ready_unreviewed", review_required=True)])
    assert doc["status"] == "degraded"
    assert not any("offline" in reason for reason in doc["reasons"])


def test_context_suppression_count_is_reported():
    assert pipeline_health(states_with_suppressions(2))["engine"][
        "context_suppressions"] == 2
```

- [ ] **Step 5: Extend health assembly**

Add optional `scene_mapping: list | None = None` to `build_health_doc`. `failed`, `stale`, and unapproved `require_reviewed` states add degraded reasons. Include status/provenance/environment/error per camera; never include source URL or frame content. In pipeline health, aggregate `len(state.context_decisions where decision == context_incompatible)` into engine diagnostics.

- [ ] **Step 6: Write failing downstream-consumer tests**

Verify custom rules and watches receive each state's resolved canonical context and never open `runs/context/<camera>` themselves. For expected actors, assert prompt text says “may include” rather than asserting identity.

- [ ] **Step 7: Update downstream consumers to use injected state context**

Pass a context provider callback from pipeline to scanner/watch components where they currently derive context independently. Remove any repository-relative scene-context reads found by `rg 'runs/context' cvti/serving`.

- [ ] **Step 8: Run focused serving and health tests and confirm GREEN**

Run: `./.venv/bin/python -m pytest tests/test_scene_mapping_pipeline.py tests/test_health.py tests/test_health_endpoint.py tests/test_custom_rules_multi.py tests/test_watches.py tests/test_serving.py -q`

Expected: all tests pass.

- [ ] **Step 9: Commit Task 5**

```bash
git add cvti/serving/pipeline.py cvti/serving/health_doc.py cvti/serving/custom_rules.py cvti/serving/watch_runner.py tests
git commit -m "feat: gate camera startup on resolved scene context"
```

---

### Task 6: Authorized Console Review API

**Files:**
- Modify: `cvti/app/console_backend.py`
- Modify: `cvti/app/bridge.py`
- Modify: `tests/test_console_backend.py`
- Create: `tests/test_scene_context_console.py`

**Interfaces:**
- Produces backend methods:
  - `scene_context(camera_id: str) -> dict | None`
  - `update_scene_context(camera_id: str, context: dict) -> dict`
  - `approve_scene_context(camera_id: str, context: dict) -> dict`
  - `request_scene_remap(camera_id: str) -> dict`
  - `accept_suggested_zone(camera_id: str, zone_id: str, dwell_seconds: float) -> dict`
- Produces bridge slots: `updateSceneContext`, `approveSceneContext`, `requestSceneRemap`, `acceptSuggestedZone`.

- [ ] **Step 1: Write failing backend path and permission tests**

```python
def test_scene_context_reads_db_parent_context_not_cwd(tmp_path, monkeypatch):
    backend = authenticated_backend(tmp_path, role="owner")
    write_context(tmp_path / "context/cam_1", environment="parking_lot")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert backend.scene_context("cam_1")["environment_type"] == "parking_lot"


def test_operator_cannot_edit_or_approve_scene_context(tmp_path):
    backend = authenticated_backend(tmp_path, role="operator")
    with pytest.raises(PermissionDenied):
        backend.approve_scene_context("cam_1", canonical_context())


def test_installer_can_edit_and_approve_scene_context(tmp_path):
    backend = authenticated_backend(tmp_path, role="installer")
    assert backend.update_scene_context("cam_1", canonical_context())["ok"]
    assert backend.approve_scene_context("cam_1", canonical_context())["ok"]


def test_owner_approval_writes_review_identity_and_audit_entry(tmp_path):
    backend = authenticated_backend(tmp_path, role="owner", username="ayo")
    backend.approve_scene_context("cam_1", canonical_context())
    assert read_status(tmp_path / "context/cam_1")["reviewed_by"] == "ayo"
    assert backend.audit.entries(limit=1)[0].action == "config_change"


def test_remap_marks_context_stale_without_deleting_it(tmp_path):
    backend = authenticated_backend(tmp_path, role="installer")
    write_context(tmp_path / "context/cam_1")
    backend.request_scene_remap("cam_1")
    assert (tmp_path / "context/cam_1/scene_context.json").exists()
    assert read_status(tmp_path / "context/cam_1")["status"] == "stale"
```

Build backend with `db_path=<output_dir>/events.db`; authenticate real test accounts and assert server-side `CONFIGURE_CAMERAS` enforcement.

- [ ] **Step 2: Run console tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_scene_context_console.py -q`

Expected: methods/path behavior are absent.

- [ ] **Step 3: Implement store-backed read/edit/approve/remap methods**

Resolve the store root as `Path(self.db_path).parent / "context"`. Read camera source from site config for fingerprint checks. `scene_context` adds a base64 `source_frame_uri` for the local representative frame without exposing a filesystem URL. `update_scene_context` calls `save_unreviewed`; `approve_scene_context` records `current_user.username`; `request_scene_remap` marks stale and writes an audit `config_change` action. Return structured validation errors without exposing stack traces.

- [ ] **Step 4: Write failing zone-acceptance test**

```python
def test_accept_suggested_zone_converts_bbox_to_inactive_polygon_then_uses_existing_add_zone(tmp_path):
    result = backend.accept_suggested_zone("cam_1", "checkout", 5.0)
    assert result["ok"] is True
    assert result["zone"]["points"] == expected_normalized_polygon
```

The test supplies source-frame dimensions and verifies `[x, y, width, height]` becomes four normalized polygon points. It also asserts no detector/rule is activated beyond the operator-selected dwell behavior.

- [ ] **Step 5: Implement suggested-zone acceptance through existing zone APIs**

Look up the canonical context zone by ID, load representative-frame dimensions, clamp coordinates, convert to normalized polygon, and call the existing `add_zone` path so zone persistence/rule regeneration remains single-sourced. Persist the accepted mapper role beside that zone in site configuration so `PerCameraState.active_zone_roles` can use it; ignored suggestions never enter runtime roles. Require `CONFIGURE_CAMERAS`.

- [ ] **Step 6: Add WebChannel slots and bridge tests**

```python
@pyqtSlot(str, str, result=str)
def approveSceneContext(self, camera_id: str, context_json: str) -> str:
    return self._safe(lambda: self._core.approve_scene_context(
        camera_id, json.loads(context_json)))
```

Add equivalent exact slots for update, remap, and suggested-zone acceptance.

- [ ] **Step 7: Run console tests and confirm GREEN**

Run: `./.venv/bin/python -m pytest tests/test_scene_context_console.py tests/test_console_backend.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit Task 6**

```bash
git add cvti/app/console_backend.py cvti/app/bridge.py tests/test_scene_context_console.py tests/test_console_backend.py
git commit -m "feat: add authorized scene context review API"
```

---

### Task 7: Current Argus Console Mapper Review UI

**Files:**
- Modify: `cvti/app/web/index.html`
- Modify: `tests/test_console_emit.py`
- Create: `tests/test_scene_context_web.py`

**Interfaces:**
- Consumes bridge methods from Task 6.
- Adds no alternative UI framework and does not revive `cvti/app/widgets/mapper.py`.

- [ ] **Step 1: Write failing static UI contract tests**

```python
def test_rules_screen_has_mapping_status_review_and_remap_controls():
    html = Path("cvti/app/web/index.html").read_text()
    for token in ("mapping-status", "approveSceneContext", "requestSceneRemap",
                  "acceptSuggestedZone", "expected_actors"):
        assert token in html


def test_mapper_ui_does_not_auto_accept_suggested_zones():
    html = Path("cvti/app/web/index.html").read_text()
    assert "acceptSuggestedZone" not in mapper_load_function_body(html)
```

- [ ] **Step 2: Run web contract tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_scene_context_web.py -q`

Expected: required controls are absent.

- [ ] **Step 3: Build the Rules-screen context panel**

Replace the read-only scene hint with a compact unframed panel containing status/provenance, representative image, canonical environment `<select>`, scene description `<textarea>`, comma-separated expected actors, confidence, and suggested-zone rows. Use existing button styles and familiar icons where available. Keep panel headings compact; avoid nested cards.

- [ ] **Step 4: Wire edit, approve, remap, and zone actions**

`loadSceneContext(camId)` populates controls. Save calls `updateSceneContext`; Approve calls `approveSceneContext`; Remap calls `requestSceneRemap` and displays “pending on next monitoring start”; each suggestion has explicit Accept and Ignore controls, with Accept calling `acceptSuggestedZone`. Hide mutation controls when `state.auth.permissions` lacks `configure_cameras`.

- [ ] **Step 5: Add visible lifecycle and failure states**

Render `pending`, `ready_unreviewed`, `ready_reviewed`, `stale`, and `failed` in plain language. Failure text uses the sanitized backend error. `require_reviewed` cameras explain that monitoring is paused for that camera; `auto` unreviewed cameras explain that monitoring is active but review is recommended.

- [ ] **Step 6: Verify JavaScript and UI contracts**

Run: `node --check cvti/app/web/index.html`

If Node rejects HTML input, extract the existing `<script>` body using the repository's current test helper and run `node --check` on the temporary JavaScript file, matching `tests/test_console_emit.py`.

Run: `./.venv/bin/python -m pytest tests/test_scene_context_web.py tests/test_console_emit.py -q`

Expected: JavaScript syntax and static UI contracts pass.

- [ ] **Step 7: Commit Task 7**

```bash
git add cvti/app/web/index.html tests/test_scene_context_web.py tests/test_console_emit.py
git commit -m "feat: add scene context review to Argus console"
```

---

### Task 8: New-Site Policy, Migration, Operations, and End-to-End Verification

**Files:**
- Modify: setup/onboarding site creation functions located by `rg -n 'scene_context_policy|site.*config|json.dump' cvti/serving/onboarding.py cvti/app/console_backend.py`
- Modify: `configs/site_*.json` only where a shipped new-site template requires explicit policy.
- Create: `docs/AGENT_MAPPER_OPERATIONS.md`
- Modify: `docs/PROJECT_CONTEXT.md`
- Modify: `plan.md`
- Create: `tests/test_scene_context_e2e.py`

**Interfaces:**
- New wizard-created sites write `"scene_context_policy": "require_reviewed"`.
- Existing files with no field remain `auto` at runtime and are not rewritten automatically.

- [ ] **Step 1: Write failing onboarding and migration tests**

```python
def test_new_wizard_site_requires_reviewed_scene_context(tmp_path):
    site = create_wizard_site(tmp_path)
    assert json.loads(site.read_text())["scene_context_policy"] == "require_reviewed"


def test_legacy_site_without_policy_resolves_as_auto():
    assert normalize_scene_context_policy(None) == "auto"


def test_existing_static_scene_description_counts_as_reviewed(tmp_path):
    cam = camera(scene_description="A shop floor.", environment_type="retail_shop")
    result = prepared_service(tmp_path, RecordingMapper()).prepare(
        [cam], "require_reviewed")
    assert result.statuses["cam_1"]["status"] == "ready_reviewed"


def test_legacy_context_is_imported_but_never_marked_reviewed(tmp_path):
    result = import_legacy_context(tmp_path, {
        "environment_type": "retail", "scene_description": "A shop."})
    assert result.status.status == "ready_unreviewed"
    assert result.context["environment_type"] == "retail_shop"
```

- [ ] **Step 2: Run migration tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_scene_context_e2e.py -q`

Expected: new-site policy assertion fails.

- [ ] **Step 3: Add explicit policy to new-site creation only**

Set the field in wizard/new-site defaults. Centralize policy parsing in `scene_map.py` or `context_store.py` as `normalize_scene_context_policy(value)`, where empty means `auto` and invalid non-empty values raise a configuration error naming the site field.

- [ ] **Step 4: Write the operations guide with exact commands**

Document:

```bash
ollama list
curl -s http://127.0.0.1:11434/api/tags
MPLCONFIGDIR=/private/tmp OLLAMA_API_KEY=ollama ./.venv/bin/python -m cvti.serving.pipeline \
  --site-config configs/site_demo.json \
  --gate-provider ollama --gate-model gemma3:4b \
  --output-dir runs/agent_mapper_e2e --seconds 30
```

Include artifact locations, each lifecycle status, how to approve/remap in Argus, how to identify blocked cameras in `gate_health.json`, and a parking-lot/shoplifting override test. State that mock mapper/gate output is never a quality evaluation.

- [ ] **Step 5: Update project context and roadmap truthfully**

Record the final architecture, exact commit sequence, tests run, remaining held-out evaluation requirement, and deployment caveats. Mark implementation complete only after full verification; keep hallucination reduction as unproven until the held-out context-confusion test passes.

- [ ] **Step 6: Run the complete automated suite**

Run: `MPLCONFIGDIR=/private/tmp ./.venv/bin/python -m pytest -q`

Expected: at least the baseline `751 passed, 1 skipped`, plus all new tests, with zero failures.

- [ ] **Step 7: Run static and repository checks**

Run: `git diff --check`

Run: `rg -n 'SCENE_PROMPT|map_cameras_async|runs/context' cvti/serving cvti/app/console_backend.py`

Expected: no duplicate production mapper prompt/background API; any remaining `runs/context` reference is explicitly legacy-import-only and covered by a test.

- [ ] **Step 8: Run local Ollama smoke verification when Ollama is available**

Map one retail and one parking clip, inspect `scene_context.json`, `mapping_status.json`, and `source_frame.jpg`, then run a parking context against an enforced shoplifting rule. Confirm the candidate is suppressed before a gate request; add `mode=override` and confirm it reaches TrueSight with parking context. If Ollama or clips are unavailable, report this verification as not run rather than substituting mock results.

- [ ] **Step 9: Commit Task 8**

```bash
git add cvti configs docs tests plan.md
git commit -m "docs: complete agent mapper rollout and operations"
```

---

## Final Review Checklist

- [ ] Every active camera has canonical context with provenance and lifecycle status.
- [ ] No candidate verification begins with silently generic context.
- [ ] Mapper sampling, prompt, parsing, and selected-frame persistence happen once.
- [ ] Cached reviewed context causes zero mapper calls on unchanged restart.
- [ ] New `require_reviewed` cameras are blocked individually, not site-wide.
- [ ] Parking-lot shoplifting is suppressed before TrueSight unless explicitly overridden.
- [ ] Fire, weapons, violence, person-down, and tamper critical baselines remain unaffected.
- [ ] Suggested zones remain inactive until an authorized acceptance action.
- [ ] RTSP credentials are absent from artifacts, logs, health, and UI.
- [ ] Health names mapping failures and context suppression counts.
- [ ] Current Argus web console, not legacy mapper widgets, owns review controls.
- [ ] Full tests and JavaScript checks pass.
- [ ] Local Ollama smoke-test status is reported accurately.
