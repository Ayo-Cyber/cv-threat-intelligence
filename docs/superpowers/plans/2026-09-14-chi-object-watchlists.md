# Chi KPI 3 Object Watchlists V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first object watchlist system for Chi KPI 3 so reviewed product/object examples can be enrolled, matched, tracked, verified, and scored without cloud inference or SKU-level overclaiming.

**Architecture:** Object examples become reviewed `ObjectTarget`s with crops and local embeddings. Runtime samples candidate boxes, embeds crops, matches them to active targets, tracks stable observations across zones, emits `object_watch` `RawEvent`s, and calls TrueSight only for configured object events. Acceptance is measured from controlled Chi recordings and audit rows, not inferred from demo footage.

**Tech Stack:** Python 3.9, OpenCV, NumPy, SQLite, existing YOLO/ByteTrack pipeline, optional local SigLIP/CLIP-compatible embedding backend, optional local YOLO-World/YOLOE bakeoff, FastAPI, Electron 44, React 19, TypeScript 7, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-14-chi-object-watchlists-design.md`

## Global Constraints

- Run locally by default; no production cloud inference for object watchlists.
- Do not add face recognition or person identity recognition.
- Do not claim autonomous inventory counting or SKU-level accuracy.
- TrueSight verifies rule-relevant candidate events only; it is never a continuous object detector.
- Generic safety detection must not be blocked by object-watch latency or model failures.
- Object watchlist overload must skip samples visibly with `object_watch_skipped_over_budget`.
- Enrollment changes require owner/installer authority and audit records.
- Operators can view object targets and evidence but cannot mutate enrollment.
- Existing Chi motion/concealment scenarios and prompt-baseline behavior must remain intact.
- Existing local dirty deployment/config files must not be overwritten.

---

## File Structure

- `cvti/object_watch/__init__.py` exports the object-watch public interfaces.
- `cvti/object_watch/store.py` owns target schema, crop storage, embedding metadata, and atomic writes.
- `cvti/object_watch/embeddings.py` owns the local embedding backend abstraction and deterministic fallback for tests.
- `cvti/object_watch/matcher.py` owns candidate crop normalization, cosine matching, negative-example rejection, persistence, and per-frame caps.
- `cvti/object_watch/tracker.py` owns per-camera object state transitions such as seen, entered, exited, removed, left behind, and loaded near vehicle.
- `cvti/event_adapters.py` gains `object_observations_to_events()`.
- `cvti/serving/camera.py` wires object watch into `PerCameraState` behind explicit camera flags.
- `cvti/serving/alert_sink.py` persists an `object_watch_audit` lifecycle table.
- `cvti/app/console_backend.py`, `cvti/api/writes.py`, `cvti/api/app.py`, and `cvti/api/sources.py` expose enrollment and target reads through existing permission patterns.
- `Frontend/src/components/ObjectWatchlistManager.tsx`, `Frontend/src/lib/types.ts`, and frontend tests add the enrollment surface.
- `tools/score_chi_objects.py` scores KPI 3 controlled recordings and audits.
- `tools/object_model_bakeoff.py` measures YOLO-World versus YOLOE proposal quality and local latency.
- `configs/chi_object_watch_v1.json` contains initial Chi object-watch rules.
- `docs/CHI_PILOT_TESTING.md` gains the KPI 3 recording and scoring workflow.

---

### Task 1: Add Object Target Storage And Validation

**Files:**
- Create: `cvti/object_watch/__init__.py`
- Create: `cvti/object_watch/store.py`
- Create: `tests/test_object_watch_store.py`

**Interfaces:**
- Produces: `ObjectTarget`, `ObjectExample`, `EmbeddingRecord`
- Produces: `load_targets(root: str | Path) -> list[ObjectTarget]`
- Produces: `save_target(root: str | Path, target: ObjectTarget) -> ObjectTarget`
- Produces: `add_example(root: str | Path, object_id: str, image_bytes: bytes, bbox: tuple[int, int, int, int], source: str) -> ObjectExample`
- Produces: `write_embedding(root: str | Path, object_id: str, example_id: str, record: EmbeddingRecord) -> None`
- Produces: `targets_needing_reembed(root: str | Path, model_fingerprint: str) -> list[ObjectTarget]`
- Consumes: standard library JSON, hashlib, pathlib, tempfile

- [ ] **Step 1: Write failing target validation tests**

```python
def test_save_target_requires_reviewed_examples_before_activation(tmp_path):
    from cvti.object_watch.store import ObjectTarget, save_target

    target = ObjectTarget(
        id="chi-carton",
        label="Chi carton",
        category="product",
        aliases=("milk carton",),
        review_state="active",
        min_similarity=0.72,
        allowed_zone_ids=("loading_bay",),
        examples=(),
    )

    with pytest.raises(ValueError, match="active target requires at least one example"):
        save_target(tmp_path, target)
```

```python
def test_embedding_model_fingerprint_controls_reembed(tmp_path):
    from cvti.object_watch.store import (
        ObjectTarget, add_example, save_target, write_embedding,
        EmbeddingRecord, targets_needing_reembed,
    )

    target = save_target(tmp_path, ObjectTarget(
        id="chi-carton", label="Chi carton", category="product",
        aliases=(), review_state="draft", min_similarity=0.72,
        allowed_zone_ids=(), examples=(),
    ))
    example = add_example(tmp_path, target.id, b"\xff\xd8\xff\xd9", (0, 0, 10, 10), "upload")
    write_embedding(tmp_path, target.id, example.id, EmbeddingRecord(
        model_name="test-embed", model_fingerprint="fp-a",
        preprocessing_version=1, crop_sha256=example.sha256,
        vector=(1.0, 0.0, 0.0),
    ))

    assert targets_needing_reembed(tmp_path, "fp-a") == []
    assert [t.id for t in targets_needing_reembed(tmp_path, "fp-b")] == ["chi-carton"]
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_watch_store.py -q`

Expected: FAIL because `cvti.object_watch.store` does not exist.

- [ ] **Step 3: Implement dataclasses and atomic storage**

Create frozen-enough dataclasses that serialize to JSON-compatible dicts:

```python
@dataclass(frozen=True)
class ObjectExample:
    id: str
    source: str
    path: str
    bbox: tuple[int, int, int, int]
    sha256: str
    reviewed: bool = True

@dataclass(frozen=True)
class EmbeddingRecord:
    model_name: str
    model_fingerprint: str
    preprocessing_version: int
    crop_sha256: str
    vector: tuple[float, ...]

@dataclass(frozen=True)
class ObjectTarget:
    id: str
    label: str
    category: str
    aliases: tuple[str, ...]
    review_state: str
    min_similarity: float
    allowed_zone_ids: tuple[str, ...]
    examples: tuple[ObjectExample, ...]
    negative_examples: tuple[ObjectExample, ...] = ()
```

Use `object_library/targets.json`, `object_library/examples/<object_id>/`, and
`object_library/embeddings/<model_fingerprint>/<object_id>.json`. Validate IDs
with the repo's existing slug-safe conventions: lowercase ASCII letters,
numbers, dash, and underscore. Reject unknown categories, empty labels,
non-finite thresholds, thresholds outside `[0.0, 1.0]`, duplicate IDs, and
active targets without examples.

- [ ] **Step 4: Add round-trip, corrupt-file, and path-safety tests**

Add tests for:

```python
def test_load_targets_round_trips_examples_and_negative_examples(tmp_path): ...
def test_add_example_rejects_path_traversal_object_id(tmp_path): ...
def test_load_targets_rejects_invalid_json_without_deleting_file(tmp_path): ...
def test_save_target_replaces_targets_atomically(tmp_path): ...
def test_unknown_category_is_rejected(tmp_path): ...
```

- [ ] **Step 5: Run focused storage tests**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_watch_store.py -q`

Expected: PASS.

- [ ] **Step 6: Commit storage layer**

```bash
git add cvti/object_watch/__init__.py cvti/object_watch/store.py tests/test_object_watch_store.py
git commit -m "feat(object-watch): add target storage"
```

---

### Task 2: Add Local Embedding Backend And Deterministic Test Backend

**Files:**
- Create: `cvti/object_watch/embeddings.py`
- Create: `tests/test_object_watch_embeddings.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: Task 1 `EmbeddingRecord`
- Produces: `EmbeddingBackend`
- Produces: `HashEmbeddingBackend`
- Produces: `load_embedding_backend(name: str = "hash") -> EmbeddingBackend`
- Produces: `embed_examples(root: str | Path, backend: EmbeddingBackend) -> int`

- [ ] **Step 1: Write failing backend contract tests**

```python
def test_hash_backend_is_deterministic_and_normalized():
    from cvti.object_watch.embeddings import HashEmbeddingBackend

    backend = HashEmbeddingBackend(dimensions=8)
    first = backend.embed_image(b"chi carton")
    second = backend.embed_image(b"chi carton")

    assert first == second
    assert abs(sum(v * v for v in first) - 1.0) < 1e-6
    assert backend.fingerprint
```

```python
def test_embed_examples_writes_records_for_missing_embeddings(tmp_path):
    from cvti.object_watch.store import ObjectTarget, add_example, save_target, load_embeddings
    from cvti.object_watch.embeddings import HashEmbeddingBackend, embed_examples

    save_target(tmp_path, ObjectTarget(
        id="chi-carton", label="Chi carton", category="product",
        aliases=(), review_state="draft", min_similarity=0.72,
        allowed_zone_ids=(), examples=(),
    ))
    add_example(tmp_path, "chi-carton", b"\xff\xd8carton\xff\xd9", (0, 0, 12, 12), "upload")

    written = embed_examples(tmp_path, HashEmbeddingBackend(dimensions=8))

    assert written == 1
    assert load_embeddings(tmp_path, "chi-carton", "hash-8-v1")
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_watch_embeddings.py -q`

Expected: FAIL because `embeddings.py` and `load_embeddings()` are absent.

- [ ] **Step 3: Implement deterministic backend and storage helpers**

Implement a lightweight `HashEmbeddingBackend` for tests and environments
without the real model:

```python
class EmbeddingBackend(Protocol):
    name: str
    fingerprint: str
    preprocessing_version: int
    def embed_image(self, image_bytes: bytes) -> tuple[float, ...]: ...
```

`HashEmbeddingBackend` hashes image bytes into a fixed vector and L2-normalizes
it. Add `load_embeddings(root, object_id, model_fingerprint)` to Task 1 storage.

- [ ] **Step 4: Add optional real-backend seam without making it required**

Add `SiglipEmbeddingBackend` behind lazy imports. If `transformers` or model
weights are unavailable, `load_embedding_backend("siglip")` raises a clear
`RuntimeError("SigLIP embedding backend unavailable: ...")`. Do not download
weights automatically. Do not change CI to require this backend.

- [ ] **Step 5: Add tests for missing optional backend and no auto-download**

```python
def test_siglip_backend_failure_is_explicit(monkeypatch):
    from cvti.object_watch.embeddings import load_embedding_backend

    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="SigLIP embedding backend unavailable"):
        load_embedding_backend("siglip")
```

- [ ] **Step 6: Run focused tests**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_watch_store.py tests/test_object_watch_embeddings.py -q`

Expected: PASS.

- [ ] **Step 7: Commit embedding layer**

```bash
git add cvti/object_watch/store.py cvti/object_watch/embeddings.py tests/test_object_watch_store.py tests/test_object_watch_embeddings.py requirements.txt
git commit -m "feat(object-watch): add local embedding backend"
```

---

### Task 3: Match Candidate Crops To Enrolled Targets

**Files:**
- Create: `cvti/object_watch/matcher.py`
- Create: `tests/test_object_watch_matcher.py`

**Interfaces:**
- Consumes: `ObjectTarget`, `EmbeddingBackend`, `load_embeddings`
- Produces: `ObjectCandidate`
- Produces: `ObjectMatch`
- Produces: `ObjectMatcher.match(camera_id: str, frame: np.ndarray, candidates: list[ObjectCandidate], timestamp: float) -> list[ObjectMatch]`

- [ ] **Step 1: Write failing crop-match tests**

```python
def test_matcher_returns_best_positive_above_threshold(tmp_path):
    from cvti.object_watch.matcher import ObjectCandidate, ObjectMatcher
    from cvti.object_watch.embeddings import HashEmbeddingBackend
    from cvti.object_watch.store import ObjectTarget, add_example, embed_target_for_test, save_target

    save_target(tmp_path, ObjectTarget(
        id="chi-carton", label="Chi carton", category="product",
        aliases=(), review_state="active", min_similarity=0.6,
        allowed_zone_ids=(), examples=(),
    ))
    add_example(tmp_path, "chi-carton", b"same crop", (0, 0, 8, 8), "upload")
    embed_target_for_test(tmp_path, "chi-carton", HashEmbeddingBackend(dimensions=8))

    matcher = ObjectMatcher(tmp_path, HashEmbeddingBackend(dimensions=8))
    match = matcher.match("cam1", image_from_bytes(b"same crop"), [
        ObjectCandidate(bbox=(0, 0, 8, 8), label_hint="box", confidence=0.8),
    ], timestamp=1.0)

    assert match[0].object_id == "chi-carton"
    assert match[0].similarity >= 0.6
```

```python
def test_negative_example_can_veto_similar_plain_box(tmp_path): ...
def test_max_candidates_per_frame_is_enforced(tmp_path): ...
def test_zone_scoped_target_does_not_match_outside_zone(tmp_path): ...
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_watch_matcher.py -q`

Expected: FAIL because `matcher.py` is absent.

- [ ] **Step 3: Implement candidate and match dataclasses**

```python
@dataclass(frozen=True)
class ObjectCandidate:
    bbox: tuple[int, int, int, int]
    label_hint: str = ""
    confidence: float = 0.0
    track_id: int | None = None
    zone_id: str | None = None

@dataclass(frozen=True)
class ObjectMatch:
    camera_id: str
    object_id: str
    object_label: str
    category: str
    bbox: tuple[int, int, int, int]
    similarity: float
    timestamp: float
    track_id: int | None = None
    zone_id: str | None = None
```

Implement cosine similarity. Positive score is the maximum similarity to the
target's examples. Negative score is the maximum similarity to the target's
negative examples. Accept a match only when positive score is at or above
`min_similarity` and at least `0.05` above the negative score.

- [ ] **Step 4: Add preprocessing and budget behavior**

Clamp boxes to the frame, reject zero-area crops, process at most
`max_candidates_per_frame`, and report skipped count in `ObjectMatcher.last_stats`.
Batching may be internal, but the public behavior must be deterministic.

- [ ] **Step 5: Run focused matcher tests**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_watch_matcher.py -q`

Expected: PASS.

- [ ] **Step 6: Commit matcher**

```bash
git add cvti/object_watch/matcher.py tests/test_object_watch_matcher.py
git commit -m "feat(object-watch): match enrolled objects locally"
```

---

### Task 4: Track Object State And Emit Object Watch Events

**Files:**
- Create: `cvti/object_watch/tracker.py`
- Create: `tests/test_object_watch_tracker.py`
- Modify: `cvti/event_adapters.py`
- Modify: `tests/test_zone_customization.py`

**Interfaces:**
- Consumes: `ObjectMatch`
- Produces: `ObjectStateEvent`
- Produces: `ObjectStateTracker.update(matches: list[ObjectMatch], timestamp: float, vehicles: list[tuple[int, int, int, int]] = ()) -> list[ObjectStateEvent]`
- Produces: `object_observations_to_events(events: list[ObjectStateEvent], timestamp: float = 0.0) -> list[RawEvent]`

- [ ] **Step 1: Write failing tracker state tests**

```python
def test_entered_and_exited_zone_are_edges_not_continuous():
    from cvti.object_watch.matcher import ObjectMatch
    from cvti.object_watch.tracker import ObjectStateTracker

    tracker = ObjectStateTracker(left_behind_seconds=120)
    first = tracker.update([ObjectMatch("cam1", "chi-carton", "Chi carton", "product", (1, 1, 10, 10), 0.8, 1.0, 7, "storage")], 1.0)
    second = tracker.update([ObjectMatch("cam1", "chi-carton", "Chi carton", "product", (2, 2, 11, 11), 0.8, 2.0, 7, "storage")], 2.0)
    third = tracker.update([ObjectMatch("cam1", "chi-carton", "Chi carton", "product", (20, 20, 30, 30), 0.8, 3.0, 7, "loading_bay")], 3.0)

    assert [e.state for e in first] == ["object_seen", "object_entered_zone"]
    assert [e.state for e in second] == ["object_seen"]
    assert "object_exited_zone" in [e.state for e in third]
    assert "object_entered_zone" in [e.state for e in third]
```

```python
def test_removed_fires_when_stable_object_disappears_after_grace(): ...
def test_left_behind_requires_stationary_dwell(): ...
def test_loaded_near_vehicle_requires_product_and_vehicle_overlap(): ...
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_watch_tracker.py -q`

Expected: FAIL because `tracker.py` is absent.

- [ ] **Step 3: Implement object state tracker**

Track by `(camera_id, object_id, track_id)` when a track exists, otherwise by
nearest stable box. Maintain last zone, first seen timestamp, last seen
timestamp, stationary dwell, and last emitted transition. `object_removed`
fires when a previously stable object disappears after `removed_grace_seconds`.
`object_left_behind` fires when a stationary object remains beyond
`left_behind_seconds`. `object_loaded_near_vehicle` fires when a product match
overlaps or approaches a vehicle box in a loading zone.

- [ ] **Step 4: Add RawEvent adapter**

`object_observations_to_events()` returns `RawEvent(detector="object_watch")`
with `state`, `object_label`, `timestamp`, and `extra` containing:

```python
{
    "object_id": event.object_id,
    "object_category": event.category,
    "zone": event.zone_id,
    "track_id": event.track_id,
    "bbox": event.bbox,
    "similarity": event.similarity,
    "dwell_seconds": event.dwell_seconds,
    "reasons": event.reasons,
}
```

- [ ] **Step 5: Add rules-engine integration tests**

Use `CustomizationEngine.evaluate()` with a rule for
`detector == "object_watch"` and `state == "object_removed"`. Assert that
`CandidateAlert.metadata` retains `object_id`, `object_category`, `zone`, and
`bbox`.

- [ ] **Step 6: Run focused tests**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_watch_tracker.py tests/test_zone_customization.py -q`

Expected: PASS.

- [ ] **Step 7: Commit state tracker**

```bash
git add cvti/object_watch/tracker.py cvti/event_adapters.py tests/test_object_watch_tracker.py tests/test_zone_customization.py
git commit -m "feat(object-watch): emit tracked object events"
```

---

### Task 5: Wire Object Watch Into The Serving Pipeline And Audit Lifecycle

**Files:**
- Modify: `cvti/serving/camera.py`
- Modify: `cvti/serving/alert_sink.py`
- Modify: `cvti/serving/retention.py`
- Modify: `cvti/verification/gate.py`
- Create: `configs/chi_object_watch_v1.json`
- Create: `tests/test_object_watch_serving.py`
- Modify: `tests/test_alert_sink.py`
- Modify: `tests/test_retention.py`
- Modify: `tests/test_gate_evidence_quality.py`

**Interfaces:**
- Consumes: `ObjectMatcher`, `ObjectStateTracker`, `object_observations_to_events`
- Produces camera config flags: `object_watch`, `object_watch_library`, `object_watch_sample_fps`, `object_watch_max_candidates_per_frame`, `object_watch_min_similarity`, `object_watch_open_vocab_provider`
- Produces SQLite table: `object_watch_audit`

- [ ] **Step 1: Write failing serving config tests**

```python
def test_camera_config_accepts_object_watch_flags(tmp_path):
    from cvti.serving.camera import build_camera_states

    site = {
        "cameras": [{
            "id": "cam1",
            "source": "data/test_clips/normal_street_01.mp4",
            "config": "configs/chi_object_watch_v1.json",
            "object_watch": True,
            "object_watch_library": str(tmp_path / "object_library"),
            "object_watch_sample_fps": 1.0,
            "object_watch_max_candidates_per_frame": 12,
        }]
    }

    states = build_camera_states(site, output_dir=tmp_path)

    assert states[0].object_watch is True
    assert states[0].object_watch_sample_fps == 1.0
```

- [ ] **Step 2: Write failing audit lifecycle tests**

Assert `AlertSink` creates `object_watch_audit` with columns:

```text
id, candidate_id, camera_id, object_id, object_label, state, generated_at,
timestamp_s, zone, similarity, admission_status, admitted_at, gate_status,
verdict_at, persisted_event_id, payload_json
```

Add tests for generated, admitted, deduplicated, capacity-dropped, confirmed,
rejected, unverified, and retention purge behavior.

- [ ] **Step 3: Run focused tests and verify they fail**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_watch_serving.py tests/test_alert_sink.py tests/test_retention.py -q`

Expected: FAIL because serving and audit wiring are absent.

- [ ] **Step 4: Add camera state wiring behind flags**

In `PerCameraState`, load object-watch components only when `object_watch` is
true. Reuse current detections as proposals first. Do not load optional
open-vocabulary models in this task. Add sample scheduling so object matching
runs at `object_watch_sample_fps`, not every frame. Emit skipped-over-budget
counts into the existing perf/health mechanism.

- [ ] **Step 5: Add audit payload callbacks**

Follow the `concealment_audit` and `motion_candidate_audit` pattern. Insert
an audit row before queue admission. Store the row ID or candidate ID in
`CandidateAlert.metadata`. Update it when the queue admits, deduplicates,
drops for capacity, gate-verifies, or persists an event.

- [ ] **Step 6: Add TrueSight object prompt**

In `cvti/verification/gate.py`, add question templates for:

```text
object_seen
object_removed
object_left_behind
object_loaded_near_vehicle
object_arrangement_changed
ppe_object_missing
ppe_object_present
```

The wording must ask about the specific enrolled label and must reject generic
similar objects when the reference crop/evidence does not support the claim.
Update `docs/prompt_baseline.json` fingerprint only as `measurement_status:
unmeasured` unless the frozen corpus is actually replayed.

- [ ] **Step 7: Add initial Chi object rules**

Create `configs/chi_object_watch_v1.json` with three rules:

```json
{
  "use_case_id": "chi_object_watch_v1",
  "rules": [
    {
      "name": "chi_product_removed_from_storage",
      "trigger": {"detector": "object_watch", "state": "object_removed"},
      "context_filter": "object_category == 'product' and zone == 'storage'",
      "priority": "high"
    },
    {
      "name": "chi_product_loaded_near_truck",
      "trigger": {"detector": "object_watch", "state": "object_loaded_near_vehicle"},
      "context_filter": "object_category == 'product' and zone == 'loading_bay'",
      "priority": "high"
    },
    {
      "name": "chi_product_left_in_walkway",
      "trigger": {"detector": "object_watch", "state": "object_left_behind"},
      "context_filter": "object_category == 'product' and zone == 'walkway' and dwell_seconds >= 120",
      "priority": "medium"
    }
  ]
}
```

- [ ] **Step 8: Run focused serving and audit tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/test_object_watch_serving.py tests/test_alert_sink.py \
  tests/test_retention.py tests/test_gate_evidence_quality.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit serving integration**

```bash
git add cvti/serving/camera.py cvti/serving/alert_sink.py cvti/serving/retention.py cvti/verification/gate.py configs/chi_object_watch_v1.json tests/test_object_watch_serving.py tests/test_alert_sink.py tests/test_retention.py tests/test_gate_evidence_quality.py docs/prompt_baseline.json
git commit -m "feat(object-watch): wire object events into serving"
```

---

### Task 6: Expose Enrollment Through Backend And API Permissions

**Files:**
- Modify: `cvti/app/console_backend.py`
- Modify: `cvti/api/writes.py`
- Modify: `cvti/api/sources.py`
- Modify: `cvti/api/app.py`
- Modify: `docs/api-v1.md`
- Modify: `docs/openapi.json`
- Create: `tests/test_object_watch_api.py`
- Modify: `tests/test_api_contract_is_frozen.py`

**Interfaces:**
- Consumes: `save_target`, `add_example`, `embed_examples`, `load_targets`
- Produces ConsoleBackend methods: `object_targets()`, `create_object_target(target: dict)`, `add_object_example(object_id: str, image_b64: str, bbox: list[int], source: str)`, `activate_object_target(object_id: str)`, `reembed_object_targets(model: str = "hash")`
- Produces API routes under `/api/v1/object-targets`

- [ ] **Step 1: Write failing permission tests**

```python
def test_owner_can_create_object_target_and_operator_cannot(self):
    owner = self.client.post("/api/v1/object-targets", headers=self.owner, json={
        "target": {
            "id": "chi-carton",
            "label": "Chi carton",
            "category": "product",
            "aliases": [],
            "allowed_zone_ids": ["storage"],
            "min_similarity": 0.72
        }
    })
    assert owner.status_code == 201

    operator = self.client.post("/api/v1/object-targets", headers=self.operator, json={
        "target": {"id": "x", "label": "X", "category": "product"}
    })
    assert operator.status_code == 403
    assert operator.json()["error"]["detail"]["permission"] == "configure_cameras"
```

```python
def test_object_target_reads_are_allowed_with_view_live(self): ...
def test_add_example_rejects_non_image_base64(): ...
def test_activation_requires_reviewed_example(): ...
```

- [ ] **Step 2: Run API tests and verify they fail**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_watch_api.py -q`

Expected: FAIL because routes are absent.

- [ ] **Step 3: Add ConsoleBackend methods**

Use existing permission constants. Reads require `VIEW_LIVE`; writes require
`CONFIGURE_CAMERAS`. Audit every create, example addition, activation, and
re-embedding action with the object ID and label, never with raw image bytes.

- [ ] **Step 4: Add API write/read routes**

Routes:

```text
GET /api/v1/object-targets
POST /api/v1/object-targets
POST /api/v1/object-targets/{object_id}/examples
POST /api/v1/object-targets/{object_id}/activate
POST /api/v1/object-targets/reembed
```

Reject payloads above the existing IPC/API size limit. Return redacted target
records: example IDs and crop hashes are fine; raw local paths and image bytes
are not returned by default.

- [ ] **Step 5: Update API docs and OpenAPI**

Document permissions, payloads, redaction, model-fingerprint behavior, and
local-only operation in `docs/api-v1.md`. Regenerate `docs/openapi.json` using
the repo's existing OpenAPI generation command.

- [ ] **Step 6: Run contract and API tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/test_object_watch_api.py tests/test_api_contract_is_frozen.py \
  tests/test_api_write_side.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit API enrollment**

```bash
git add cvti/app/console_backend.py cvti/api/writes.py cvti/api/sources.py cvti/api/app.py docs/api-v1.md docs/openapi.json tests/test_object_watch_api.py tests/test_api_contract_is_frozen.py
git commit -m "feat(api): expose object watchlist enrollment"
```

---

### Task 7: Add Frontend Object Enrollment And Status UI

**Files:**
- Create: `Frontend/src/components/ObjectWatchlistManager.tsx`
- Create: `Frontend/tests/object-watchlist.test.ts`
- Modify: `Frontend/src/lib/types.ts`
- Modify: `Frontend/src/lib/demo.ts`
- Modify: `Frontend/src/App.tsx`
- Modify: `Frontend/src/components/SettingsPanel.tsx`
- Modify: `Frontend/tests/api-client.test.ts`
- Modify: `Frontend/tests/ui.spec.ts`
- Modify: `Frontend/src/styles.css`

**Interfaces:**
- Consumes: API routes from Task 6
- Produces frontend types: `ObjectTarget`, `ObjectExample`, `ObjectWatchStatus`
- Produces UI component: `<ObjectWatchlistManager transport authState hierarchy />`

- [ ] **Step 1: Write failing frontend adapter tests**

```typescript
it("creates an object target through the API adapter", async () => {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  const client = new ArgusApiClient("http://127.0.0.1:8787", {
    fetch: async (url, init) => {
      calls.push({ url: String(url), init: init ?? {} });
      return jsonResponse({ target: { id: "chi-carton", label: "Chi carton" } }, 201);
    },
  });

  await client.invoke("create_object_target", [{
    id: "chi-carton",
    label: "Chi carton",
    category: "product",
  }]);

  expect(calls[0].url).toContain("/api/v1/object-targets");
  expect(calls[0].init.method).toBe("POST");
});
```

- [ ] **Step 2: Write failing UI tests**

Test that owners/installers see object enrollment, operators see read-only
targets, upload requires a label/category, activating without examples shows the
API error, and demo data remains visibly fixture-backed.

- [ ] **Step 3: Run focused frontend tests and verify they fail**

Run: `cd Frontend && npm test -- object-watchlist.test.ts api-client.test.ts`

Expected: FAIL because object target types and operations are absent.

- [ ] **Step 4: Add frontend types and adapter operations**

Add methods to the typed API operation registry:

```text
object_targets
create_object_target
add_object_example
activate_object_target
reembed_object_targets
```

Normalize errors into the existing `ApiError` shape. Do not expose image bytes
after upload.

- [ ] **Step 5: Implement ObjectWatchlistManager**

The component should support:

- list active/draft/degraded targets;
- create target with label, category, aliases, zones, and threshold;
- upload an image file and send base64 plus a full-image bbox when no frame crop
  is used;
- activate only after examples exist;
- show `needs_reembed` and `degraded_unavailable`;
- hide mutation controls for operators.

- [ ] **Step 6: Add camera-frame crop enrollment entry point**

From camera details or settings, allow "Enroll object from frame" using the
current frame snapshot and drawn rectangle. The saved crop uses the same API as
uploaded images, with `source="camera_frame"`.

- [ ] **Step 7: Run frontend tests and build**

Run:

```bash
cd Frontend
npm test -- object-watchlist.test.ts api-client.test.ts
npm test
npm run build
```

Expected: PASS.

- [ ] **Step 8: Commit frontend enrollment**

```bash
git add Frontend/src/components/ObjectWatchlistManager.tsx Frontend/tests/object-watchlist.test.ts Frontend/src/lib/types.ts Frontend/src/lib/demo.ts Frontend/src/App.tsx Frontend/src/components/SettingsPanel.tsx Frontend/tests/api-client.test.ts Frontend/tests/ui.spec.ts Frontend/src/styles.css
git commit -m "feat(frontend): enroll object watchlist targets"
```

---

### Task 8: Add KPI 3 Scorer And Chi Object Acceptance Workflow

**Files:**
- Create: `tools/score_chi_objects.py`
- Create: `tests/test_score_chi_objects.py`
- Modify: `docs/CHI_PILOT_TESTING.md`
- Modify: `docs/PROJECT_CONTEXT.md`

**Interfaces:**
- Consumes: `object_watch_audit` SQLite table or exported JSON
- Produces: retained JSON score artifact for Chi KPI 3

- [ ] **Step 1: Write failing scorer input-validation tests**

Required label CSV columns:

```text
case_id,clip_sha256,clip_duration_s,target_fps,start_s,end_s,label,
object_ref,expected_zone,expected_state,expected_object_id,incident_id
```

Required observation CSV columns:

```text
case_id,timestamp_s,object_ref,visible,detected,track_id,object_id,
similarity,zone,bbox
```

Required audit columns:

```text
candidate_id,case_id,timestamp_s,object_id,object_label,state,
admission_status,gate_status,persisted_event_id
```

Tests:

```python
def test_missing_visible_observation_invalidates_score(tmp_path): ...
def test_candidate_outside_positive_interval_is_false_positive(tmp_path): ...
def test_duplicate_persisted_alert_is_reported(tmp_path): ...
def test_unknown_case_in_audit_is_malformed(tmp_path): ...
```

- [ ] **Step 2: Run scorer tests and verify they fail**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_score_chi_objects.py -q`

Expected: FAIL because `tools.score_chi_objects` is absent.

- [ ] **Step 3: Implement scorer**

Follow `tools/score_chi_motion.py` conventions: strict CSV validation,
half-open intervals, final-frame exception, explicit visible misses, candidate
classification, duplicate candidate and persisted-alert counts, and retained
input hashes. Add metrics:

```text
object_match_recall
false_match_count
event_recall_by_state
event_precision_by_state
duplicate_candidates
duplicate_persisted_alerts
detection_delay_s
gate_status_counts
```

- [ ] **Step 4: Add Chi KPI 3 recording matrix to docs**

Append cases from the design:

```text
OBJ-P01 Chi product stationary in expected zone
OBJ-P02 Chi product enters loading bay
OBJ-P03 Chi product removed from storage
OBJ-P04 product loaded near delivery truck
OBJ-P05 object left in walkway beyond dwell threshold
OBJ-N01 visually similar non-Chi carton
OBJ-N02 Chi product in allowed storage area
OBJ-N03 person/vehicle motion without target product
```

Include commands for staging clips outside Git, hashing files, creating
labels/observations, exporting `object_watch_audit`, and running the scorer.

- [ ] **Step 5: Run scorer and docs-focused tests**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/test_score_chi_objects.py tests/test_score_chi_motion.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit scorer and workflow**

```bash
git add tools/score_chi_objects.py tests/test_score_chi_objects.py docs/CHI_PILOT_TESTING.md docs/PROJECT_CONTEXT.md
git commit -m "test(object-watch): add Chi KPI 3 scorer"
```

---

### Task 9: Add Local Model Bakeoff Harness

**Files:**
- Create: `tools/object_model_bakeoff.py`
- Create: `tests/test_object_model_bakeoff.py`
- Modify: `docs/CHI_PILOT_TESTING.md`
- Modify: `docs/BACKLOG.md`

**Interfaces:**
- Consumes: object library and KPI 3 manifest clips
- Produces: `runs/eval/object_watch/bakeoff_<provider>.json`

- [ ] **Step 1: Write failing bakeoff manifest tests**

```python
def test_bakeoff_uses_same_frozen_cases_for_every_provider(tmp_path): ...
def test_bakeoff_marks_missing_provider_as_unavailable_not_failed_accuracy(tmp_path): ...
def test_latency_summary_reports_median_p95_and_memory_when_available(tmp_path): ...
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_model_bakeoff.py -q`

Expected: FAIL because the bakeoff tool is absent.

- [ ] **Step 3: Implement provider interface**

Provider names:

```text
generic-yolo-embeddings
yolo-world
yoloe
grounding-dino-offline
```

Only `generic-yolo-embeddings` is required to run in CI. Other providers must
return a structured unavailable result when dependencies or weights are absent.

- [ ] **Step 4: Implement output schema**

Each output JSON records:

```json
{
  "schema_version": 1,
  "provider": "generic-yolo-embeddings",
  "manifest_digest": "...",
  "model_versions": {},
  "latency": {"median_ms": 0.0, "p95_ms": 0.0},
  "memory": {"peak_mb": null},
  "rows": []
}
```

Do not compare providers unless they ran on the exact same manifest digest.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_object_model_bakeoff.py -q`

Expected: PASS.

- [ ] **Step 6: Commit bakeoff harness**

```bash
git add tools/object_model_bakeoff.py tests/test_object_model_bakeoff.py docs/CHI_PILOT_TESTING.md docs/BACKLOG.md
git commit -m "test(object-watch): add local model bakeoff harness"
```

---

### Task 10: Final Verification And Publication Preparation

**Files:**
- Review only: all files changed in Tasks 1-9
- Modify: `docs/PROJECT_CONTEXT.md`
- Do not stage: local footage, object crops from real customers, `.venv`, runtime databases, screenshots, recordings, generated model weights

**Interfaces:**
- Consumes: all previous task interfaces
- Produces: reviewable `feat/chi-object-watchlists-v1` branch

- [ ] **Step 1: Run focused backend verification**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/test_object_watch_store.py tests/test_object_watch_embeddings.py \
  tests/test_object_watch_matcher.py tests/test_object_watch_tracker.py \
  tests/test_object_watch_serving.py tests/test_object_watch_api.py \
  tests/test_score_chi_objects.py tests/test_object_model_bakeoff.py -q
```

Expected: PASS.

- [ ] **Step 2: Run regression suites touched by the integration**

Run:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/test_serving.py tests/test_alert_sink.py tests/test_retention.py \
  tests/test_gate_evidence_quality.py tests/test_prompt_regression.py \
  tests/test_score_chi_motion.py tests/test_api_write_side.py \
  tests/test_api_contract_is_frozen.py -q
```

Expected: PASS, except the already-known local Ultralytics mismatch if using the
shared off-pin interpreter for the full suite. Do not hide or relabel that
environment failure.

- [ ] **Step 3: Run frontend verification**

Run:

```bash
cd Frontend
npm test
npm run build
```

Expected: PASS.

- [ ] **Step 4: Run prompt check**

Run: `PYTHONPATH=. ./.venv/bin/python tools/prompt_regression.py check`

Expected: PASS with object-watch prompt metrics reported as `UNMEASURED` unless
the frozen corpus has actually been replayed.

- [ ] **Step 5: Inspect final diff**

Run:

```bash
git status --short
git diff --check
git diff ayo/main...HEAD --stat
```

Confirm no customer images, clips, embeddings from real footage, runtime DBs,
model weights, `.venv`, or unrelated local config files are staged.

- [ ] **Step 6: Update project context**

Prepend a checkpoint to `docs/PROJECT_CONTEXT.md` stating:

- Object Watchlists V1 is implemented as local-first candidate generation plus
  embedding matching and event-only TrueSight verification.
- KPI 3 remains unmeasured until controlled Chi recordings and labels are run.
- Face recognition, SKU inventory counting, and cloud inference remain out of
  scope.
- The local model bakeoff records latency and resource fit before selecting an
  open-vocabulary runtime model.

- [ ] **Step 7: Commit final docs**

```bash
git add docs/PROJECT_CONTEXT.md
git commit -m "docs(object-watch): record Chi KPI 3 status"
```

- [ ] **Step 8: Prepare handoff**

Write a final summary with:

- branch name;
- base commit;
- commits created;
- verification commands and results;
- unmeasured Chi acceptance cells;
- local model bakeoff status;
- explicit instruction not to advertise KPI 3 accuracy until `score_chi_objects`
  runs on controlled Chi footage.
