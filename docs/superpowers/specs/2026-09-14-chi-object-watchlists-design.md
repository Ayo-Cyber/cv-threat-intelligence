# Chi KPI 3 Object Watchlists V1

## Purpose

Build Chi KPI 3 as a local-first object watchlist system, not as a claim that
Argus can identify every product or SKU. Chi should be able to enroll examples
of products, vehicles, pallets, PPE, and other operational objects, bind those
objects to zones and rules, and receive verified alerts when a watched object is
seen, moved, removed, left behind, or loaded in a way that matters to the site.

The Ring-style product pattern is useful: the customer names something they care
about, and the system watches for it. The Argus version must avoid biometric
identity and must stay inside the existing product boundary:

```text
Customer defines object and policy.
Local detectors propose object candidates.
Local matching associates candidates with enrolled objects.
Tracking and zones describe state changes.
Rules decide whether the state matters.
TrueSight verifies important candidate events.
Operator remains final authority.
```

## Scope

This work covers:

- object enrollment from uploaded product photos and paused camera frames;
- an on-device object library containing reviewed examples, crops, embeddings,
  aliases, categories, zones, and rule bindings;
- local candidate proposal using generic detectors and a small open-vocabulary
  detector bakeoff;
- local crop embedding and similarity matching against enrolled objects;
- tracked object state transitions across zones;
- `object_watch` events consumed by the existing rules engine and alert queue;
- TrueSight verification only for rule-relevant object events;
- an acceptance matrix for Chi product, loading, removal, and negative cases;
- latency budgets and local resource controls suitable for edge deployment.

This work does not include face recognition, person identity recognition,
autonomous inventory counting, full SKU-level accuracy claims, cloud-only
inference, or training a custom detector before Chi supplies reviewed examples.

## Product Model

An enrolled object is a customer-owned target. It can start from either uploaded
photos or a crop drawn on a camera frame. The crop is only evidence; the durable
record is the label, examples, embeddings, scope, and rules.

```text
ObjectTarget
  id
  label
  category: product | vehicle | pallet | ppe | custom
  aliases
  examples
  embeddings
  negative_examples
  min_similarity
  allowed_zone_ids
  review_state
  created_by / reviewed_by
```

Example labels for Chi:

- `Chi carton`
- `finished goods pallet`
- `delivery truck`
- `loaded pallet`
- `empty pallet`
- `PPE hard hat`
- `restricted loading item`

The UI should say "possible match" until TrueSight or a human confirms. It must
not say "identified with certainty" or imply theft or misconduct from detection
alone.

## Runtime Architecture

The runtime has three speeds.

### Fast Path

The existing detector loop continues to run every frame or at the configured
target FPS. It should propose generic boxes for people, vehicles, and broad
object classes, and it should maintain tracks. This path must not call a VLM.

Initial generic sources:

- existing YOLO object/person detections;
- existing ByteTrack-style tracking;
- existing zone membership and dwell logic;
- existing vehicle/person events where available.

### Matched Object Path

Object matching runs locally and sampled. It should not process every crop on
every frame for every camera.

For each selected frame:

1. collect candidate boxes from the generic detector and, when enabled, the
   open-vocabulary proposal model;
2. crop candidate regions with stable padding;
3. compute crop embeddings in batches;
4. compare against reviewed `ObjectTarget` embeddings;
5. apply per-target threshold, negative examples, zone scope, and temporal
   smoothing;
6. attach an object identity candidate to a track only after persistence.

The matcher emits observations, not alerts:

```text
ObjectObservation
  camera_id
  object_id
  object_label
  category
  track_id
  bbox
  zone_id
  similarity
  timestamp_s
  persistence
```

### Slow Verification Path

TrueSight runs only when a rule-relevant state transition occurs. It receives a
small evidence bundle: chronological frames, the object crop, the enrolled
reference crop, zone context, and the exact rule question.

Example questions:

- "Does this evidence show the enrolled object 'Chi carton' in the loading bay?"
- "Does this sequence show the enrolled object being removed from the marked
  storage area?"
- "Does this sequence show product being loaded near a truck?"
- "Does this evidence show a non-PPE or unsafe object state?"

Verification failures remain fail-visible and must not be converted into
confirmed alerts.

## Model Strategy

V1 uses a bakeoff rather than a single assumed model.

### Required Local Baseline

Use existing YOLO plus a local image embedding model as the minimum viable path.
This keeps latency and memory predictable and makes uploaded examples useful
before any open-vocabulary detector is selected.

The embedding model should be SigLIP or CLIP-style and loaded once per process.
It should encode enrolled examples during setup and encode runtime crops in
batches. The first implementation should prefer a small model variant that can
run on CPU/MPS/CUDA locally, then record latency and memory before widening.

### Open-Vocabulary Bakeoff

Evaluate two local proposal options on the same frozen Chi object manifest:

- YOLO-World: likely best first runtime candidate because it is YOLO-shaped and
  intended for efficient prompt-then-detect open-vocabulary detection.
- YOLOE: promising because it supports text and visual prompts in an
  Ultralytics-style workflow; treat as a contender, not a default until measured.

Grounding DINO may be used as an offline enrollment or labeling helper, but it
should not be the default always-on runtime unless local measurements prove it
fits the edge latency and memory budget.

### No Cloud Dependency

The production path must run without cloud inference. Cloud models may be used
only for one-off research or labeling if the customer and data-rights rules
allow it. Pilot acceptance must be local.

## Latency And Resource Budget

Object Watchlists must not starve the existing detector, gate, or camera wall.
The default local budget is:

- generic detection and tracking: target 4-6 FPS per enabled camera;
- object matching: 1-2 FPS per watched camera, or event/zone-triggered sampling;
- embedding batches: bounded by max crops per sampled frame and max watched
  targets per camera;
- open-vocabulary proposals: disabled by default until measured, then enabled
  only per selected camera or setup run;
- TrueSight: event-only, never continuous;
- UI overlays: must not change detection state or candidate counts.

The implementation should expose explicit caps:

```text
object_watch_enabled
object_watch_sample_fps
object_watch_max_candidates_per_frame
object_watch_max_targets_per_camera
object_watch_min_similarity
object_watch_persistence_frames
object_watch_cooldown_seconds
object_watch_open_vocab_provider
```

When overloaded, the system should skip object-watch samples and report
`object_watch_skipped_over_budget`; it must not silently fall behind or block
critical safety detection.

## Events And Rules

The matcher and tracker emit `RawEvent`s under a new detector name:
`object_watch`.

Initial states:

- `object_seen`
- `object_entered_zone`
- `object_exited_zone`
- `object_removed`
- `object_left_behind`
- `object_loaded_near_vehicle`
- `object_arrangement_changed`
- `ppe_object_missing`
- `ppe_object_present`

Example Chi rules:

```json
{
  "name": "chi_product_loaded_near_truck",
  "trigger": {"detector": "object_watch", "state": "object_loaded_near_vehicle"},
  "context_filter": "object_category == 'product' and zone == 'loading_bay'",
  "priority": "high"
}
```

```json
{
  "name": "chi_product_removed_from_storage",
  "trigger": {"detector": "object_watch", "state": "object_removed"},
  "context_filter": "object_label == 'Chi carton' and zone == 'storage'",
  "priority": "high"
}
```

```json
{
  "name": "chi_product_left_in_walkway",
  "trigger": {"detector": "object_watch", "state": "object_left_behind"},
  "context_filter": "object_category == 'product' and zone == 'walkway' and dwell_seconds >= 120",
  "priority": "medium"
}
```

The rules engine remains the authority for whether an object observation matters.
An object match alone should not page an operator unless a configured rule says
it should.

## Storage

Store object library data under the site output or site configuration boundary,
not in global runtime scratch space.

Suggested artifacts:

```text
object_library/
  targets.json
  examples/<object_id>/<example_id>.jpg
  embeddings/<model_fingerprint>/<object_id>.json
```

`targets.json` should be human-readable enough for support, but embeddings may
be stored separately to avoid bloating site config. Every embedding record must
include model name, model fingerprint, preprocessing version, crop hash, and
created timestamp. If the embedding model changes, existing targets become
`needs_reembed` rather than silently mixing vector spaces.

## UI Workflow

The owner or installer can enroll objects in two ways:

1. upload product photos;
2. pause a camera frame and draw one or more boxes.

Enrollment flow:

1. choose category;
2. name object;
3. add 3-20 positive examples where possible;
4. optionally add negative examples such as competitor cartons or plain boxes;
5. choose allowed zones and alert templates;
6. review the generated object target;
7. mark it active.

Operators can view active targets and evidence but cannot mutate enrollment.
Enrollment changes must be audited.

## Acceptance Matrix

KPI 3 should be measured through concrete Chi cases, not a generic "object
accuracy" claim.

Minimum controlled recordings:

| Case | Expectation |
| --- | --- |
| `OBJ-P01` Chi product stationary in expected zone | Object matched after persistence; no alert unless rule requires seen-state alert. |
| `OBJ-P02` Chi product enters loading bay | `object_entered_zone` candidate; verified if the rule is enabled. |
| `OBJ-P03` Chi product removed from storage zone | `object_removed` candidate within labeled interval. |
| `OBJ-P04` product loaded near delivery truck | `object_loaded_near_vehicle` candidate; requires product and vehicle evidence. |
| `OBJ-P05` object left in walkway beyond dwell threshold | `object_left_behind` candidate after configured dwell. |
| `OBJ-N01` visually similar non-Chi carton | No Chi-object match above threshold, or TrueSight rejects. |
| `OBJ-N02` Chi product in allowed storage area | Match may be telemetry; no alert. |
| `OBJ-N03` ordinary person/vehicle motion without target product | No object-watch product candidate. |

Report:

- object-match recall by case;
- false matches on negative cases;
- event recall for zone/removal/loading states;
- duplicate candidates;
- duplicate shown alerts;
- detection delay;
- per-camera object-watch FPS and skipped-over-budget counts;
- TrueSight confirmation/rejection/error counts separately from detector metrics.

Do not claim SKU-level or inventory accuracy from these cases.

## Failure Handling

- If the embedding model is unavailable, object-watch targets become
  `degraded_unavailable`; critical safety detection continues.
- If an object target lacks reviewed examples, it cannot activate.
- If too many targets are assigned to one camera, the UI should show the cap and
  require the installer to narrow scope by zone or priority.
- If a model fingerprint changes, targets require re-embedding before use.
- If TrueSight fails, the candidate is retained as unverified and visible.
- If object identity is ambiguous, prefer a low-confidence observation or
  no-match over a confident false label.

## Implementation Order

1. Add object target schema, storage, and embedding fingerprints.
2. Add enrollment commands for uploaded images and frame crops.
3. Add local embedding model wrapper and offline target embedding.
4. Add crop proposal and matching path using existing detector boxes.
5. Add object tracking state and initial `object_seen` / zone transition events.
6. Add rule integration and TrueSight verification prompts.
7. Add object-watch audit rows and scorer inputs.
8. Add UI enrollment and object-watch status surfaces.
9. Run YOLO-World versus YOLOE bakeoff on the same Chi object manifest.
10. Capture and score the controlled Chi KPI 3 matrix.

## Open Questions Before Implementation

- Which Chi product families matter for the first demo: cartons, pallets,
  trucks, PPE, or all four?
- Does Chi need alerting on presence, movement/removal, loading, or arrangement
  change first?
- How many cameras will run object watchlists in the first pilot?
- What edge hardware is expected for the pilot machine?
- Can Chi provide 3-20 product photos and 8 controlled recordings with rights
  cleared for internal evaluation?
