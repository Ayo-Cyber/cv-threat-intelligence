# Hierarchical Scene Mapping and Review Design

**Date:** 2026-09-01

**Status:** Implemented; real multi-camera Ollama acceptance remains pending

**Production base:** `ayo/main@126b53a` (`v1.6.1`)

## Summary

Argus currently maps every camera independently and exposes the result under
Configure/Rules. This is useful for a small site, but it does not scale to a
deployment with 100 cameras, does not represent several cameras watching one
physical place, and does not bring context review into the onboarding journey.

This design introduces a hierarchy:

```text
Site
  -> Area
      -> Camera view
```

Every camera still performs independent visual inference. Grouping never
replaces looking at the camera. Camera observations are combined
deterministically to propose shared area and site context, while conflicts stay
visible for human review. After onboarding, Argus opens a scene-review
workspace automatically. Users can confirm, correct, reject/remap, or keep a
camera paused. Approved cameras receive full context-aware monitoring; cameras
still awaiting review retain only the always-on critical safety baseline.

## Problem Statement

The current production behavior has four scaling gaps:

1. One hundred cameras can require one hundred sequential VLM calls during
   startup.
2. Cameras watching the same physical place can receive inconsistent labels.
3. The user must discover Scene context under Rules and review cameras one by
   one.
4. `environment_type` currently mixes site identity, physical area, and camera
   view into one value.

The desired product must infer that a site is a manufacturing plant,
supermarket, bank, warehouse, office, estate, or similar environment while
also understanding that individual cameras may watch a loading bay, checkout,
parking lot, production line, or reception.

## Goals

1. Preserve independent visual inference for every camera.
2. Infer and represent site type, area type, and camera-specific view
   separately.
3. Let operators group cameras that watch the same physical area.
4. Use grouping as a prior and aggregation boundary, never as a forced label.
5. Make scene review the immediate next step after onboarding and after adding
   a camera later.
6. Support 100-camera sites without blocking all monitoring on a serialized
   first-run mapping job.
7. Preserve existing reviewed contexts, site files, rule compatibility, and
   artifact paths.
8. Keep suggested zone geometry camera-specific.

## Non-Goals

- Automatic person re-identification across cameras.
- Automatic camera grouping without user confirmation.
- Building a 3D model of the site.
- Copying zone coordinates between camera angles.
- Letting Agent Mapper choose customer threat policy.
- Treating VLM confidence as calibrated probability.
- Retraining the visual model in this workstream.

## Product Principles

```text
Every camera is observed independently.
Areas share meaning, not pixels.
Site and area labels are proposals until reviewed.
Operator knowledge is a prior, not an instruction to hallucinate agreement.
Conflicts become review tasks, not hidden averages.
Critical safety monitoring never waits for semantic review.
```

## Alternatives Considered

### Camera-only mapping

Keep the current model and improve the individual review panel. This has the
smallest code change but produces repeated work, contradictory labels, and an
unusable 100-camera review experience. Rejected.

### One site-wide VLM call

Create a montage of representative frames and ask the VLM to classify the
whole site. This is brittle: 100 images exceed practical visual context,
details disappear, and one dominant scene can hide other areas. Rejected.

### Hierarchical independent mapping

Map each camera, aggregate observations within operator-confirmed areas, then
derive site context from informative area evidence. This retains visual
grounding, supports conflicts, and scales review. Selected.

## Domain Model

### Site Context

One site-level artifact describes the deployment as a whole:

```json
{
  "site_id": "deluxe_paints_lagos",
  "site_type": "manufacturing_plant",
  "site_description": "Paint manufacturing and storage facility.",
  "confidence": 0.88,
  "evidence_area_ids": ["production", "warehouse", "loading_bay"],
  "generated_at": "2026-09-01T12:00:00Z"
}
```

Initial `site_type` vocabulary:

```text
manufacturing_plant
supermarket
retail_store
bank
warehouse
office_building
residential_estate
shopping_mall
school
hospital
hotel
transport_hub
mixed_use
unknown
```

This vocabulary describes the business/site, not what every camera sees.

### Area Context

An area is a real physical place that one or more cameras watch:

```json
{
  "area_id": "production_floor",
  "name": "Production floor",
  "site_type": "manufacturing_plant",
  "area_type": "production_floor",
  "area_description": "Paint mixing lines and worker walkways.",
  "expected_actors": ["production staff", "maintenance staff"],
  "confidence": 0.91,
  "evidence_camera_ids": ["mixing_wide", "mixing_exit", "line_two"],
  "conflicts": [],
  "generated_at": "2026-09-01T12:00:00Z"
}
```

The area vocabulary expands the existing camera environment vocabulary with
real deployment types such as:

```text
production_floor
assembly_line
loading_bay
storage_aisle
chemical_store
machine_room
reception
retail_floor
checkout
banking_hall
vault_approach
office_floor
parking_lot
perimeter
entrance
walkway
unknown
```

The final enum will preserve all existing `environment_type` values as aliases
or valid area types so current rules do not break.

### Camera Context

The existing canonical `SceneContext` remains the camera-level artifact and
retains `environment_type` for compatibility. It gains optional hierarchy
references and an explicit view description:

```json
{
  "camera_id": "mixing_exit",
  "area_id": "production_floor",
  "site_type_candidate": "manufacturing_plant",
  "area_type_candidate": "production_floor",
  "environment_type": "production_floor",
  "scene_description": "Side view of the mixing line and emergency exit.",
  "view_description": "Covers the west walkway and exit door.",
  "expected_actors": ["production staff"],
  "zones": [],
  "confidence": 0.87
}
```

`environment_type` remains the compatibility field consumed by current rules.
New code prefers the reviewed area type when available.

### Site Configuration

Camera membership has one source of truth: `camera.area_id`. Areas do not store
a second mutable list of camera IDs.

```json
{
  "site_id": "deluxe_paints_lagos",
  "site_type_hint": "manufacturing_plant",
  "areas": [
    {"id": "production_floor", "name": "Production floor"},
    {"id": "loading_bay", "name": "Loading bay"}
  ],
  "cameras": [
    {"id": "mixing_wide", "area_id": "production_floor", "source": "rtsp://..."},
    {"id": "dock_gate", "area_id": "loading_bay", "source": "rtsp://..."}
  ]
}
```

Operator hints are optional. They are included in the mapper prompt as priors,
and disagreements are retained as explicit conflicts.

## Artifact Layout

Existing per-camera paths remain valid:

```text
<output_dir>/context/
  _site/
    site_context.json
    mapping_status.json
  _areas/<safe_area_id>/
    area_context.json
    mapping_status.json
  <safe_camera_id>/
    scene_context.json
    mapping_status.json
    source_frame.jpg
```

The reserved `_site` and `_areas` namespaces avoid moving existing camera
artifacts. All writes remain atomic and credential-redacted.

## Inference Flow

### Camera Observation

Each camera independently samples representative frames through the existing
Agent Mapper. The prompt receives optional site and area hints but must return
what the imagery supports. The result includes site and area candidates,
camera-view context, confidence, and suggested camera-specific zones.

An ambiguous image returns `unknown` or low confidence. It must not copy the
operator hint merely to appear consistent.

### Area Aggregation

`SceneContextAggregator` combines camera observations for one `area_id` using
a deterministic compatibility matrix and transparent evidence counts.

- Agreement among informative camera observations proposes an area type.
- Generic views such as parking lots and corridors carry little site-type
  evidence.
- Conflicting high-confidence observations create a review conflict.
- Numeric VLM confidence weights evidence but is never treated as calibrated.
- Operator-confirmed area context outranks all future automatic proposals until
  the operator explicitly requests remapping.

No extra LLM call is required for aggregation.

### Site Aggregation

Site type is derived from informative reviewed/proposed areas. For example,
production floor plus assembly line plus loading bay supports
`manufacturing_plant`; checkout plus retail floor supports `supermarket` or
`retail_store`. Ambiguous combinations produce `mixed_use` or `unknown`.

The inferred site type is shown to the user for confirmation. It does not
silently rewrite the site configuration.

## Mapping Coordinator and Scale

The current synchronous startup preflight is unsuitable for 100 fresh cameras.
Introduce a persistent `SceneMappingCoordinator` owned by the running engine:

1. Load reviewed camera/area/site contexts immediately.
2. Queue only new, stale, failed, or explicitly remapped cameras.
3. Run a bounded number of mapper workers; default one local Ollama request at
   a time, configurable up to the measured hardware limit.
4. Persist job state so an interrupted onboarding resumes.
5. Recompute only the affected area and site proposal when a camera finishes.
6. Publish progress and conflicts through health and console APIs.

For an unreviewed camera under `require_reviewed`:

- always-on critical baseline detectors remain active;
- context-dependent customer rules remain paused;
- the camera is visibly marked `limited monitoring`;
- approval injects context and enables the full rule pack without restarting
  unrelated cameras.

This replaces the current all-or-nothing interpretation of strict review with
a safer degraded mode. Fire, weapon, serious violence, person down, and camera
tampering do not wait for semantic mapping.

## Onboarding and Review UX

### Camera and Area Setup

The camera onboarding step allows the installer to:

- create or select an area;
- assign several cameras to the same area;
- provide optional site/area/actor hints;
- leave cameras ungrouped when the physical relationship is unknown.

Ungrouped cameras remain valid and receive temporary individual areas.

### Automatic Scene Review

After setup, Argus starts the mapper and opens a full Scene Review workspace.
For a small site it feels like a modal; for a large site it has an area sidebar
and progress counts:

```text
Mapped 24 / 100    Reviewed 18    Conflicts 2    Failed 1
```

Each area shows:

- inferred site and area type;
- all camera thumbnails used as evidence;
- camera-specific descriptions and confidence;
- conflicts between cameras or operator hints;
- camera-specific suggested zones.

Actions are explicit:

- **Confirm area:** approve shared site/area meaning after viewing all camera
  evidence; non-conflicting camera contexts in that area become reviewed.
- **Edit and confirm:** overwrite site/area text or individual camera context,
  then approve.
- **Reject and remap:** mark selected cameras stale and enqueue new mapping.
- **Keep paused:** close review while context-dependent rules remain disabled
  for those cameras.
- **Accept/ignore zone:** remains per camera because coordinates cannot be
  transferred between angles.

The workspace also opens automatically when cameras are added later. It never
creates 100 separate popups.

## Runtime Context Composition

Rules and TrueSight receive one rendered context assembled in precedence order:

```text
reviewed site context
  + reviewed area context
  + reviewed camera context
  + accepted camera-specific zone roles
```

Camera-specific facts override broad area wording. Human-authored and reviewed
values outrank model proposals. Existing `environment_type` compatibility
continues to work while new rules may target `site_type` and `area_type`.

## API Changes

Add Qt/backend methods that return plain JSON contracts:

- `sceneReviewSummary()`
- `listAreas()`
- `createArea(name, hints)`
- `assignCameraArea(camera_id, area_id)`
- `areaContext(area_id)`
- `approveAreaContext(area_id, payload)`
- `updateSiteContext(payload)`
- `approveSiteContext(payload)`
- `enqueueSceneMapping(camera_ids)`
- `sceneMappingProgress()`

Existing camera-level review APIs remain supported.

Mutations require `CONFIGURE_CAMERAS` or `CONFIGURE_SITE` as appropriate and
write audit entries. Operators may view status but cannot alter context.

## Migration and Compatibility

1. Existing site files without `areas` continue to load.
2. Each legacy camera is assigned a stable implicit area only in memory until
   the operator groups it; the site file is not rewritten merely by opening it.
3. Existing reviewed `scene_context.json` artifacts remain authoritative and
   seed area/site proposals.
4. Existing `environment_type` values remain valid.
5. Existing camera-only rule configs behave unchanged.
6. Missing hierarchy means no additional suppression beyond current behavior.
7. Migration is additive and reversible; old releases can ignore unknown
   top-level site fields only after their loaders are verified to tolerate them.

## Failure Handling

- **Ollama unavailable:** queued jobs remain retryable; critical baseline runs;
  UI shows the failed camera and corrective action.
- **Camera unavailable:** preserve prior reviewed context as stale, do not erase
  it, and show that fresh visual confirmation is unavailable.
- **Camera disagreement:** do not majority-vote it away; show a conflict.
- **Moved camera/source fingerprint change:** invalidate only that camera, then
  recompute its area proposal.
- **Area reassignment:** keep the camera artifact but mark area/site proposals
  stale; never copy zones into the new area.
- **Low confidence:** require individual review; do not include it in bulk area
  approval by default.
- **Duplicate-looking areas:** visual similarity can suggest grouping later but
  never changes `area_id` automatically.

## Security and Privacy

- All inference and artifacts remain local by default.
- RTSP credentials remain redacted from hashes, logs, health, and UI.
- Expected actors describe roles that may appear, not personal identity.
- No face recognition or cross-camera identity tracking is introduced.
- Site/area/context edits and approvals remain role-gated and audited.

## Testing Strategy

### Unit Tests

- site, area, and camera schema validation;
- vocabulary alias migration;
- deterministic area/site aggregation;
- conflict and low-confidence behavior;
- precedence of reviewed human context;
- queue persistence, deduplication, and source invalidation;
- critical-baseline-only behavior before review.

### Integration Tests

- legacy site opens without mutation;
- 100 synthetic cameras queue without 100 simultaneous Ollama calls;
- three cameras in one area produce one area review task;
- conflicting camera observations prevent bulk approval;
- approval enables full rules without restarting unrelated cameras;
- failed mapping does not disable critical safety monitoring;
- accepted zones remain camera-specific.

### UI Tests

- onboarding creates/selects areas and assigns cameras;
- Scene Review opens automatically after setup;
- pending, ready, conflict, failed, and paused states render;
- confirm, edit-and-confirm, reject/remap, and keep-paused actions call the
  correct backend boundaries;
- a 100-camera fixture renders by area without one modal per camera;
- mutation controls remain hidden from unauthorized roles.

### Manual Acceptance

- small retail site with one camera;
- manufacturing site with production, warehouse, loading, reception, and
  parking footage;
- three cameras watching one area from different angles;
- mixed-use or ambiguous site;
- interrupted local Ollama mapping resumed after restart;
- genuine parking footage proving retail-only rules remain suppressed while
  critical baselines continue.

## Delivery Sequence

1. Add additive schemas, vocabularies, migration helpers, and aggregation tests.
2. Add area assignment to site configuration and onboarding.
3. Add the persistent bounded mapping coordinator and limited-monitoring mode.
4. Add site/area inference and composed runtime context.
5. Add the automatic Scene Review workspace and bulk/conflict workflows.
6. Run 100-camera synthetic load tests and real multi-angle acceptance footage.
7. Measure mapper latency, review workload, and context-confusion reduction
   before enabling strict review as the default for large deployments.

## Success Criteria

- Every new camera is visually mapped independently.
- One hundred cameras do not produce one hundred simultaneous VLM requests or
  block critical safety monitoring.
- Cameras can share reviewed area context while retaining independent view and
  zone context.
- The system proposes manufacturing plant, supermarket, bank, warehouse, and
  other site types from visual evidence and makes uncertainty visible.
- The post-onboarding Scene Review appears without the user discovering it in
  Rules.
- A user can confirm, overwrite, reject/remap, or keep cameras paused.
- Existing sites and reviewed contexts continue operating without manual
  migration.
