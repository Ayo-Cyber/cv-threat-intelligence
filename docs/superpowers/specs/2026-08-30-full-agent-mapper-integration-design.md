# Full Agent Mapper Integration Design

**Date:** 2026-08-30

**Status:** Proposed for founder review

**Production base:** `ayo/main@b32dbec`

## Summary

Argus currently starts a lightweight background scene mapper after detection is
already running. It samples one fixed frame, asks the local VLM for two fields,
and writes an incomplete context document. This provides some grounding, but it
does not use the full `AgentMapper`, does not satisfy the canonical schema, and
allows early alerts to be verified with generic context.

This design replaces the duplicate mapper with the full schema-validated Agent
Mapper and makes scene context a first-class, site-scoped artifact. New cameras
must acquire context before their threat pipeline is enabled. Later starts load
validated cached context immediately. A deterministic compatibility check then
prevents contextually nonsensical built-in rules, such as shoplifting in a car
park, from reaching TrueSight unless the customer explicitly overrides the
default.

The Agent Mapper remains descriptive. It does not select threat rules, activate
zones, or make alert decisions.

## Problem Statement

During a founder test, Argus reportedly described an event as shoplifting in a
car park. This is not merely an inaccurate visual verdict; it is a mismatch
between the threat category and the environment.

Today the production path has four contributing weaknesses:

1. `cvti/serving/scene_map.py` uses a separate prompt and parser instead of
   `cvti.scene.agent_mapper.AgentMapper`.
2. It returns only `environment_type` and `scene_description`; the result does
   not satisfy `schemas/scene_context.schema.json`.
3. Mapping starts after `MultiStreamPipeline.start()`, so early candidates use
   generic or static context.
4. The gate receives scene text, but there is no deterministic check that a
   built-in rule is semantically applicable to the mapped environment.

Adding more prompt text alone will not solve this. The upstream mapper is also
a VLM and can hallucinate. Its output must be bounded, validated, reviewable,
and treated as evidence rather than authority.

## Goals

1. Use the full Agent Mapper in the production serving path.
2. Ensure every enabled camera has explicit or validated cached context before
   candidate verification begins.
3. Persist complete scene context and mapping state under the site's output
   directory so the engine and console share one artifact contract.
4. Let operators review, correct, approve, or remap scene context in the current
   Argus web console.
5. Prevent incompatible built-in rule categories from reaching TrueSight by
   default while preserving explicit customer overrides.
6. Expose mapper readiness and failure states through existing health and UI
   surfaces.
7. Preserve compatibility with existing site files and demo workflows.

## Non-Goals

- The mapper will not decide which threats a customer should monitor.
- The mapper will not activate suggested zones automatically.
- The mapper will not continuously inspect every frame.
- The mapper will not rename a rejected candidate into a different threat.
- The first implementation will not train or fine-tune a scene model.
- The first implementation will not infer employee identity or sensitive
  demographic attributes.
- The compatibility layer will not block always-on critical safety rules such
  as fire, visible weapons, serious violence, person down, or camera tampering.

## Product Principles

```text
Agent Mapper describes the place.
The customer defines threat policy.
Detectors propose events.
Context compatibility rejects category mismatches.
TrueSight verifies applicable candidates.
The operator remains the final authority.
```

Context reduces hallucination; it does not guarantee truth. Human-authored
context and explicit customer choices therefore outrank mapper output.

## Current Production Behavior

At `ayo/main@b32dbec`, `run_site()` starts the detection pipeline and then calls
`map_cameras_async()` only for `ollama` or `local` gates. The mapper:

- opens each source separately;
- samples one frame at 15% of a local video's duration;
- asks for `environment_type` and `scene_description`;
- maps cameras sequentially in one daemon thread;
- mutates `PerCameraState.scene_context` after startup;
- writes `runs/context/<camera_id>/scene_context.json` relative to the process
  working directory.

The full `AgentMapper.map()` already supports multiple samples,
representative-frame selection, strict prompt construction, bounded vocabulary,
schema-shaped output, and local Ollama. It is not used by this live path.

## Proposed Architecture

```text
Camera onboarding / engine preflight
  -> SceneContextStore resolves explicit override or cached context
  -> cache missing/stale: FullAgentMapperService maps representative frame
  -> schema validation
  -> atomic site-scoped persistence
  -> operator review/approval where policy requires it
  -> PerCameraState receives validated context
  -> detectors and CustomizationEngine run
  -> ContextCompatibility evaluates each matched rule
  -> applicable candidates enter AlertQueue
  -> TrueSight receives candidate + complete scene context
  -> decision/evidence/compatibility reason remains inspectable
```

## Component Design

### 1. Canonical Scene Context

`schemas/scene_context.schema.json` remains the canonical descriptive contract.
The runtime context contains:

- `camera_id`
- `source_type`
- `environment_type`
- `scene_description`
- `expected_actors`
- suggested `zones`
- mapper `confidence`
- `generated_at`
- `source_frame_path`
- optional `notes`

The allowed environment vocabulary will be expanded only when a real supported
deployment type is missing. The lightweight values `retail`, `warehouse`, and
`parking` must not form a parallel vocabulary; they normalize to canonical
values such as `retail_shop`, `warehouse_floor`, and `parking_lot`.

Lifecycle metadata does not belong inside this descriptive schema. It is stored
in a separate status sidecar.

### 2. SceneContextStore

Add a focused `cvti/scene/context_store.py` component responsible for loading,
validating, writing, invalidating, and resolving context.

Site-scoped layout:

```text
<output_dir>/context/<safe_camera_id>/
  scene_context.json
  mapping_status.json
  source_frame.jpg
  raw_response.txt          # optional, debug only
```

`mapping_status.json` contains lifecycle metadata:

```json
{
  "status": "ready_reviewed",
  "source_fingerprint": "sha256:...",
  "mapped_at": "2026-08-30T15:00:00Z",
  "reviewed_at": "2026-08-30T15:02:00Z",
  "reviewed_by": "owner",
  "error": ""
}
```

Allowed statuses:

- `pending`
- `ready_unreviewed`
- `ready_reviewed`
- `stale`
- `failed`

Writes use a temporary sibling followed by `Path.replace()` so the console
never reads a partially written JSON document. Camera IDs are sanitized before
becoming path components.

The source fingerprint is derived from a credential-redacted source identity.
For files it also includes resolved path, size, and modification time. For RTSP
it includes host/path identity but never username or password. A changed
fingerprint makes cached context stale.

### 3. Context Precedence

Context resolution follows this order:

1. **Human-authored context in site configuration.** Existing
   `environment_type` and `scene_description` fields remain supported. New
   `expected_actors` is optional. Explicit fields count as reviewed.
2. **Validated reviewed cache.** Used immediately when the fingerprint matches.
3. **Validated unreviewed cache.** Allowed only under legacy/automatic policy
   and displayed with a review warning.
4. **Fresh full mapping.** Required when no usable context exists.
5. **Failure state.** Never silently represented as trusted generic context.

The store returns both the context and its provenance (`manual`, `cache`, or
`mapper`) so health, logs, and UI can explain what Argus is using.

### 4. FullAgentMapperService

Refactor `cvti/serving/scene_map.py` into a serving adapter over
`cvti.scene.agent_mapper.AgentMapper`. The adapter coordinates site paths,
status updates, and cancellation; it does not define another VLM prompt or
parser.

For each camera it:

1. Marks mapping `pending`.
2. Calls `AgentMapper.map(source, camera_id, sample_count=3)`.
3. Validates the returned context again at the storage boundary.
4. Saves the selected representative frame and context artifacts.
5. Marks the context `ready_unreviewed` or `failed`.
6. Returns a structured result rather than mutating global state implicitly.

`AgentMapper` will expose an artifact-bearing result or a `map_and_save()` seam
so the service can persist the exact selected frame without reopening and
resampling the source. There must be one sampling path, one prompt, and one
parser.

Mapping uses the configured mapper model, defaulting to the local gate model so
no second model occupies memory. Mapper calls are serialized by default because
Ollama saturation can delay both mapping and verification. During first-time
onboarding this cost is acceptable and visible.

### 5. Startup and Onboarding Lifecycle

New site configurations written by the setup wizard include:

```json
{
  "scene_context_policy": "require_reviewed"
}
```

Supported policies:

- `require_reviewed`: mapping and operator approval are required before the
  camera's threat pipeline is enabled.
- `auto`: mapping must complete, but validated unreviewed context may enable the
  camera with a visible warning. This is the compatibility mode for legacy demo
  and command-line site files.
- `manual`: the mapper is disabled and explicit context is required.

For backward compatibility, site files without the field behave as `auto`.
They do not regain the old behavior of running with generic context: missing
context is mapped synchronously before that camera is enabled.

Mapping readiness is per camera. Existing ready cameras continue monitoring
while a newly added camera is pending. The engine does not need to hold the
entire site because one camera lacks context.

At engine startup:

1. Build context stores and resolve all camera contexts before starting their
   detector states.
2. Map missing/stale `auto` cameras during preflight.
3. Exclude unresolved `require_reviewed` or `manual` cameras from active
   sources and report why.
4. Pass resolved context into `build_camera_states()`.
5. Remove the post-`pipe.start()` generic-context mapping race.

### 6. Operator Review in the Current Console

The shipped console is the Qt WebEngine shell using `cvti/app/web/index.html`.
The legacy `MapperPanel` widget is not the production integration target.

The Rules or camera setup surface will show:

- mapping status and provenance;
- representative source frame;
- environment type selector;
- editable scene description;
- editable expected-actor list;
- mapper confidence as supporting information, not a quality guarantee;
- suggested zone rectangles with Accept, Edit, and Ignore actions;
- Remap command;
- approval action.

Only users with camera-configuration permission may edit, approve, or remap.
Approval writes the corrected context atomically and records reviewer identity
and time in the sidecar. It does not alter the append-only security audit design;
the existing audit mechanism records the operator action.

Accepted suggested zones are converted from mapper `bbox` coordinates into the
existing polygon-zone format. They inherit no threat rule until the operator
chooses the zone behavior. Ignored suggestions remain inactive.

### 7. Deterministic Context Compatibility

Add `cvti/rules/context_compatibility.py`. It evaluates a matched candidate
against optional rule metadata before enqueueing.

Rules may declare:

```json
{
  "context_requirements": {
    "environment_types": ["retail_shop", "mall_corridor"],
    "zone_roles_any": ["merchandise", "checkout"],
    "mode": "enforce"
  }
}
```

Semantics:

- No `context_requirements`: allow, preserving compatibility.
- `mode: enforce`: at least one declared environment or zone requirement must
  match as specified by the rule.
- `mode: override`: allow despite mismatch and record that the customer
  explicitly overrode the default.
- Unknown/failed context does not count as a positive match.
- Always-on critical baseline rules bypass contextual restrictions.
- Customer-created plain-English threats are explicit customer policy and use
  override semantics unless the customer chooses a restriction.

The initial built-in metadata is deliberately narrow. Shoplifting/concealment
requires a retail environment or an accepted merchandise/checkout zone.
Loitering, after-hours presence, perimeter intrusion, violence, fire, weapons,
person-down, and tampering are not globally restricted merely because they can
occur in many environments.

An incompatible candidate is suppressed with a structured reason:

```text
rule=shoplifting
environment=parking_lot
decision=context_incompatible
reason=requires retail environment or merchandise/checkout zone
```

Argus does not rename it to vehicle theft or robbery. Other detectors may raise
those independent candidates if evidence supports them.

Compatibility suppression increments a health counter and is available to logs
and the operator's diagnostic surface. This prevents a silent policy layer from
looking like detector failure.

### 8. Downstream Consumers

The resolved canonical context is passed to:

- `PerCameraState` and `CustomizationEngine`;
- context compatibility;
- TrueSight verification prompts;
- the plain-English custom-rule scanner;
- subject watches;
- evidence metadata and diagnostics.

Only fields relevant to a consumer should be rendered into its prompt. Expected
actors are presented as normal possibilities, not guaranteed identities. For
example: “Expected actors may include customers and staff,” not “this person is
a staff member.”

### 9. Health and Failure Behavior

The existing health document gains a per-camera mapping component:

```json
{
  "scene_mapping": {
    "camera_id": "parking_1",
    "status": "ready_reviewed",
    "provenance": "cache",
    "environment_type": "parking_lot",
    "last_mapped_at": "2026-08-30T15:00:00Z",
    "error": ""
  }
}
```

Failure behavior:

- Sampling failure: mark `failed`, preserve previous valid context if its source
  fingerprint still matches, otherwise keep that camera disabled.
- VLM transport/parse failure: mark `failed`; never save raw prose as a scene
  description.
- Schema failure: mark `failed`, retain raw response only when debugging is
  enabled, and expose a corrective message.
- Stale cache: do not present it as current. Remap or require manual approval.
- Ollama saturation: mapper retries with bounded backoff and publishes pending
  status; it does not compete indefinitely with active gate work.

### 10. Security and Privacy

- Context and representative frames remain site-local.
- RTSP credentials never appear in status files, fingerprints, logs, or UI.
- Context artifact directories inherit the output directory's restrictive
  permissions.
- Raw VLM responses are off by default in packaged deployments.
- Mapping/remap/approval requires existing configuration permissions and is
  recorded in the audit log.
- The context store rejects path traversal through camera IDs.

## Data Flow Examples

### New Camera, Reviewed Policy

```text
operator adds parking camera
  -> status pending
  -> mapper samples 3 frames and selects representative frame
  -> schema validates parking_lot + actors + proposed zones
  -> status ready_unreviewed
  -> operator corrects description and accepts entry zone
  -> status ready_reviewed
  -> camera threat pipeline starts with canonical context
```

### Existing Camera Restart

```text
engine starts
  -> source fingerprint matches reviewed cache
  -> context loads before PerCameraState construction
  -> camera starts immediately
  -> no mapper VLM call
```

### Contextually Incompatible Candidate

```text
detector emits theft-like candidate
  -> customer rule resolves to built-in shoplifting
  -> scene is parking_lot; no merchandise/checkout zone
  -> compatibility decision suppresses shoplifting
  -> reason appears in diagnostics
  -> independent violence/robbery/vehicle-theft candidates remain possible
```

### Explicit Customer Override

```text
customer deliberately enables shoplifting for parking_1
  -> rule stores context mode=override
  -> compatibility allows candidate and records override
  -> TrueSight sees parking-lot context and the customer's exact question
```

## Migration and Backward Compatibility

1. Existing site files without `scene_context_policy` use `auto`.
2. Existing static `environment_type` and `scene_description` are treated as
   reviewed manual context.
3. Existing two-field files under repository-relative `runs/context` may be
   imported only as unreviewed legacy data after normalization and schema
   completion; they are never treated as reviewed canonical context.
4. New writes go only to `<output_dir>/context`.
5. Existing rules without context metadata remain allowed.
6. Existing critical baseline rules are behaviorally unchanged.

Migration is additive. No existing context, site, or rule file is deleted
automatically.

## Testing Strategy

### Unit tests

- Full context schema validation and canonical environment normalization.
- Store atomic write/read behavior.
- Camera-ID path traversal rejection.
- Credential-redacted source fingerprinting.
- Cache precedence, source-change invalidation, and legacy import.
- Mapper success, transport failure, malformed JSON, and schema failure.
- Compatibility allow, enforce, override, unknown-context, zone-role, and
  critical-baseline behavior.

### Integration tests

- `run_site()` resolves context before constructing active camera states.
- A missing `auto` context uses a fake mapper and starts only after success.
- A missing `require_reviewed` context remains disabled and is named in health.
- Cached context avoids a mapper call on restart.
- TrueSight receives the resolved canonical context.
- English rules and watches read the site-scoped context rather than CWD state.
- A parking-lot shoplifting candidate is suppressed unless explicitly
  overridden.

### Console tests

- Scene context/status is read from the database/output-directory parent.
- Only authorized roles can remap, edit, or approve.
- Suggested zones remain inactive until accepted.
- Approval records reviewer identity and an audit event.
- UI JavaScript remains valid under `node --check`.

### End-to-end verification

1. Run the complete test suite.
2. Start a local Ollama mapping test on one retail clip and one parking clip.
3. Inspect the exact representative frame and complete context artifacts.
4. Run the parking clip through a shoplifting-enabled built-in test config and
   confirm context incompatibility suppresses it.
5. Add an explicit customer override and confirm the candidate reaches
   TrueSight with parking context.
6. Restart and confirm no mapper call occurs while the source fingerprint is
   unchanged.

## Rollout

1. Land store, schema, and compatibility units behind no runtime behavior
   change.
2. Replace the lightweight serving mapper and preserve `auto` compatibility.
3. Add health and diagnostics.
4. Add current-console review/remap controls.
5. Enable `require_reviewed` for newly created sites.
6. Run held-out context-confusion tests before making hallucination-reduction
   claims.

## Success Criteria

- No camera begins candidate verification with silently generic context.
- Production uses one Agent Mapper prompt/parser/validation path.
- Every active camera has context with known provenance and readiness.
- A built-in shoplifting candidate in a mapped parking lot is suppressed unless
  explicitly overridden.
- Critical safety candidates remain unaffected by contextual restrictions.
- Suggested zones never become active without operator action.
- Restarting with unchanged reviewed context makes zero mapper VLM calls.
- Context failures are visible in health, logs, and the console.
- Full tests remain green.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Mapper itself hallucinates | Bounded schema, representative frames, confidence, review and manual override |
| First setup feels slow | Visible per-camera progress; pay cost once; cache later starts |
| Ollama saturation delays setup | Serialize mapping, bounded backoff, no repeated startup remapping |
| Compatibility blocks a valid unusual rule | Explicit customer override with recorded reason |
| Hard-coded taxonomy grows uncontrollably | Optional per-rule metadata; restrict only demonstrated category mismatches |
| Suggested zones are geometrically poor | Proposal-only UI; edit/accept required before activation |
| Packaged app writes to wrong directory | Site `output_dir` is injected into every context store; hostile-install tests |
| Old demos stop starting | Missing policy defaults to `auto`; existing static context counts as reviewed |

## Open Questions Resolved by This Design

- **Block or background on first map?** Per-camera gated preflight. Existing
  ready cameras continue; the new camera waits.
- **Map every restart?** No. Use fingerprinted validated cache.
- **Can the mapper activate zones?** No. Suggestions require human acceptance.
- **Can context suppress threats?** Only optional built-in rule compatibility;
  critical baselines bypass it and customer override is authoritative.
- **Can Argus rename shoplifting to another theft category?** No. Independent
  detectors must support the alternative candidate.
- **Where are artifacts stored?** Under the site's configured output directory,
  never repository-relative CWD state.
