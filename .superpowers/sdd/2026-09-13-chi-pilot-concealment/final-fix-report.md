# Chi Pilot Scenario 10 Final Fix Report

## Status

The complete final review list was implemented on base `1f5e62d` and committed
as `0507de8` (`fix(concealment): close final pilot review gaps`). Focused tests
pass. The exact-head full Python suite has no failures in the changed behavior;
its sole remaining failure is the supplied shared environment's known
Ultralytics version mismatch.

Real Chi footage and the frozen prompt corpus are unavailable. No real-clip
acceptance, candidate recall, TrueSight precision/recall, or prompt measurement
is claimed in this report.

## Implemented Fixes

### Exclusive Bag Ownership In Every Runtime

- Added `ConcealmentDetector.update_with_bag_detections()` as the production
  entrypoint for stateful, exclusive physical-bag assignment.
- Migrated `cvti/pipelines/retail_pipeline.py`, `cvti/app/worker.py`,
  `cvti/detector/core.py`, `cvti/serving/camera.py`, and the standalone
  concealment demo away from global bag fan-out.
- Preserved `ConcealmentDetector.update(..., bag_bboxes=...)` for legacy
  standalone compatibility, as required by the binding design.
- Added production-adapter integration tests for the retail CLI, desktop
  worker, and detector core. The existing serving integration test covers
  `PerCameraState`.

### Temporal Physical-Bag Ownership

- Tracks a detected physical bag across frames using overlap and normalized
  center distance.
- Keeps its existing owner while that owner remains spatially plausible, so
  near-equal nearest-person jitter cannot alternate bag evidence between
  temporal track buffers.
- If the owner disappears, the bag remains temporarily unassigned until the
  former owner's evidence has aged out of the scoring window. It can then
  transfer to a new owner.
- Expires stale physical-bag identity before matching a reappearing detection.

This is the closest safe behavior to the spec's exclusive per-track ownership:
briefly withholding ambiguous evidence is preferable to leaving persistent
evidence on two people. Thresholds and four-sample candidate persistence were
not changed.

### Pre-Admission Candidate Audit

- Replaced verdict-time `concealment_audit.jsonl` output with a
  `concealment_audit` table in the existing `events.db`.
- Inserts every generated concealment candidate before `AlertQueue` makes its
  admission decision.
- Records `admitted`, `deduplicated`, and `capacity_dropped` explicitly.
- When capacity displaces an earlier admitted row, updates that earlier row to
  `capacity_dropped` while preserving its original admission time.
- Attaches confirmed, rejected, or unverified gate data to the same row later.
- Keeps audit failures best-effort so storage trouble cannot stop detection or
  gate delivery.
- Stores malformed/cyclic nested metadata with safe fallback JSON and an
  `audit_error`, preserving the candidate row instead of losing it.

The pilot extractor now calculates candidate recall from all generated rows,
not only rows that reached a verdict. It reports admitted, deduplicated, and
capacity-dropped counts separately.

### Retention And Emergency Purge

- Normal retention deletes `concealment_audit` rows older than the configured
  site retention period.
- Dry-run output reports audit rows that would be deleted.
- Emergency disk purge deletes oldest audit rows before considering deletion
  of held-aware event evidence.
- Missing/older databases without the table remain tolerated.

Audit rows are operational candidate measurements rather than incident
evidence and do not receive an independent legal-hold state. Confirmed event
evidence retains the existing legal-hold protections.

### Minor Review Items

- Deep-copied nested assessment components and reasons into `RawEvent`, then
  deep-copied event metadata and reasons into `CandidateAlert`.
- Added mutation tests at both boundaries.
- Corrected the local frame-cap comment to two frames normally and four for
  concealment.
- Added concurrent SQLite audit-write coverage.
- Added cyclic metadata serialization-failure coverage.
- Moved the lazy performance-board import before the `pose_infer` timer starts.
- Added equivalent production concealment candidate coverage at pose strides
  1 and 2.

## TDD Evidence

### Initial RED

Command:

```bash
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" -m pytest tests/test_concealment.py tests/test_concealment_entrypoints.py tests/test_alert_sink.py tests/test_retention.py tests/test_gate_evidence_quality.py tests/test_heavy_models_earn_their_frames.py -q
```

Result: `16 failed, 75 passed, 14 warnings in 4.61s`.

Expected failures named the missing stateful ownership API and runtime
adapters, absent SQLite audit table and queue callbacks, missing retention and
emergency purge behavior, shared nested metadata, and lazy-import time inside
`pose_infer`.

### Initial GREEN

Same command after the minimal implementations:

`91 passed, 14 warnings in 4.00s`.

### Ownership Expiry RED/GREEN

The new direct-reappearance test initially failed because a stale physical bag
could resurrect its prior owner after the scoring evidence had expired:

```text
FAILED test_physical_bag_can_transfer_after_prior_evidence_ages_out
assert 1.0 == 0.0
```

After expiring ownership before matching, the same test passed:

`1 passed, 14 warnings in 1.02s`.

### Nested Reasons RED/GREEN

The strengthened mutation test initially showed that `CandidateAlert.reasons`
still shared a nested list with `RawEvent.extra`:

```text
FAILED test_nested_assessment_metadata_is_detached_at_each_boundary
event-mutated remained visible in alert.reasons
```

After deep-copying alert reasons, the same test passed:

`1 passed, 14 warnings in 1.05s`.

### Final Focused GREEN

Command:

```bash
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" -m pytest tests/test_concealment.py tests/test_concealment_entrypoints.py tests/test_heavy_models_earn_their_frames.py tests/test_serving.py tests/test_alert_sink.py tests/test_retention.py tests/test_gate_evidence_quality.py tests/test_prompt_regression.py tests/test_ram_bounds.py tests/test_value_surface.py tests/test_two_tier.py tests/test_verify_bypass_tier.py tests/test_async_enrichment.py -q
```

Result: `187 passed, 14 warnings in 8.74s`.

## Full Python Suite

Exact command:

```bash
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" -m pytest
```

The managed sandbox run completed with `51 failed, 1358 passed, 8 skipped, 16
warnings, 8 errors in 129.53s`. The failures were dominated by denied localhost
socket binds and denied writes under `~/Library/Application Support/Argus`; the
same environment-only pattern was recorded before this fix wave.

The identical command was rerun with the required local permissions. Result:

`1 failed, 1416 passed, 8 skipped, 15 warnings in 158.19s`.

The sole failure was:

```text
tests/test_detection_rides_the_accelerator.py::SeamPins::
test_ultralytics_is_pinned_and_the_seam_looks_as_assumed
```

The supplied interpreter imports `ultralytics==8.4.64`, while
`requirements.txt` pins `ultralytics==8.4.35`. Dependency pins and the shared
environment were not changed, per the task instruction.

## Additional Verification

- Prompt fingerprint check: exit `0`; fingerprint `a2e093837f82...` is
  recorded and correctly reported as `UNMEASURED` because the frozen corpus is
  unavailable.
- Python compile check for all changed Python modules/tests: exit `0` with
  `PYTHONPYCACHEPREFIX=/private/tmp/chi-pilot-pyc`.
- `git diff --check`: exit `0` before commit.
- `git show --check --stat --oneline 0507de8`: exit `0`.

## Self-Review

- Re-read the binding design, implementation plan, progress ledger, and full
  `cc58839..1f5e62d` review patch before editing.
- Confirmed all named runtime entrypoints use the stateful exclusive API and
  no production call still passes one global bag list.
- Confirmed legacy global input remains available only as a compatibility
  interface.
- Confirmed candidate audit insertion occurs before queue admission and that
  capacity eviction updates the displaced candidate, not only the newcomer.
- Confirmed queue callbacks execute outside the queue lock and sink writes are
  serialized under the sink lock.
- Confirmed audit serialization and SQLite failures remain fail-open for the
  detection path.
- Confirmed retention handles missing audit tables and does not weaken event
  legal holds.
- Confirmed no detector thresholds, dependency pins, prompt text, or unrelated
  runtime behavior changed.
- Reviewed the immutable `0507de8` file list; all 19 files are in scope.

## Remaining Concerns

1. Real Chi clips are unavailable, so the seven-case acceptance matrix has not
   been executed and scenario 10 is not empirically accepted.
2. The frozen prompt corpus is unavailable, so current prompt precision and
   recall remain unmeasured.
3. The shared project interpreter does not match the repository's pinned
   Ultralytics version, leaving one unrelated full-suite failure.
4. Physical-bag identity is geometry-based because the shared detector exposes
   boxes, not persistent bag track IDs. The hysteresis is covered for near-equal
   shoppers, disappearance, and transfer, but real Chi footage is
   still needed to characterize detector-box instability.
