# Task 2 Report: Camera Pipeline Motion Scenarios

## Implementation

- Added per-camera scenario 4 and 5 toggles plus the Task 1 motion thresholds, count,
  persistence, and optional permitted-zone list.
- Added camera-specific validation for malformed, non-positive, inverted, and below-minimum
  movement settings.
- Instantiated `PersonMotionTracker` and `SimultaneousMovementDetector` per camera when either
  movement scenario is enabled.
- Ran motion classification after ByteTrack and after the existing zone monitor has produced
  membership. Scenario 4 stores moving-track telemetry only for permitted zones and emits no
  `RawEvent`; an absent permitted-zone list treats the whole frame as permitted.
- Adapted one latched aggregate into `RawEvent(detector="multiple_people_moving")` with all
  track IDs, per-track motion measurements, confidence, people count, and union group box.
- Routed the event through the existing Customization Engine and `QueuedAlert` evidence payload.
  Group events use the aggregate box and exactly three chronologically selected full frames.
- Added the Chi pilot rule at configurable JSON priority and a TrueSight question explicitly
  about simultaneous visible movement, independent of crowd density, proximity, or panic.
- Added both scenario controls and conservative defaults to the backend, React detector catalog,
  and the legacy web detector catalog required by the existing parity contract.

## Files

- Modified `cvti/serving/camera.py`.
- Modified `cvti/event_adapters.py`.
- Modified `cvti/verification/frame_select.py`.
- Modified `cvti/verification/gate.py`.
- Modified `cvti/app/console_backend.py`.
- Modified `Frontend/src/lib/types.ts`.
- Modified `cvti/app/web/index.html` to preserve detector-catalog parity.
- Created `configs/chi_pilot_v1.json`.
- Modified `tests/test_detector_toggles.py`.
- Modified `tests/test_serving.py`.
- Modified `tests/test_gate_evidence_quality.py`.

## RED / GREEN Evidence

All commands used the required shared Python 3.9 virtual environment.

### Configuration And Zone RED

```text
<shared-python> -m pytest tests/test_detector_toggles.py tests/test_serving.py -q
4 failed, 30 passed, 14 warnings
```

The expected failures were missing movement constructor fields, missing configuration
validation, and absent zone-filtered overlays.

### Configuration And Zone GREEN

```text
<shared-python> -m pytest tests/test_detector_toggles.py tests/test_serving.py -q
34 passed, 14 warnings
```

### Scenario 5 And Gate RED

```text
<shared-python> -m pytest tests/test_serving.py tests/test_gate_evidence_quality.py -q
2 failed, 40 passed, 14 warnings
```

The camera emitted no candidate and the gate used its generic threat question.

### Scenario 5 And Gate GREEN

```text
<shared-python> -m pytest tests/test_serving.py tests/test_gate_evidence_quality.py -q
42 passed, 14 warnings
```

### Malformed Threshold RED / GREEN

The focused validation test first failed because `float("fast")` omitted the camera ID, then
passed after movement parsing was wrapped with a camera-specific `ValueError`.

## Final Verification

```text
<shared-python> -m pytest tests/test_person_motion.py tests/test_detector_toggles.py \
  tests/test_serving.py tests/test_gate_evidence_quality.py -q
61 passed, 14 warnings

<shared-python> -m pytest tests/test_retail_zones.py tests/test_zone_customization.py \
  tests/test_zone_rules_e2e.py -q
33 passed, 15 warnings
```

Python 3.9 bytecode compilation passed with `PYTHONPYCACHEPREFIX` directed to `/tmp`, and
`git diff --check` passed.

## Self-Review

- Re-read the task brief and design against the final diff without spawning a subagent.
- Confirmed motion consumes ByteTrack identities and the same zone-state snapshot used by the
  unchanged zone entry, presence, dwell, and exit adapters.
- Confirmed permitted zones filter both overlays and scenario 5 count, while an absent list
  permits the whole view.
- Confirmed scenario 4 has no event adapter or rule and therefore remains telemetry only.
- Confirmed scenario 5 is independent of `CrowdFormationDetector` and accepts spread-out movers.
- Confirmed the detector latch suppresses per-frame duplicates and resets in the Task 1 unit.
- Confirmed candidate metadata survives customization and queue mapping, and the aggregate box
  is retained instead of being replaced by one person's box.
- Confirmed the gate receives three full context frames and movement-specific wording.
- Confirmed all engine boolean flags remain reachable through both operator catalogs.
- Left the two deferred Task 1 hardening items untouched because neither is load-bearing here.

## Concerns

- The shared environment emits existing matplotlib/pyparsing deprecation warnings; the zone
  suite also emits one existing NumPy two-dimensional-vector deprecation warning.
- A first `py_compile` attempt could not write macOS's default user cache under the sandbox.
  Re-running with `PYTHONPYCACHEPREFIX=/tmp/cvti-task2-pycache` passed.
