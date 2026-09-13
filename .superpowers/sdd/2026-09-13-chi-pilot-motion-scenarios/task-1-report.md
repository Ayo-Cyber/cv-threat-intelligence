# Task 1 Report: Reusable Person Motion State

## Implementation

- Added immutable `PersonMotion` snapshots containing track ID, bounding box, normalized
  speed, movement state, continuous movement duration, and zone names.
- Added `PersonMotionTracker` with center-displacement velocity normalized by frame diagonal,
  an EMA (`ema_alpha=0.20`), `0.05`/`0.02` enter/exit hysteresis defaults, a `0.4` second
  minimum track age, and stale-track expiry after `1.0` second by default.
- Added `SimultaneousMovementDetector` with configurable count and persistence, one event per
  sustained interval, and reset only when the moving count falls below threshold.
- Added aggregate event metadata with confidence, people count, ordered track IDs, measured
  per-person speed and duration, zone names, and the union group bounding box.
- Kept normal movement as state only. No alert, camera-pipeline, UI, or other integration path
  was added or modified.

## Files

- Created `cvti/detector/person_motion.py`.
- Created `tests/test_person_motion.py`.
- Created `.superpowers/sdd/2026-09-13-chi-pilot-motion-scenarios/task-1-report.md`.

## RED / GREEN Evidence

All commands used the required shared Python 3.9 environment:

`/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python`

### Initial RED

Command:

```text
<shared-python> -m pytest tests/test_person_motion.py -q
```

Relevant output:

```text
E   ModuleNotFoundError: No module named 'cvti.detector.person_motion'
ERROR tests/test_person_motion.py
1 error in 15.85s
```

This was the expected failure because the production module did not exist.

### Tracker GREEN

The first implementation run correctly exposed jitter calibration as still red:

```text
FAILED tests/test_person_motion.py::test_bbox_jitter_stays_stationary
speed_ratio=0.051237620277384585, moving=True
1 failed, 4 passed
```

After changing the default EMA alpha from `0.25` to `0.20`:

```text
.....                                                                    [100%]
5 passed, 14 warnings in 15.34s
```

### Aggregate RED

Command:

```text
<shared-python> -m pytest tests/test_person_motion.py -q
```

Relevant output:

```text
E   ImportError: cannot import name 'SimultaneousMovementDetector'
ERROR tests/test_person_motion.py
1 error in 15.51s
```

This was the expected failure before implementing the aggregate detector.

### Final GREEN

Command:

```text
<shared-python> -m pytest tests/test_person_motion.py -q
```

Relevant output:

```text
..........                                                               [100%]
10 passed, 14 warnings in 15.27s
```

## Tests

The focused tests cover sustained normalized displacement, bounding-box jitter rejection,
minimum track age, EMA smoothing, enter/exit hysteresis, continuous moving duration,
stale-track expiry and ID reuse, snapshot boxes and zones, simultaneous count, persistence,
stationary groups, spread-out moving people, event metadata, latching, and count-drop reset.

Final regression command:

```text
PYTHONPYCACHEPREFIX=/tmp/cvti-pycache <shared-python> -m py_compile \
  cvti/detector/person_motion.py tests/test_person_motion.py
<shared-python> -m pytest tests/test_person_motion.py tests/test_situational_hse.py -q
```

Relevant output:

```text
.................                                                        [100%]
17 passed, 14 warnings in 15.41s
```

## Self-Review

- Re-read the task brief, design, and Task 2 consumer contract against the implementation.
- Confirmed the public dataclass fields and both `update` signatures match the brief.
- Confirmed frame normalization uses `(height, width)` correctly and has a nonzero diagonal
  floor.
- Confirmed only currently observed tracks are returned while missing state is retained until
  expiry.
- Confirmed the aggregate detector filters by `moving`, has no proximity rule, and resets its
  latch only below `min_people`.
- Confirmed realistic mutations to thresholds, smoothing, expiry, qualifying count,
  persistence, latch reset, metadata, or group-box union are covered by focused assertions.
- Confirmed `git diff --check` passed and no existing source or integration file was changed.

## Concerns

- Ruff is not installed in the shared virtual environment (`No module named ruff`), so Ruff
  could not be run. Manual line-length review, `git diff --check`, Python 3.9 bytecode
  compilation, and pytest were used instead.
- Pytest emits 14 existing matplotlib/pyparsing deprecation warnings. Test startup also prints
  existing duplicate AVFoundation class warnings from the shared `cv2` and `av` binaries;
  neither warning originates in this task.
