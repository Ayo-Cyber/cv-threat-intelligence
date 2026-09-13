# Task 3 Report: Authenticated Raw And Tracking Streams

## Status

Implemented authenticated raw and tracking stream variants end to end. Raw JPEGs remain canonical and WebRTC-first. `tracking=true` forces the authenticated local MJPEG variant, and annotated JPEG work occurs only while a tracking viewer is connected.

## Implementation

- Added immutable `FrameOverlay` records and separate raw/tracking JPEG caches, sequences, and viewer counts.
- Preserved raw pixels and source arrays while drawing semantic movement labels and boxes into the tracking variant.
- Refreshed tracking frames when overlays clear so old boxes cannot remain frozen.
- Fed Task 2 `_motion_overlays` through smooth, direct, and paced live-source publishing. View-only cameras continue to receive no overlays.
- Added `tracking` to the FastAPI descriptor route. Raw requests retain WebRTC-first selection; tracking requests return `/stream/{camera}?tracking=1&token=...` MJPEG.
- Forwarded an exact boolean through the Electron API client, legacy custom-protocol bridge, shared types, and frontend stream resolver. Account tokens remain in authorization headers, and the legacy publisher token remains behind `argus-stream:`.

## TDD Evidence

RED:

- Publisher tests failed because `FrameOverlay` and `_tracking_viewers` did not exist.
- Smooth/API tests failed because overlays were omitted and `tracking=true` still selected WebRTC.
- Frontend tests failed because the API query, bridge descriptor, and resolver invocation dropped the boolean.
- Follow-up regression tests failed on stale tracking boxes and missing overlays for paced JPEG playout.

GREEN:

- `/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python -m pytest tests/test_frame_publisher.py tests/test_engine_api.py tests/test_serving.py -q`
  - `85 passed, 14 warnings in 33.27s`
- `cd Frontend && npm test -- api-client.test.ts bridge-transport.test.ts whep.test.ts`
  - `3 passed` test files, `33 passed` tests
- `cd Frontend && npm run build`
  - TypeScript, Vite production build, Electron TypeScript, and preload generation completed successfully.

## Self-Review

No blocking findings remain. Raw and tracking routes share authentication, tracking counts are released on disconnect, false/default tracking remains WebRTC-first, and annotation does not affect detection cadence. The 14 Python warnings are existing matplotlib/pyparsing deprecations and are unrelated to this task.

## Fix Round 1

Hardened same-frame cache ownership and publication atomicity:

- Added a per-camera tracking viewer generation. The first viewer of a session and the final disconnect invalidate tracking frame and sequence state.
- Guarded commits with the captured generation and a live tracking-viewer count, preventing annotation started by an old session from repopulating a later session's cache.
- Prepared paced-JPEG annotations before mutating raw state, then committed raw/tracking variants together under one lock with the same publish sequence.
- On annotation failure, committed the new canonical raw frame while atomically invalidating the old tracking frame and sequence.
- Added regressions for disconnect, intervening raw-only publications, reconnect, in-flight annotation across generations, paired sequences, and annotation failure.

RED:

- The three new focused regressions failed: disconnect retained `(old_jpeg, 1)`, in-flight work repopulated tracking after reconnect, and annotation preparation observed raw already advanced.

GREEN:

- `python -m pytest tests/test_frame_publisher.py -q`
  - `31 passed in 17.86s`
- `python -m pytest tests/test_engine_api.py tests/test_serving.py -q`
  - `57 passed, 14 warnings in 16.64s`
- `cd Frontend && npm test -- api-client.test.ts bridge-transport.test.ts whep.test.ts`
  - `3 passed` test files, `33 passed` tests

Self-review found no blocking concerns. The Python warnings remain the existing matplotlib/pyparsing deprecations.
