# Task 4 Report: Global And Per-Camera Overlay Controls

## Status

Implemented global and per-camera tracking-overlay controls across the standard and
streams-only camera walls. Tracking overlays default to hidden, only the authenticated
operator's global preference is persisted, and camera overrides reset with the React session.

## Implementation

- Added `TrackingPreference = "global" | "show" | "hide"`, strict global preference
  load/save helpers, immutable override updates, and explicit global/override precedence.
- Added an accessible `Show tracking` scan-icon toggle to both wall toolbars.
- Added compact scan-icon camera menus with `Use global`, `Show`, and `Hide` options in each
  tile's caption/action row, outside the video content.
- Kept global preference state and session-only camera overrides in `App`, clearing overrides
  when the authenticated identity changes.
- Passed each camera's resolved tracking boolean into `CameraStream` and included it in the
  stream-resolution effect dependencies. The resolver invokes `camera_stream(cameraId,
  tracking)` without touching monitoring state.
- Kept hidden streams-only prefetch connections raw; they remount into the visible tree with
  the resolved setting when displayed.

## Files

- Created `Frontend/src/lib/tracking-overlay.ts`.
- Modified `Frontend/src/App.tsx`.
- Modified `Frontend/src/components/StreamsWall.tsx`.
- Modified `Frontend/src/components/CameraStream.tsx`.
- Modified `Frontend/src/styles.css`.
- Created `Frontend/tests/tracking-overlay.test.ts`.
- Modified `Frontend/tests/streams-wall.test.ts`.
- Modified `Frontend/tests/ui.spec.ts`.
- Created this report.

## RED / GREEN Evidence

### Preference RED

```text
cd Frontend && npm test -- tracking-overlay.test.ts streams-wall.test.ts

FAIL tests/tracking-overlay.test.ts
FAIL tests/streams-wall.test.ts
Error: Cannot find module '../src/lib/tracking-overlay'
Test Files  2 failed (2)
```

This was the expected failure before the preference module existed.

### Preference GREEN

```text
cd Frontend && npm test -- tracking-overlay.test.ts streams-wall.test.ts

Test Files  2 passed (2)
Tests       16 passed (16)
```

### UI RED

```text
npx playwright test tests/ui.spec.ts --grep "tracking overlay controls"

FAIL tracking overlay controls persist globally and keep camera overrides session-only
Locator: getByRole('button', { name: 'Show tracking' })
Error: element(s) not found
```

This was the expected product failure before the controls were implemented. An initial
sandboxed attempt could not launch Chrome; the authoritative RED run used the approved
outside-sandbox Playwright invocation.

### UI GREEN

```text
npx playwright test tests/ui.spec.ts --grep "tracking overlay controls"

1 passed (2.7s)
```

The workflow proves the default raw request, per-camera precedence, global persistence,
session-only override reset, both wall surfaces, unchanged monitoring state, and that a
camera whose resolved value does not change is not re-resolved.

## Final Verification

```text
cd Frontend && npm test
Test Files  22 passed (22)
Tests       118 passed (118)

cd Frontend && npm run build
TypeScript build, Vite production build, Electron compile, and preload generation passed.

npx playwright test tests/ui.spec.ts --grep "streams-only toolbar auto-hides"
1 passed (8.4s)
```

The streams-only visual workflow regenerated desktop and mobile screenshots. Manual inspection
confirmed stable tiles, no horizontal overflow, and all overlay controls outside video content.

The complete UI run finished with `18 passed, 2 failed`. Both failures are unchanged demo-media
tests: this worktree has no `Frontend/public/demo` directory, so all referenced MP4 and snapshot
assets return unavailable. The media-count test receives zero videos, and the zone-canvas test
cannot obtain a natural image ratio. Both failures reproduce individually and do not traverse
the tracking stream controls.

## Self-Review

- Re-read the Task 4 brief and parent design against the final diff.
- Confirmed the exact storage key is `argus:tracking-overlay:${operatorId}` and only the global
  boolean reaches storage.
- Confirmed malformed/missing storage values remain hidden and operator IDs isolate values.
- Confirmed `show` overrides global false, `hide` overrides global true, and `Use global`
  removes the camera entry from session state.
- Confirmed stream resolution depends on the resolved boolean, not the override object, so one
  override reopens only that camera and a global change skips cameras already at that value.
- Confirmed visibility changes do not call `start_monitoring` or `stop_monitoring`.
- Confirmed icon controls have accessible labels, tooltips, keyboard focus treatment, and
  responsive dimensions at density 16.
- Confirmed `git diff --check` and the production build pass. No subagents were used.

## Concerns

- The two existing demo-media UI tests cannot pass in this worktree until its ignored/untracked
  `Frontend/public/demo` assets are provisioned. Task 4's targeted UI workflow and the other 18
  UI tests pass.

## Fix Round 1

### Changes

- Replaced separate global and override state with one atomic `TrackingPreferenceState` keyed
  by operator ID.
- Reconciled tracking state as soon as `auth_state` returns. A newly observed operator loads
  only their exact `argus:tracking-overlay:${operatorId}` value and starts with no camera
  overrides.
- Gated standard-wall stream activation and streams-only rendering until the preference state
  key exactly matches the authenticated username. During a same-transport identity refresh,
  old streams deactivate before the new workspace request completes.
- Added a mode-change handler that clears workspace, tracking preferences, location preferences,
  selected details, streams-only state, and sync metadata in the same React update as the
  transport change.
- Expanded the real transport workflow to prove `Hide` requests `false` only for the selected
  camera, then `Use global` requests `true` only for that camera, with no sibling remount.
- Added a real-component regression that starts operator A with global tracking enabled and a
  camera override, switches to operator B on a new transport with no saved preference, and
  proves B receives no `camera_stream(..., true)` request.

### RED / GREEN Evidence

Operator-isolation RED:

```text
npx playwright test tests/ui.spec.ts --grep "operator transport switch"

Expected: false
Received: true
1 failed
```

The B-side transport invocation ledger contained an annotated stream request originating from
operator A's preference state.

Focused GREEN:

```text
npx playwright test tests/ui.spec.ts --grep "operator transport switch"
1 passed (2.3s)

npx playwright test tests/ui.spec.ts --grep "tracking overlay controls"
1 passed (2.6s)

npx playwright test tests/ui.spec.ts --grep \
  "tracking overlay controls|operator transport switch"
2 passed (4.1s)
```

### Verification

```text
cd Frontend && npm test -- tracking-overlay.test.ts streams-wall.test.ts
Test Files  2 passed (2)
Tests       16 passed (16)

cd Frontend && npm test
Test Files  22 passed (22)
Tests       118 passed (118)

cd Frontend && npm run build
TypeScript build, Vite production build, Electron compile, and preload generation passed.

cd Frontend && npm run test:ui
19 passed, 2 failed
```

The two full-UI failures are unchanged: `Frontend/public/demo` is absent, so the existing real
media test finds no playable videos and the existing zone-canvas test has no natural snapshot
ratio. All remaining UI tests, including both fix-round workflows, pass.

### Self-Review

- Confirmed the exact global storage key and session-only override behavior are unchanged.
- Confirmed a new identity is keyed before its workspace data is installed, making old streams
  inactive during the transition rather than resolving with another operator's preference.
- Confirmed transport mode changes invalidate in-flight refreshes and synchronously clear both
  workspace and preference state before the new API can render a camera.
- Confirmed global writes occur only when preference ownership matches the current username,
  preventing a stale value from being persisted under the new operator's key.
- Confirmed camera handlers reject updates when preference ownership and authenticated username
  do not match.
- Confirmed exact invocation arrays cover both target remounts and absence of non-target
  remounts for `Hide` and `Use global`.
- No subagents were used.

### Concerns

- The worktree still lacks the untracked demo media described above; this remains unrelated to
  tracking preference isolation.
