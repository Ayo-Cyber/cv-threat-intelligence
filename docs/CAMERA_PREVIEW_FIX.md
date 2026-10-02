# Independent Camera Preview

## Problem and Changes

The Electron API requested only engine-owned stream descriptors. Before
monitoring started there was no publisher, so reachable cameras appeared offline.

- The authenticated stream route now offers preview MJPEG when monitoring is
  stopped or has never started. Descriptor requests alone do not open cameras.
- `CameraPreview` opens one decoder per requested camera when a viewer connects.
  Multiple viewers share it; the last disconnect releases it. No inference starts.
- Pre-monitoring zone snapshots borrow this capture and preserve source dimensions.
- Starting the engine closes previews first. If a capture is still blocked,
  startup is refused with a retryable 503 (`preview_releasing`) rather than
  opening a competing webcam handle, and the next preview request opens a
  fresh capture.
- The frontend refreshes descriptors to switch between preview and engine streams.
  Preview media is not hidden by stale engine-health status; its label is LIVE PREVIEW.
- Removed cameras, changed camera configuration, feed switches, sign-out and API
  shutdown close previews. Camera changes/sign-out currently reset all preview tiles;
  remaining authorized viewers reconnect automatically.
- The existing localhost token requirement remains. Failed captures no longer
  serve stale JPEGs; stalled streams close so clients can retry.

## Verification

API, gateway, preview lifecycle and frontend stream unit tests pass. A real
localhost MJPEG test uses a generated clip to prove multiple different frames
arrive and the decoder stops after disconnect. TypeScript compilation passes.

## Remaining Manual Acceptance

No physical webcam, RTSP endpoint, AI inference run or packaged Windows build
was exercised. Test on the target machine: add webcam, preview before monitoring,
draw zones, start monitoring, stop monitoring, hide tiles, sign out and reconnect.
Check scene mapping separately; this change does not alter its queue or inference.
Monitoring-time zone snapshots retain the existing snapshot implementation.
Externally launched engine publisher-file validation retains its existing behavior.
