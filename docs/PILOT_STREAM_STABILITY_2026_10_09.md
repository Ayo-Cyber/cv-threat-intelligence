# Pilot stream stability: 9 October 2026

Baseline: v1.9.4 / 86715e9. These changes are local source fixes, not an
installed Windows update. No customer cameras or server settings were changed.

## Latest evidence

Reviewed the supplied diagnostics ending `224229-0a39c834` and screenshots.
The engine has 12 cameras, 11 connected and Combi block filling room offline.
Only SMB BAY 3/4, Beefie Loading Bay and KC Production 2 have Normal movement
enabled. All other exported detector flags are off, including Caprisone's.
CPU remains 100%; detector batches average 2400 ms, p95 5326 ms. No pose stage
appears in this report. Disabling heavy detectors alone did not remove overload.
The stopped-monitoring screenshot shows about 38% total CPU, 34.3% for Argus.

## Reproduced code defects and fixes

1. Camera upsert removed an existing camera and appended it. Saving detector
   settings consequently moved its tile, changing viewport-based subscriptions.
   Replace in place instead; new cameras still append. Detector saves do not
   intentionally restart monitoring.
2. LiveWall slept after every live read. A 30 FPS source read at an 8 FPS preview
   cadence could accumulate old frames even with monitoring stopped. Drain live
   capture at source cadence, but throttle JPEG publication. Preserve local-file
   pacing and webcam exposure warmup. This removes artificial read delay, not
   camera/network/codec delays, and is not a claim of reduced decode CPU.
3. A stream that never supplied its first image could remain Connecting forever.
   Add a 20-second first-frame deadline and reuse the existing five-second retry.
   This does not detect every post-first-frame freeze or repair an invalid RTSP URL.
4. Smooth publication reused person coordinates without an age limit. Expire
   person overlays after one second from inference admission, not completion.
   Preserve exact-frame mode's boxes on their matching inference frame. Slow
   inference now means fewer visible boxes rather than seconds-old boxes stuck
   on newer frames. This does not improve detector recall or compute throughput.

## Unresolved and next acceptance checks

- HEVC corruption remains unverified on real sources. Coloured blocks and grey
  video plus RTP/HEVC parser errors indicate a media-path issue, not just UI
  styling. Compare one affected stream in an independent authorized player on
  the server, and test a camera-provided H.264 alternative with IT approval.
  Do not guess or silently rewrite camera URLs or production codec settings.
- Combi still needs a verified source/port/path/authentication check.
- Caprisone missing boxes cannot be attributed solely to fisheye geometry while
  its movement detector is disabled. Enable it deliberately in a test window;
  then assess distortion, person size and dewarping separately.
- CPU capacity remains insufficient for the measured workload. Benchmark after
  installing the fixes; do not promise an FPS improvement from source tests.
- Test save-detector tile stability, no-first-frame retry, stopped-monitoring
  live latency, restart handover and movement-box expiry on the Windows server.
- Support bundles are private and remain outside this source change. Do not
  commit credentials, footage or unredacted client logs.

## Verification

Passed locally: 27 preview/onboarding tests, 72 pipeline/publisher/PPE tests,
248 frontend unit tests, four browser stream tests (including stalled-first-frame
recovery), and the production frontend/Electron build. Initial sandbox socket
restrictions required rerunning server tests with loopback access. These checks
are not equivalent to a Windows pilot camera retest.
