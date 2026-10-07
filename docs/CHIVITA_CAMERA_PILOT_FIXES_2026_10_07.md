# Chivita Camera Pilot: Findings and Retest

## Confirmed Defects

1. **False connection-test success.** `ConsoleBackend.test` returned an error
   without `ok: false` when the RTSP probe failed. The API and AddCamera could
   interpret this as success. The backend now returns explicit failure; the
   form also rejects legacy error responses and requires decoded dimensions.
   Success reads `Video received (width x height)`, not merely reachable.
2. **Slash-containing camera IDs.** The saved name `SMB BAY 3/4` is also its ID.
   ASGI decodes `%2F` before path matching, producing a routing 404 for preview,
   deletion and other camera operations. Query-addressed `/camera-by-id`
   aliases preserve existing IDs and reuse the same permission checks and
   backend operations. Electron uses them when the camera ID contains `/`.
3. **Conflicting monitoring status.** WebSocket health read the heartbeat file
   while `/monitor` also considered the child process. Both now share the
   process-aware state, including transitions before the first heartbeat and
   after stopping. The frontend distinguishes Starting from Running.
4. **Unbounded network video waits.** FFmpeg opens/reads now request 8,000 ms
   and 5,000 ms timeouts, respectively, including software fallback. The
   connection test uses the same network capture configuration as the engine.
   Failed hardware captures are released before retry. Two open attempts can
   still take approximately 16 seconds; these settings do not bound model load
   time, RTSP probing, or the entire Start operation.

## Detection Configuration

The submitted screenshot shows `0 detectors enabled` for both cameras. The
Person boxes toggle controls rendering, not activation of movement tracking.
Open the camera's Scene review, select Detectors, and enable Normal movement.
Save the configuration and restart monitoring once before the acceptance run.

The circular KC Production 2 view is not a conventional perspective image.
Enabling a detector does not prove it will recognize small, rotated people
around a fisheye image. Prefer a camera/NVR-provided dewarped perspective stream
if available. Confirm its RTSP address with site IT; do not guess vendor paths.
Test a bullet camera first to separate configuration/engine health from optical
distortion. No fisheye dewarping or accuracy claim was added in this patch.

## Windows Retest

1. Back up the site's configuration and database before upgrading. Do not
   delete the site or rename existing camera IDs as a workaround.
2. Build an updated Windows package containing BOTH Electron frontend and
   Python API/engine changes. These changes are not in the installed v1.9.2
   release just because they exist in this workspace.
3. On the server, test each camera individually. Require a visible feed, not
   just ping or TCP reachability. Confirm RTSP credentials, path, codec and
   camera session limits if a test fails. Use a supported H.264 substream where
   the site permits it, but do not change camera settings without IT approval.
4. Confirm the existing SMB camera can preview and be removed when intended.
   The regression suite verifies deletion preserves other camera entries.
5. Enable Normal movement on one bullet camera. Keep Person boxes on. Click
   Start monitoring once, record elapsed time and the displayed startup phase.
6. Verify moving people receive boxes, then add the remaining cameras one at
   a time and check CPU/GPU usage, frame freshness and reconnect counts.
7. Validate the fisheye separately with its dewarped stream or an agreed
   camera-specific calibration. Do not count raw circular-feed accuracy as passed.

## Evidence Needed for Remaining Diagnosis

From the Windows server, inspect the last startup section of
`%APPDATA%\Argus\site\monitor.log` and the contemporaneous
`%APPDATA%\Argus\site\gate_health.json`. Confirm actual paths under Settings
if the installation uses a different site directory. Redact passwords, RTSP
userinfo, tokens and other secrets before sharing logs. Also record the
installed Argus version and GPU model/driver information.

The seven-minute delay and the other cameras' exact failures have not been
reproduced against Chivita's network. A false-positive connection test and a
routing 404 are confirmed code defects, not proof that every missing feed has
the same cause. No remote server or camera configuration was changed here.

## Local Verification

- 150 backend tests passed: API reads/writes and authorization, query camera
  aliases, contract coverage, RTSP probe fixtures, capture configuration,
  preview streaming/release, startup health and gateway integration.
- 248 frontend unit tests passed.
- Six Playwright camera/registered-object checks passed on browser fixtures.
- Frontend production build and Electron TypeScript compilation passed.

Local automated tests are not a Windows deployment or a Chivita RTSP soak test.
Project context was updated. Changes remain uncommitted and undeployed.
