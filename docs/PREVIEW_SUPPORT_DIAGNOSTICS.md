# Preview Support Diagnostics

Support exports now include `preview_diagnostics.json` from the running API's
viewer-owned captures. No additional camera connection or inference is started
by exporting diagnostics. Source URLs and footage are not included.

Each camera record contains open attempts, last open duration, decoded and
published counts, first decoded timestamp and delay, last decoded age, read
failures, average publication FPS and the latest 20 state transitions.
Publication FPS measures JPEG production, not browser delivery or camera FPS.
Successful OpenCV open return does not prove a decoded image was received.

The most recent completed capture session per camera survives viewer disconnects
in memory (up to 128 cameras). Active sessions replace that history in exports.
Counters are per capture session, not lifetime totals. Restarting the API clears
preview history. An inactive record is historical, not current camera health.

OpenCV does not provide structured RTSP/authentication responses or native
decoder stderr through this capture interface. The export explicitly marks
protocol details unavailable; a read failure must not be called an authentication
failure. Credentials are redacted again at bundle creation.

`health.json.saved_report_freshness` labels saved reports recent, stale or unknown
using their generation timestamps. Recent does not prove a process is running.
Reports older than 30 seconds or dated in the future are labelled stale.

The camera tile says "Preview unavailable" on failure. When monitoring is stopped,
the overview identifies its log message as potentially historical instead of
presenting it as a new monitoring failure.

## Pilot Retest

1. Install the new Windows artifact, then open the existing site.
2. Leave monitoring stopped and view the affected cameras for two minutes.
3. Export support diagnostics before closing Argus.
4. Compare active capture records, first-frame delay, failures and frame age.
5. Correlate with Task Manager; old perf reports are not current CPU readings.

Local validation: 33 focused Python tests, 248 frontend tests, frontend and
Electron production compilation. Windows packaging and live-camera validation
are separate gates; these changes do not establish an HEVC corruption fix.
