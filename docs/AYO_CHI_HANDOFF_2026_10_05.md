# CHI pilot handoff to Ayo - 5 October 2026

This is an engineering checkpoint, not production acceptance of all KPIs.
The Electron live test was stopped at the user's request for this handoff.

## Scope and status

- KPI 4/5: person tracking overlays, visible per-camera/all-camera switches,
  synchronized frame overlays and multi-camera validation improvements. Difficult
  lighting, perspective and small subjects still require broader validation.
- KPI 10: concealment evidence and queue improvements, transient stream notices,
  customer-facing review wording and tests. Alerts mean possible concealment,
  not confirmed theft. Repeated alerts and detection reliability need further
  positive/negative evaluation; three-camera tests were not a production pass.
- KPI 9: registered-object ROI monitoring, reference persistence, API and live
  registration editor are implemented. LIVE BOTTLE REMOVAL HAS NOT PASSED.
  Repeated runs invalidated the baseline as scene_or_lighting_changed and
  produced no removal incident. Automatic abandoned-object discovery is not done.
- KPI 3: the real SigLIP model loaded locally and produced a normalized 768-value
  embedding. No bottle was enrolled in the final session, and recognition
  accuracy or end-to-end alerts have not been demonstrated.

## Immediate next steps

1. For KPI 9, retain the exact frame and metrics that invalidate the reference,
   then diagnose exposure/scene changes before another live test. Do not silently
   recapture a missing object or treat synthetic replay as a live pass.
2. For KPI 3, enroll reviewed bottle reference photos and negative examples,
   prepare/activate embeddings, configure the camera rule, then test independent
   positive and negative views through the actual engine and incident queue.
3. Package/download the recognition model through customer setup; weights are
   local test assets only and are deliberately NOT included in this PR.
4. Validate concealment recall, false positives and duplicate incidents across
   held-out footage. Keep operator review and uncertainty visible.

## Preview and setup fixes included

- Numeric webcam preview waits for exposure warm-up before publishing.
- Preview descriptors survive the Electron API adapter; stopped engines use
  preview rather than stale monitoring frames.
- Add-camera accepts valid derived areas backed by existing cameras and rejects
  stale areas.
- Registered-object drawing uses a continuous camera stream, preserves aspect
  ratio and scrolls to registration controls on short windows.

## Tests and local assets

Run backend tests from the repository root with `python -m pytest -q` and the
prompt guard with `python tools/prompt_regression.py check`. In `Frontend`, run
`npm test`, `npm run build`, and `npm run test:camera-stream`.

Historical evidence and datasets are under `runs/chi_validation/` on the test
Mac, not distributed in Git. Diagnostic scripts use those local assets; they
are not turnkey customer setup. Desktop scripts accept `ARGUS_PYTHON` for the
Python interpreter and require the existing isolated test credentials/config.
Do not publish credentials, webcam photographs, event databases or model files.

The local SigLIP probe used google/siglip-base-patch16-224 revision
7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed, installed under
`runs/chi_validation/desktop/object_library/models/siglip`.
The CPU embedding probe took about 0.13 seconds; this is not pipeline latency.

See PROJECT_CONTEXT.md, CHI_KPI9_REGISTERED_OBJECT_TEST.md,
CHI_KPI10_THREE_CAMERA_CHECKPOINT.md and CONCEALMENT_RELIABILITY_HANDOFF.md
for detailed history, commands, evidence paths and limitations.
