# Task 5 Report: Scenarios 4 And 5 End-To-End Validation

## Status

Complete as a validation and operator-documentation task. The focused motion,
serving, stream, and frontend contracts pass. A real local scenario-5 replay
also reached TrueSight and persistence with tracking overlays shown.

The controlled Chi acceptance matrix is not complete. Required labeled media
and frame/identity annotations are unavailable for several cases, so no
unsupported metric or passing acceptance claim is recorded.

## Documentation

- Extended `docs/CHI_PILOT_TESTING.md` with the seven-case scenarios 4/5
  recording matrix, exact CSV annotation schemas, metric definitions,
  acceptance expectations, camera/UI defaults, and a repeatable operator
  workflow.
- Documented that scenario 4 is permitted-zone, moving-person telemetry and
  never an alert.
- Documented that scenario 5 is one aggregate candidate per latched sustained
  interval and does not depend on crowd proximity or density.
- Documented that overlay visibility changes only the selected stream and
  never changes person detection, tracking, detector enablement, or monitoring.
- Updated `docs/PROJECT_CONTEXT.md` with current validation evidence and
  explicit environment/media limitations.
- Added no media fixture because no new recording with documented
  redistribution rights was available.

## Automated Validation

Focused backend, rerun with localhost bind permission after the sandbox denied
the frame-publisher's ephemeral port:

```text
<shared-python> -m pytest tests/test_person_motion.py \
  tests/test_retail_zones.py tests/test_serving.py \
  tests/test_frame_publisher.py tests/test_engine_api.py -q
123 passed, 15 warnings in 35.04s
```

Frontend:

```text
cd Frontend && npm test
22 test files passed; 118 tests passed

cd Frontend && npm run build
TypeScript, Vite production build, Electron compile, and preload generation passed
```

Full Python:

```text
<shared-python> -m pytest
1446 passed, 8 skipped, 2 failed, 15 warnings in 159.86s
```

The two failures are:

1. `test_ultralytics_is_pinned_and_the_seam_looks_as_assumed`: installed
   Ultralytics is 8.4.64; `requirements.txt` pins 8.4.35.
2. `test_the_committed_baseline_matches_the_current_prompts`: committed
   fingerprint is `a2e093837f82...`; current prompt fingerprint is
   `0bf660f4a024...`.

The prompt baseline was not rewritten because the frozen corpus is unavailable
and no replacement metrics were fabricated.

Full Playwright, with Vite started separately as required by
`Frontend/playwright.config.ts`:

```text
cd Frontend && npm run test:ui
19 passed, 2 failed in 51.9s
```

Both failures reproduce the existing absent `Frontend/public/demo` media
condition: the overview receives zero video elements, and the zone workspace
cannot obtain a snapshot's natural image ratio. Both tracking-overlay workflows
pass. An earlier invocation without Vite produced 21 connection-refused
failures and was discarded as an invalid harness run.

## Local Replay

The production pipeline replayed
`data/test_clips/normal_street_01.mp4` (SHA-256
`2ca3549f53d02f645106edadf97f52f3c0760d1f9b07e32e5befba6df1c894c8`)
with both motion scenarios enabled, default thresholds, the whole view
permitted, real local `gemma3:4b`, and an authenticated `tracking=1` viewer.

Observed evidence:

- 205,828 tracking MJPEG bytes received before the intentional 8-second client
  timeout;
- one admitted candidate at clip timestamp 10.0s for tracks 35 and 52;
- one additional candidate suppressed by shared queue deduplication;
- one verified/confirmed TrueSight verdict at confidence 0.90;
- zero gate errors and zero unverified results;
- one non-provisional persisted event;
- three gate frames, 17 event frames, subject crop, and replay clip;
- detection stage 48 samples / 14.595s, rate 3.289/s;
- decode stage 108 samples / 20.037s, rate 5.39/s;
- first detection inference 5.329s.

This is plumbing evidence, not a scenario-5 precision/recall result. The clip
was visually suitable for an exploratory multi-mover replay but had no frozen
pre-run interval, frame, or identity labels.

## Unmeasured Acceptance Cells

- Person-detection recall and ID switches.
- Scenario-5 precision, recall, and formal detection delay.
- Stationary crowd, temporary occlusion, and permitted zone-boundary behavior.
- Exact duplicate candidate rate; scenario 5 has no dedicated pre-queue audit,
  and this replay recorded one queue-deduplicated candidate.
- Hidden-versus-shown FPS. The shown run overlapped the full regression, and no
  overlay-specific timing stage makes a single-run comparison reproducible.

The operator workflow supplies exact clip names, CSV columns, label vocabulary,
zone-authoring command, site generation, production replay, result extraction,
and a three-pair hidden/shown protocol for the team to complete these cells.

## Self-Review

- Re-read the Task 5 brief and parent design against the final documentation.
- Confirmed all seven required recording cases are present.
- Confirmed the docs separate detector metrics, queue deduplication, TrueSight
  verdicts, and persisted alert counts.
- Confirmed missing labels remain unmeasured and the local replay is called a
  smoke result.
- Confirmed global/per-camera controls, lifetimes, defaults, and detection
  independence are explicit.
- Confirmed the environment mismatch, prompt fingerprint failure, and absent
  frontend media are visible.
- Confirmed no subagents were used.

## Concerns

- The shared Python environment must be rebuilt at the pinned Ultralytics
  8.4.35 before a release-quality rerun.
- The prompt fingerprint needs a complete frozen-corpus measurement before the
  baseline can be updated honestly.
- Controlled, rights-cleared Chi recordings and independently reviewed labels
  are still required to complete acceptance.
- Scenario 5 currently lacks a dedicated candidate-audit record before queue
  admission, so a queue duplicate cannot be reconstructed as precisely as a
  concealment candidate.
