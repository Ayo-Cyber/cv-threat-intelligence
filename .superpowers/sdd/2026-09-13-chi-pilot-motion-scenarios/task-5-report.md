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

The initial run had two failures with different causes:

1. `test_ultralytics_is_pinned_and_the_seam_looks_as_assumed`: installed
   Ultralytics is 8.4.64; `requirements.txt` pins 8.4.35. This is the only
   environment-caused failure.
2. `test_the_committed_baseline_matches_the_current_prompts`: committed
   fingerprint is `a2e093837f82...`; current prompt fingerprint is
   `0bf660f4a024...`. This was feature-caused by the scenario-5 prompt change,
   not by the environment.

The initial pass did not rewrite the prompt baseline. Fix round 1 updates only
the current fingerprint and prompt counts; the measurement remains explicitly
unmeasured and no replacement metrics are fabricated.

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
- Confirmed the Ultralytics environment mismatch, the feature-caused historical
  prompt fingerprint failure, and absent frontend media are visible.
- Confirmed no subagents were used.

## Concerns

- The shared Python environment must be rebuilt at the pinned Ultralytics
  8.4.35 before a release-quality rerun.
- The current prompt needs a complete frozen-corpus measurement before its null
  precision and recall can be replaced honestly.
- Controlled, rights-cleared Chi recordings and independently reviewed labels
  are still required to complete acceptance.
- Scenario 5 currently lacks a dedicated candidate-audit record before queue
  admission, so a queue duplicate cannot be reconstructed as precisely as a
  concealment candidate.

## Fix Round 1

### Changes

- Updated `docs/prompt_baseline.json` to scenario-5-aware fingerprint
  `0bf660f4a02480122fc35b7b961672f9d151420d38d450a0779cf2f2bd2f41c6`
  and detector-question count 8. `measurement_status=unmeasured`, its reason,
  all null current metrics, and the complete previous measurement provenance
  are unchanged.
- Added `tools/score_chi_motion.py` and focused tests. The executable validates
  half-open labels, per-frame observations, generated-candidate/gate audit rows
  from SQLite or exported JSON, and exactly three distinct hidden plus three
  distinct shown performance reports. It atomically writes a retained JSON
  artifact with input hashes, person recall, ID switches, scenario-4 box
  checks, scenario-5 precision/recall, admission/gate counts, duplicate
  candidates and alerts per incident, detection delay, and hidden/shown FPS
  impact.
- Replaced the old generic `expected` workflow with separate
  `scenario4_expected` and `scenario5_expected` fields. Documented half-open
  `[start_s,end_s)` intervals and the exact final-frame exception.
- Added exact capture, candidate-audit export, and scoring commands. The docs
  explicitly require a complete pre-queue scenario-5 audit and reject gate
  directories as a substitute.
- Retained sanitized replay evidence in
  `docs/evidence/chi-motion-local-replay-2026-09-13.json`, including the source
  hash, transient artifact hashes, runtime versions, limitations, and Ollama
  `gemma3:4b` digest
  `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`.

### RED And GREEN

RED was observed before implementation:

```text
python -m pytest tests/test_score_chi_motion.py -q
ModuleNotFoundError: No module named 'tools.score_chi_motion'

python tools/prompt_regression.py check
committed a2e093837f82... != current 0bf660f4a024...
```

Additional red tests demonstrated that impossible audit state combinations and
scenario-4 box mismatches were accepted before their focused validation was
added. Final green results:

```text
python tools/prompt_regression.py check
fingerprint 0bf660f4a024... recorded; metrics UNMEASURED

python -m pytest tests/test_score_chi_motion.py -q
18 passed in 0.12s

python -m pytest tests/test_prompt_regression.py tests/test_score_chi_motion.py \
  tests/test_person_motion.py tests/test_retail_zones.py tests/test_serving.py \
  tests/test_frame_publisher.py tests/test_engine_api.py -q
161 passed, 15 warnings in 35.62s
```

The first focused invocation inside the filesystem sandbox produced 27
localhost-bind permission failures and was rerun with bind permission; those
failures were harness restrictions, not product failures.

After the broad runs, a final parser-hardening check added two focused cases
for malformed JSON structures and case-qualified incident keys:

```text
python -m pytest tests/test_score_chi_motion.py tests/test_prompt_regression.py -q
41 passed in 0.33s
```

Post-fix full Python regression:

```text
1462 passed, 8 skipped, 1 failed, 15 warnings in 160.51s
```

The only failure is
`test_ultralytics_is_pinned_and_the_seam_looks_as_assumed`: local Ultralytics
8.4.64 differs from pinned 8.4.35. The feature-caused prompt fingerprint
failure is resolved. Frontend suites were not rerun because fix round 1 changes
only Python tooling and documentation; the prior 118 unit tests/build pass and
the two absent-demo-media Playwright failures remain the latest frontend
result.

### Remaining Unmeasured Work

The repository still lacks rights-cleared controlled two-mover,
stationary-crowd, occlusion, and zone-boundary recordings with reviewed labels.
The current runtime also lacks a scenario-5 pre-queue audit hook, and no six-pair
isolated overlay captures exist. Therefore person recall, ID switches,
scenario-5 precision/recall, duplicate alerts, formal delay, and hidden/shown
FPS impact remain unmeasured. The scorer and exact operator workflow are ready
for those inputs; no acceptance value was inferred or fabricated.
