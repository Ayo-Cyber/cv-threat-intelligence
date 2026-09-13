# Task 4 Report: Validate Scenario 10 End To End

## Status

Documentation and automated concealment validation are complete. The focused
suite passes. The full Python suite has one environment/dependency failure
unrelated to concealment. Scenario 10 is not yet empirically accepted because
the seven Chi clips have not been recorded/run and the changed prompt cannot be
replayed while the frozen golden corpus is unavailable.

Documentation commit: `2cf3de3` (`docs(pilot): add Chi concealment acceptance
matrix`).

## Documentation

- Updated `docs/PROJECT_CONTEXT.md` with the actual repaired architecture and
  an explicit unmeasured-prompt handoff.
- Added `docs/CHI_PILOT_TESTING.md` with the concealment-only seven-case matrix,
  fixed relative event intervals, candidate and TrueSight expectations,
  persistence/UI acceptance, result fields, acceptance gates, and exact
  commands.
- Kept motion scenarios 4 and 5 out of scope for their later plan.

## Verification

Focused command:

```bash
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" -m pytest tests/test_concealment.py tests/test_heavy_models_earn_their_frames.py tests/test_serving.py tests/test_prompt_regression.py tests/test_gate_evidence_quality.py -q
```

Result: `88 passed, 14 warnings in 4.06s`.

Full command:

```bash
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" -m pytest
```

Sandbox result: `51 failed, 1343 passed, 8 skipped, 16 warnings, 8 errors in
129.00s`. The failures/errors were dominated by denied localhost binds
(`PermissionError: [Errno 1] Operation not permitted`) and included two denied
writes under `~/Library/Application Support/Argus`. No production code was
changed in response.

The identical command was rerun with local permission. Result: `1 failed, 1401
passed, 8 skipped, 15 warnings in 152.71s`. All localhost and Argus-directory
permission failures cleared. The sole remaining failure was:

```text
tests/test_detection_rides_the_accelerator.py::SeamPins::
test_ultralytics_is_pinned_and_the_seam_looks_as_assumed
```

The supplied shared interpreter imports `ultralytics==8.4.64`, while
`requirements.txt` pins `ultralytics==8.4.35`. This task does not modify the
shared environment or dependency pins.

Prompt status command:

```bash
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" tools/prompt_regression.py check
```

Result: exit `0`; fingerprint `a2e093837f82...` is recorded, with metrics
labeled `UNMEASURED` because the changed prompt cannot be replayed without the
frozen golden corpus. Sandbox logging fell back to console because the Argus
log directory is outside the writable roots.

## Self-Review

- Confirmed the matrix contains pocket-positive, bag-positive, trolley-safe,
  phone-to-pocket, clothing-adjustment, browsing, and open-carry cases.
- Confirmed every case has labeled intervals and separate candidate, gate, and
  delivered-event expectations.
- Checked the architecture against `cvti/retail/concealment.py`,
  `cvti/serving/camera.py`, `cvti/event_adapters.py`,
  `cvti/verification/frame_select.py`, `cvti/verification/gate.py`, the alert
  queue/gate pool/sink, and the focused tests.
- Corrected the handoff to describe provisional alerts being retained on
  confirmation/unverified outcomes and retracted on normal gate rejection.
- `git diff --check` reports no whitespace errors.

## Concerns

1. The changed concealment prompt has no current precision or recall result.
   Previous-prompt metrics must not be attributed to fingerprint
   `a2e093837f82...`.
2. The seven-case Chi matrix is an acceptance protocol, not a completed pilot
   result. Positive end-to-end persistence/UI evidence and negative-case
   rejection still need real recorded clips and a running TrueSight provider.
3. The full suite is not entirely green under the supplied interpreter because
   its installed Ultralytics version differs from the repository pin.

## Fix Round 1

Review requested an exact end-to-end operator workflow and portable command
setup in `docs/CHI_PILOT_TESTING.md`.

Fix commit: `bd91d8a` (`docs(pilot): add scenario 10 operator workflow`).

Changes:

- Replaced workstation-bound regression commands with `REPO_ROOT`, `PYTHON`,
  and `MPLCONFIGDIR` variables, retaining the source-venv path only as an
  explicit fallback for this linked worktree.
- Added new-checkout virtual environment and desktop UI dependency setup.
- Separated synthetic/mock/stub regression smoke from real local TrueSight
  acceptance and explicitly prohibited mock-gate acceptance claims.
- Added ordered commands to stage all seven clips, verify/start Ollama, select
  one case, generate and validate its one-camera production site config, run
  the synchronous production mapping preflight, approve the mapped context in
  the application, run the finite clip once through the production engine and
  TrueSight, retain candidate/verdict logs, query SQLite persistence, enumerate
  evidence artifacts, and inspect replay in the UI.
- Kept each case in a timestamped output directory so candidate, gate,
  persistence, replay, and UI results have an unambiguous denominator.

Command and contract validation:

- `cvti.serving.pipeline --help` confirmed every documented engine flag.
- `cvti.app.shell --help` confirmed `--site-config` and `--db`.
- `tools/prompt_regression.py run --help` confirmed all replay flags.
- `ollama --help` confirmed `serve`, `pull`, and `list`.
- The documented config generator was executed against a disposable clip; the
  real `load_site_config()` accepted the generated camera, rules path,
  concealment toggle, stride, and accepted zone role.
- `prepare_scene_mapping()` was signature-checked and executed with its mock
  provider against the disposable config. It returned `ready_unreviewed` and
  wrote `scene_context.json`, `source_frame.jpg`, and `mapping_status.json` in
  the documented locations. The acceptance command uses the same function with
  the real Ollama provider.
- The documented SQLite query executed against an `AlertSink`-initialized
  database using the current event columns.
- `SceneContextStore.approve()` and `ConsoleBackend.event_clip()` signatures
  match the documented approval and replay actions.
- Every Markdown link and referenced repository test/tool/config path exists.
- `git diff --check` passed.

No broad test suite was rerun because this round changes documentation only.

## Fix Round 2

Review found that production retained gate log lines but no structured
concealment candidate record, so candidate recall, ownership, delay, and pose
throughput could not be recovered after a real run.

Runtime changes:

- Added defaulted `CandidateAlert.metadata` and copied the complete
  `RawEvent.extra` dictionary into it at the simple-rule boundary. The queued
  payload already retains the `CandidateAlert`, so destination, score,
  components, limited status, associated bag, and reasons now survive through
  the gate.
- Added append-only `<output-dir>/concealment_audit.jsonl`. `AlertSink.handle()`
  writes one self-contained record for every queued concealment gate callback,
  including confirmed, rejected, errored, and missing-result/unverified
  outcomes. Each record carries video timestamp, track, detector measurements,
  queue/verdict timestamps, gate latency, verdict details, and prompt version.
- Kept delivery semantics unchanged: rejected and unverified high-priority
  concealment candidates are audit-only; only confirmed results enter
  `events.db`, notify an operator, and appear in the UI.
- Timed the actual pose extraction call as `pose_infer`, keyed by camera, on the
  existing performance board. The final `perf_report.json` now exposes count,
  invocation rate, per-unit latency, and percentiles without a second report.

TDD evidence:

- RED command: focused contract, sink, workflow, and heavy-model tests.
  Result: `8 failed, 24 passed`. Failures were the expected missing
  `CandidateAlert.metadata`, absent `concealment_audit.jsonl`, and absent
  `pose_infer` stage.
- GREEN command with the same tests: `32 passed, 14 warnings in 3.25s`.
- Expanded documented focused command:
  `96 passed, 14 warnings in 3.89s`.

Documentation and command verification:

- Updated `docs/PROJECT_CONTEXT.md` to describe the actual metadata, audit,
  delivery, and performance boundaries.
- Updated `docs/CHI_PILOT_TESTING.md` to use the JSONL audit as candidate/gate
  truth, SQLite/UI as confirmed-delivery truth, and `perf_report.json` as pose
  performance truth.
- Added an exact portable extractor that validates required audit fields,
  derives positive detection delay from the matrix's fixed action start,
  records all candidate rows and the maximum score, reads
  `stages.pose_infer[CASE_ID]`, and writes `scenario10_result.json`.
- Executed that exact extractor against production-shaped sink/performance
  artifacts. It recovered timestamp, track, score, components, limited status,
  associated bag, rejected verdict, 0.406-second gate latency, 2.25-second
  labeled detection delay, and a non-null pose invocation rate.
- Rechecked `cvti.serving.pipeline --help`, `cvti.app.shell --help`, and
  `tools/prompt_regression.py run --help`; all documented options remain valid.
- All 22 Bash blocks in `docs/CHI_PILOT_TESTING.md` pass `bash -n`. Every
  Markdown link and referenced config/tool/test/runtime path exists.
- `git diff --check` passes.

Round-2 self-review:

- Kept `question` in its original positional slot in `CandidateAlert`; the new
  field is appended with a default, preserving callers outside this checkout.
- Confirmed the JSONL write is serialized with the sink's existing lock and is
  best-effort, so audit I/O cannot terminate gate processing.
- Confirmed detector timestamps remain clip-relative while queue/verdict timing
  remains wall-clock; the operator extractor combines only the clip-relative
  candidate timestamp with the predeclared clip label.
- Confirmed the audit is concealment-only and does not alter other detector
  artifacts or the later motion-document scope.

Remaining concerns are unchanged: the seven Chi clips have not yet been run,
and the changed prompt remains unmeasured while the frozen golden corpus is
unavailable. The audit records candidates admitted to the deduplicated queue;
the queue's separate `deduped` count remains the source for suppressed repeats.
