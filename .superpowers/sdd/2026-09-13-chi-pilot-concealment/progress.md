# SDD ledger — plan: docs/superpowers/plans/2026-09-13-chi-pilot-concealment.md

## Baseline

- Worktree: `/Users/macbook/Desktop/CV Threat Intelligence/.worktrees/chi-pilot-motion-concealment`
- Branch: `feat/chi-pilot-motion-concealment`
- Initial implementation base: `cc58839`
- Sandbox run: 103 passed; 21 failed solely because localhost socket binding and the normal Argus application-support backup path were denied.
- Permitted rerun: 124 passed, 0 failed (14 existing dependency deprecation warnings).

## Preflight Scan

| Scope | Producer / test intent | Consumer / implementation intent | Finding |
| --- | --- | --- | --- |
| Task 1 self | Lifecycle tests require pose gaps to preserve state and stale tracks to expire. | `expire()` owns time-based cleanup; `update()` runs only on sampled pose frames. | Consistent. |
| Task 2 self | Extraction, two-person association, and trolley exclusion tests. | Reuses COCO IDs 24/26/28 and adds track-specific bags without another inference pass. | Consistent. |
| Task 3 self | Metadata, normal-action, prompt, and evidence tests. | Event adapter and gate retain temporal cues while rejecting safe actions. | Consistent. |
| Task 4 self | Positive/negative pilot matrix and regression commands. | Documents and validates the implementation from Tasks 1-3. | Consistent. |
| Tasks 1 + 2 | Task 1 stabilizes per-track concealment history. | Task 2 adds per-track bag destinations to the same update lifecycle. | Compatible; Task 2 builds on Task 1. |
| Tasks 1 + 3 | Task 1 makes assessments survive production pose stride. | Task 3 forwards resulting assessment fields through the event adapter. | Compatible; no schema dependency is removed. |
| Tasks 1 + 4 | Task 1 adds stride and expiry regression coverage. | Task 4 reruns those tests in pilot validation. | Compatible. |
| Tasks 2 + 3 | Task 2 produces `associated_bag` and component evidence. | Task 3 serializes those cues for TrueSight. | Compatible; Task 3 consumes the Task 2 field. |
| Tasks 2 + 4 | Task 2 activates bagging in multi-camera serving. | Task 4 requires one true bag sequence end to end. | Compatible. |
| Tasks 3 + 4 | Task 3 defines verification behavior and evidence. | Task 4 measures post-gate outcomes and safe negatives. | Compatible. |

## Rulings

- Ruling: The legacy global `bag_bboxes` argument remains supported, while production uses `bag_bboxes_by_track`; track-specific input takes precedence when both are supplied — this satisfies backward compatibility and prevents one shopper's bag grounding another shopper — if wrong, a legacy caller that intentionally combines both inputs could observe different bag selection.
- Ruling: The `1.5`-second default grace is lifecycle expiry, not a concealment score/persistence threshold, so it does not require pilot evidence before this repair — it is configurable and preserves heavy-stride observations — if wrong, an unusually slow pose cadence could expire a valid sequence and require measured adjustment.
- Ruling: Exclusive ownership is guaranteed in the production `bag_bboxes_by_track` path; the backward-compatible global `bag_bboxes` path cannot infer ownership because it has no per-track association input — preserving its behavior avoids breaking standalone callers — if wrong, a legacy multi-person caller may still need migration to the new track-specific argument.
- Ruling: Task 4 may make the smallest tested runtime-contract change needed to expose concealment candidate metadata and outcomes in a structured audit artifact — the spec requires measurable candidate recall, ownership, delay, and throughput, and documentation cannot satisfy that while the pipeline discards the fields — if wrong, this broadens a documentation task into a shared contract change that needs extra regression coverage.

## Progress

- Task 1: fix round 1/5 (2 addressed, 0 open; commits `824ab29..d8c7cee`).
- Task 1: complete (commits `cc58839..d8c7cee`, review clean).
- Task 2: fix round 1/5 (2 addressed, 0 open; commits `49983d6..8ae7f49`).
- Task 2: complete (commits `d8c7cee..8ae7f49`, review clean).
- Task 3: minor (deferred): mutate source assessment collections after conversion and assert copied event/alert metadata remains unchanged.
- Task 3: fix round 1/5 (2 original findings addressed, 1 new Important open: partial replay mislabeled measured; commits `dbd2064..8b70dd0`).
- Task 3: minor (deferred): update the pre-existing serving-pipeline comment that says derived local providers receive one frame; current behavior is two generally and four for concealment.
- Task 3: fix round 2/5 (limited replay behavior addressed, 1 Important open: full-length replay with errors still mislabeled measured; commits `8b70dd0..713f127`).
- Task 3: fix round 3/5 (1 addressed, 0 open: errored full-corpus replays cannot become measurements; commits `713f127..0da99f9`).
- Task 3: complete (commits `8ae7f49..0da99f9`, review clean; 2 deferred minors remain for final review).
- Task 4: fix round 1/5 (portable workflow fixed, 1 Important open: required candidate/performance fields are not production-observable; commits `2cf3de3..bd91d8a`).
- Task 4: fix round 2/5 (structured audit and pose performance added, 1 Important open: extractor does not sort/filter to labeled action window; commits `bd91d8a..190e4f5`).
- Task 4: minor (deferred): add direct concurrency and serialization-failure tests for best-effort concealment audit writes.
- Task 4: minor (deferred): move the lazy performance-board import outside the timed `pose_infer` region so first-call latency excludes import cost.
- Task 4: fix round 3/5 (acceptance extractor now sorts and filters by labeled action window; 1 addressed, 0 open; commits `190e4f5..1f5e62d`).
- Task 4: complete (commits `0da99f9..1f5e62d`, review clean; 2 deferred minors remain for final review).
- Final review: revise (0 Critical, 5 Important, 5 Minor). Required fixes: exclusive bag ownership across all supported runtimes and across frames; pre-admission candidate audit; retained audit storage. Real-clip acceptance remains external evidence work. Deferred minors were returned for triage in the final fix wave.
- Final fix wave: commits `1f5e62d..fdc3f6a`; 187 focused tests passed; exact-head full suite reported 1416 passed, 8 skipped, with 1 unrelated local Ultralytics pin mismatch.
- Final re-review: 9/10 original findings addressed; no Critical findings. Real Chi footage and the frozen prompt corpus remain unavailable.
- Final review parked — real-clip acceptance evidence — Ruling: code/test readiness can proceed, but scenario 10 must not be called pilot-accepted until the documented seven-case protocol is run on supplied Chi footage; cost if wrong: real-world precision/recall may remain below the target despite passing automated tests.
- Final review parked — emergency SQLite deletion may not immediately reclaim filesystem bytes — Ruling: this is real but does not bear on the motion interfaces built next; carry it into the combined branch final review and fix before merge rather than reopening the completed one-wave concealment review; cost if wrong: low-disk emergency purge may erase audit history before reclaiming enough space.
- Final review minor (external evidence): geometry-based bag identity still needs real-footage characterization for detector-box instability.
- Combined final fix wave: emergency audit deletion commits, checkpoints and
  truncates WAL state, VACUUMs SQLite, and verifies reclamation before it may
  proceed to event evidence. Reclamation failure stops the emergency purge;
  tests cover real file shrink/freelist exit and evidence preservation.
- Combined final fix wave: retention includes `motion_candidate_audit`, whose
  runtime lifecycle records generated, queue-admission, gate/error, and
  persistence outcomes in the scorer-compatible schema.
- Final constrained verification: focused backend `262 passed, 14 warnings in
  37.85s`; final edited aggregate-ID regression `1 passed, 14 warnings in
  0.98s`; frontend `22` files / `118` tests and production build passed; prompt
  check passed with metrics **UNMEASURED**. The user-stopped broad run reached
  `1418 passed, 8 skipped, 1 failed, 15 warnings in 162.18s`; the sole observed
  failure was Ultralytics 8.4.64 versus pinned 8.4.35.
- Residuals: no measured Chi accuracy; controlled rights-cleared clips and the
  frozen corpus remain unavailable, geometry-based bag identity remains an
  empirical risk, and the shared Ultralytics installation remains off-pin.
