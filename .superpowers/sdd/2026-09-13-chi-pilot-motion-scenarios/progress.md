# SDD ledger — plan: docs/superpowers/plans/2026-09-13-chi-pilot-motion-scenarios.md

## Baseline

- Worktree: `/Users/macbook/Desktop/CV Threat Intelligence/.worktrees/chi-pilot-motion-concealment`
- Branch: `feat/chi-pilot-motion-concealment`
- Motion implementation base: `fdc3f6a` after the concealment final fix wave.
- Relevant backend baseline: 124 passed, 0 failed (14 existing dependency deprecation warnings).
- Frontend baseline: 21 test files and 107 tests passed after `npm ci`.

## Preflight Scan

| Scope | Producer / test intent | Consumer / implementation intent | Finding |
| --- | --- | --- | --- |
| Task 1 self | Speed, jitter, hysteresis, expiry, persistence, and latch tests. | Focused motion state and simultaneous movement units. | Consistent. |
| Task 2 self | Config, zone filtering, aggregation, metadata, and duplicate suppression tests. | Integrates Task 1 after ByteTrack without making scenario 4 alert. | Consistent. |
| Task 3 self | Raw/annotated pixels, authentication, viewer lifecycle, and transport tests. | Publishes overlays on demand while keeping raw/WebRTC behavior. | Consistent. |
| Task 4 self | Default, persistence, override precedence, and monitoring continuity tests. | React controls request the stream variant without changing detection state. | Consistent. |
| Task 5 self | Positive/negative replay matrix and performance measurements. | Validates Tasks 1-4 end to end. | Consistent. |
| Tasks 1 + 2 | Task 1 produces `PersonMotion` and a latched aggregate event. | Task 2 consumes both in `PerCameraState` and emits one scenario-5 `RawEvent`. | Compatible; Task 2 depends on Task 1. |
| Tasks 1 + 5 | Task 1 adds focused motion tests. | Task 5 reruns them in the acceptance suite. | Compatible. |
| Tasks 2 + 3 | Task 2 produces current semantic motion overlays. | Task 3 consumes overlays in raw/annotated publishing. | Compatible; Task 3 depends on Task 2. |
| Tasks 2 + 5 | Task 2 adds camera/rules/gate integration. | Task 5 validates zone behavior, candidates, and no scenario-4 alerts. | Compatible. |
| Tasks 3 + 4 | Task 3 exposes `camera_stream(camera_id, tracking)`. | Task 4 resolves global/per-camera preference and invokes that contract. | Compatible; Task 4 depends on Task 3. |
| Tasks 3 + 5 | Task 3 adds publisher/API/transport tests. | Task 5 includes them in focused and full regressions. | Compatible. |
| Tasks 4 + 5 | Task 4 adds frontend workflow tests and controls. | Task 5 validates both hidden and shown overlay throughput. | Compatible. |

## Rulings

- None at preflight.
- Carry-forward Ruling: the concealment audit emergency-purge filesystem-reclamation issue remains open for the combined branch final review; it does not alter the person-motion interfaces in Tasks 1-5 — cost if wrong: a low-disk purge may remove audit rows without immediately shrinking SQLite storage.
- Carry-forward Ruling: scenario 10 real-clip acceptance is explicitly unmeasured until Chi footage and the frozen prompt corpus are available — cost if wrong: automated correctness may be mistaken for measured field performance.

## Progress

- Task 1: minor (deferred): add an explicit short-dropout retention test contrasting return before and after expiry.
- Task 1: minor (deferred): validate EMA, threshold ordering, durations, expiry, and aggregate minimum-person constructor values.
- Task 1: complete (commits `fdc3f6a..ffe1ba9`, spec compliant and review approved; 2 deferred minors remain for final review).
- Task 2: fix round 1/5 (3 addressed, 1 open: boolean values remain accepted for float-valued movement settings; commits `b85dc9b..7c94d6c`).
- Task 2: fix round 2/5 (1 addressed, 0 open: all float-valued movement settings reject booleans with camera-specific errors; commits `7c94d6c..4197401`).
- Task 2: complete (commits `ffe1ba9..4197401`, spec compliant and review clean).
- Task 3: fix round 1/5 (2 addressed, 0 open: tracking cache invalidates across sessions and raw/annotated frames commit coherently; commits `24a7ec1..13f9703`).
- Task 3: minor (deferred): synthetic overlapping old/new publisher completion is not directly tested; production publishers are serialized per camera.
- Task 3: complete (commits `4197401..13f9703`, spec compliant and review clean; 1 deferred minor remains for final review).
- Task 4: fix round 1/5 (1 addressed, 1 open: exact target-only request assertions missing for two per-camera transitions; commits `0da30c2..9773d61`).
- Task 4: fix round 2/5 (1 addressed, 0 open: every per-camera transition now asserts an exact target-only request ledger; commits `9773d61..a451cb8`).
- Task 4: complete (commits `13f9703..a451cb8`, spec compliant and review clean; 2 unrelated Playwright failures remain tied to absent demo media assets).
- Task 5: fix round 1/5 (5 original findings addressed, 3 new Important scorer-integrity findings open: out-of-window false positives, observation coverage, and paired/mode-qualified performance evidence; commits `be40ced..48c964b`).
- Task 5: fix round 2/3 (3 scorer-integrity findings addressed, 2 new Important acceptance blockers open: rate arithmetic and actual unique MJPEG evidence; commits `48c964b..2d5dcaa`).
- Task 5: fix round 3/3 (rate arithmetic and unique structured MJPEG evidence addressed; final-round commit follows `2d5dcaa`).
- Task 5: fix round 4 (scoped regression: performance `sample_count` now binds to frame `engine.units`, while `observation_count` separately binds to batch `engine.count`; commit follows `9b06251`).
- Task 5: complete (commit `2c5bd9b`; scoped review clean. Reproducible scorer now rejects out-of-window false positives, incomplete person/frame coverage, inconsistent throughput arithmetic, malformed MJPEG captures, and reused capture evidence).
- Task 5: empirical Chi acceptance remains explicitly unmeasured pending controlled labeled clips, a complete scenario-5 candidate audit, and three real hidden/shown capture pairs.
- Combined final fix wave: sub-expiry observation dropouts preserve tracker
  output and the same scenario-5 latch; expiry and observed below-threshold
  motion reset it. Unit and pipeline tests cover return before/after expiry.
- Combined final fix wave: scenario-4 overlays are green, active scenario-5
  constituents are amber even on scenario-5-only cameras, confirmed/associated
  tracks use the alert colour, and aggregate `track_ids` reach publisher state.
- Combined final fix wave: every generated scenario-5 candidate receives a
  retained scorer-compatible `motion_candidate_audit` lifecycle row covering
  admission, deduplication, capacity drop, gate/error, and persistence outcomes.
- Combined final fix wave: motion constructors validate finite and ordered
  values, detector flags require booleans, and stale overlapping publisher
  completion cannot replace a newer same-session publication.
- Final constrained verification: focused backend `262 passed, 14 warnings in
  37.85s`; final edited aggregate-ID regression `1 passed, 14 warnings in
  0.98s`; frontend `22` files / `118` tests and production build passed; prompt
  check passed with metrics **UNMEASURED**. The user-stopped broad run reached
  `1418 passed, 8 skipped, 1 failed, 15 warnings in 162.18s`; the sole observed
  failure was Ultralytics 8.4.64 versus pinned 8.4.35.
- Empirical Chi accuracy remains unmeasured pending controlled rights-cleared
  clips, the frozen prompt corpus, and three real hidden/shown capture pairs.
