# Changelog

Notable changes to Argus. Newest first.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html), and a
`v*` tag is what builds and publishes installers.

Every measured figure quoted here is stated with the sample size it came from —
the full set, with confidence intervals, lives in [docs/NUMBERS.md](docs/NUMBERS.md).

---

## [Unreleased]

Sprint 0 of the [three-week plan](docs/SPRINT_PLAN.md): make the demo impossible
to fail silently, and the claims survivable under scrutiny.

### Added

- **One installer that contains the product** (EP-05-T1). The bundle now ships
  BOTH executables — the operator console and `argus-engine`, the full
  detection pipeline (YOLO + VideoMAE + the TrueSight gate) — plus the Ollama
  runtime, so "Start monitoring" works on a machine with no Python and no
  Ollama. The ~3.3 GB verifier model is the one thing not inside: it downloads
  on first run, in-app, with progress, and an interrupted download resumes.
  Replaces the viewer-only build the audit described as "installers that
  cannot detect anything on their own" (USR-01). Artifacts, bundle and data
  directory now say **Argus** (an existing CVTI data directory is migrated in
  place, events and logs intact).

- **Mobile response view** (EP-06-T3). One page served by the engine over the
  site's own network — no app, no cloud. Telegram alerts now deep-link straight
  to the specific alert; from the phone a guard sees the frames and the model's
  reasoning, claims with **"I'm on it"**, and resolves Real / False alarm with a
  handover note — through the same state machine and audit trail as the desktop.
  This is the first Argus surface that leaves localhost, so the security is the
  point: every route requires a session (same accounts and lockout as the
  console), HttpOnly+SameSite cookie, CSRF tokens on actions, roles enforced,
  and *"no unauthenticated route exists"* is a named test. Unverified alerts
  render with the dashed treatment and *"What the detector claimed — NOT
  verified"*.
- **Incident record as PDF, and a shift handover** (EP-06-T2). Every alert
  exports as a one-file incident record — metadata, the model's reasoning, the
  responder's conclusion and note, and the evidence frames embedded — written by
  a dependency-free PDF writer, because the record is what a manager reviews and
  what goes to an insurer or the police. An **open** incident's record says
  *OPEN — not yet concluded* rather than faking a conclusion. The Alerts screen
  gains a **Handover** view: what fired, what was resolved by whom (with notes),
  and — loudest — what is still open; open items are *not* windowed, and
  anything older than the window is flagged **carried over** with its age.
  Retention now holds every unresolved incident: an alert a guard claimed but
  never concluded previously became purgeable the moment it was acknowledged.
- **Alert ownership and a real state machine** (EP-06-T1). Alerts move
  `NEW → ACKNOWLEDGED(by whom, when) → RESOLVED(outcome, note)`, enforced in one
  place. **Claiming an alert shows your name to every operator** — with two
  guards on shift, both responding or neither was the audit's largest product
  gap. A second claim is refused *by name*; a resolved alert is finished;
  resolving an unclaimed alert claims it in the same breath, both transitions
  audit-logged. Outcomes are `real` / `false_alarm` / `inconclusive` — the
  honest third option that feeds nothing, because forcing a binary answer on a
  genuinely unclear clip poisons the training data. The legacy `review` column
  is maintained as a projection, so the feedback loop and Value screen are
  unchanged. The Alerts screen gains the queue-as-a-number strip, an "I'm on
  it" claim button, owner chips on rows, and a handover note at resolution.
- **Opt-in heartbeat and a sites dashboard** (EP-04-T2). Off by default —
  nothing is sent anywhere until a site owner enters a monitoring URL and key.
  The payload is the health document copied through a **whitelist**, so a field
  added to `/health` later cannot leak by omission; the schema is public
  ([docs/HEARTBEAT.md](docs/HEARTBEAT.md)) and every payload sent is written to
  `heartbeat_last.json` and viewable in System → Remote monitoring — "what
  leaves my machine?" answered by looking, not trusting. Outbound-only POST, so
  it works through ordinary NAT. The receiver
  (`tools/heartbeat_receiver.py`, stdlib+SQLite, one file) shows every site
  worst-first, flags a site **MISSED** after ~2.5 silent intervals regardless of
  what it last claimed, and sends a Telegram message once per transition —
  missed, degraded, critical, recovered — not once per check.
- **`/health`** (EP-04-T1): one authenticated endpoint answering "is this site
  OK right now?" — status + named reasons, per-camera link state with
  last-frame age, gate reachability and median verify latency, disk, memory,
  per-component error counters, uptime. Served by the engine's authenticated
  frame server, written to `gate_health.json` for the System panel, and
  structurally free of frames or event content — it becomes the heartbeat
  payload. Verified by killing a real camera and a real gate mid-run and
  watching the status change.
- **Daily proof of life** (EP-04-T4): a scheduled self-test that exercises a
  real frame → the real gate → a real notification and raises an alert when any
  hop fails, plus a daily "all systems normal" message (on by default, per-site
  opt-out) so silence stops being the success signal.
- **Sign-in and first-run screens** (UI_SPEC §2.2). First run creates the owner
  account — nothing ships with a password, so there is nothing to change — then
  hands off to the existing setup wizard as steps 2–5. Sign-in shows the
  backend's refusal message verbatim (one message for wrong password and unknown
  user, on purpose). After sign-in the app routes to the role's landing surface,
  the nav footer shows who is signed in, and nav items the role cannot use are
  not rendered — courtesy only; the backend re-checks every call.
- **Identity, three roles, and an append-only audit trail** (`cvti/security/`).
  Local accounts with scrypt or PBKDF2 hashing, session timeout, lockout after
  five failures. No default account and no default password ship — the first
  owner is created at setup. `Owner` / `Operator` / `Installer` are enforced in
  the backend, not by hiding controls: an operator cannot disable a detector and
  an installer cannot read recorded incidents, both tested by calling past the
  interface. The audit log is append-only (SQLite triggers refuse `UPDATE` and
  `DELETE` even against direct SQL) and hash-chained, so a partial edit — the
  realistic attack — is detectable and names the row it started at.
- **Authenticated camera endpoints.** Both frame servers previously served live
  camera frames on every route with no auth and `Access-Control-Allow-Origin: *`.
  Every route now requires a per-run capability token, compared in constant time,
  and the wildcard CORS header is gone. This is the hole EP-06 would have exposed
  to the network.
- **Disk-encryption check** (FileVault / BitLocker / LUKS), reported at setup and
  in the System panel, with an honest `unknown` where it cannot be determined.
- **`SECURITY.md`** — the security model, the threat model including what is
  deliberately *not* defended, and a procurement answer sheet.
- **Logging, with rotation and per-module attribution** (`cvti.logging_setup`).
  `get_logger(__name__)` everywhere; a rotating file handler (10 MB × 5) writing
  to `<output_dir>/logs/`, level from `ARGUS_LOG_LEVEL`. The engine and the app
  write separate files — they share an output directory, and two processes
  rotating one handle loses records on POSIX and fails outright on Windows. In
  a packaged build logs go to the per-user application-support directory, since
  the working directory may not be writable and there is no terminal to fall
  back on.
- **Retention, purge and legal hold.** Evidence is deleted 30 days after
  recording by default (per-site configurable) — frames, clips and the database
  row together, files first so a failure can never leave personal data on disk
  that nothing references. Two categories deliberately outlive their expiry:
  anything an operator places on **legal hold**, and anything **not yet
  reviewed**, because deleting an open incident on a timer destroys the record
  while the question is still live. Both are counted and shown, so "why is this
  still here?" has an answer. Adds a disk warning at 85%, oldest-first emergency
  purge at 95% that still refuses to touch held evidence, an orphan sweep for
  evidence directories no record points to, and an evidence export so a customer
  can keep an incident past its expiry. Policy documented in
  [docs/DATA_RETENTION.md](docs/DATA_RETENTION.md).
- **Camera link state — `connected` / `reconnecting` / `offline`** — with
  time-in-state, exponential capped backoff, bounded attempt history, and an
  offline alert raised through normal routing after a configurable grace period.
  Recovery is announced too. The Cameras screen now shows observed link state
  where it previously showed a green "configured" dot regardless of whether the
  camera was reachable, and reports `unknown` when no engine is running rather
  than claiming coverage nobody is watching.
- **Per-component error counters** (`cvti.health`), shown in the System panel
  with frames processed, error count, error rate and last error per camera. A
  component failing on more than 1 in 10 attempts is flagged degraded. Logging
  is rate-limited — first few occurrences of each error type, then one in a
  hundred — so a detector throwing every frame cannot fill the disk with its own
  traceback while the counter still carries the true scale.
- **Download diagnostics** (System panel). Zips logs and a health snapshot —
  versions, disk, gate status, event *counts* — for support. It contains no
  camera images, no video, and no event rows: this is a surveillance product,
  and "send us your logs" must not quietly mean "send us your footage". The
  bundle's manifest states what is excluded so the sender can verify it rather
  than trust it.
- **Value screen.** The suppression counter reframed from an engineering metric
  into what a buyer decides on: incidents detected, false alarms prevented,
  duplicate alerts collapsed, attention-hours saved. Site-configurable rates
  (minutes to triage an alert, guard hourly cost, value of an incident) so the
  money figures are the customer's numbers — each stays hidden until a rate is
  entered. Every figure is a count of real rows and the screen shows the
  arithmetic behind it.
- **Suppression ledger** (`suppression_daily` table). Only confirmed alerts were
  ever persisted, so the "raw detectors would have shown you N alerts" claim had
  no evidence behind it once the engine restarted. The engine now writes daily
  shown/rejected/deduped/error counts as it verifies.
- **Gate health published to the UI.** The engine writes `gate_health.json` every
  3s; the System panel shows live verified/confirmed/rejected/queued/errors and
  the last gate error. Health older than 30s reads as "engine not running" — a
  green gate for a dead engine is worse than no gate at all.
- **Confidence intervals on every published rate.** `wilson_interval()` in
  `cvti.eval.metrics`; `docs/NUMBERS.md` and the eval report now print every
  precision/recall/FPR as estimate + n + 95% Wilson interval, generated rather
  than typed.
- **SHA-256 checksums on releases.** `SHA256SUMS.txt` attached to each release
  and the sums appended to the release body, with verification steps in the
  README.
- **CI test gate.** The suite runs on every push and PR; the installer build
  declares `needs: test`.

### Changed

- **Fire headline restated** as *"100% recall on 9 held-out positive clips
  (95% CI 70.1–100%)"*. The point estimate alone invited the first question a
  technical reader asks, and lost the room when the answer was nine.
- **The Value screen distinguishes "no false alarms" from "no measurement".**
  A database with incidents but no suppression ledger — the bundled playback
  demo, or any run predating the ledger — would have rendered *0 false alarms
  prevented, 0.0 hours saved*, which reads as the product doing nothing rather
  than as a measurement never taken.
- **Rejected and deduped alerts are counted apart.** Both cost an operator
  attention, but only a rejected one is a false alarm. Rolling them together
  would have inflated the headline claim several times over.
- **Documentation split into current and historical.** Eight superseded plans
  moved to `docs/archive/` with a "superseded by" table; `docs/README.md` indexes
  what remains. The root README described a 36-hour proof of concept and every
  quickstart command pointed at a `detector.py` that no longer exists — rewritten
  against what actually runs, with each command verified.

### Fixed

- **An empty red strip rendered at the top of every screen.** The mock-gate
  banner's `display:flex` overrode the `hidden` attribute's `display:none`, so
  the banner showed as a thin empty red bar whenever the gate was *not* mock.
  Caught during login-screen verification.
- **The fine-tuned VideoMAE model was silently broken.** The `print()` → logging
  conversion rewrote `print(..., file=sys.stderr)` as `emit(..., file=...)`, but
  `emit` takes `err=True`. Every inference raised `TypeError` inside a broad
  handler, so the headline model failed **249 times per run** while reporting
  only `[VideoAction error]` — and no `video_theft_candidate` alert ever fired.
  A static test now checks every `emit()` call site against the signature.
- **`cvti-detect` could not run offline.** Its `--gate-provider` choices were
  `mock/anthropic/openrouter`, so once the mock gate was refused the only
  remaining options sent frames off-device. It now offers every provider the
  gate supports and defaults to on-device `ollama`.
- **One failing camera no longer stops the other five.** `process()` guarded its
  detector section, but tracking, zones, rule evaluation and evidence selection
  sat outside it — a failure there propagated out and stopped every camera. The
  comment inside promised one bad detector could not kill the camera loop; it
  now does.
- **68 broad `except Exception` handlers all leave a record**, with the
  traceback. 22 swallowed entirely. Catching is not handling: without this,
  "this detector correctly found nothing" and "this detector has thrown on every
  frame for a week" are the same silence. Exemptions must carry a `SILENT-OK`
  comment saying why, and a test enforces it.
- **A dead gate reported as healthy.** Fail-visible turned transport failures
  into UNVERIFIED *results*, which the gate pool counted as successful
  verifications — so `/health` said `reachable=true, verified=8` while every
  verdict was "could not decide". Unverified verdicts now count separately and
  drive reachability; caught by the kill-the-gate acceptance run.
- **A gate that cannot decide no longer reports "safe".** `_parse_response`
  returned `confirmed=False` on any exception — the same value it returns when
  TrueSight examines a frame and rejects it. A fire during an Ollama restart was
  indistinguishable from a fire the model looked at and dismissed, and it was
  dropped with nothing on screen. `VerificationResult.error` now separates *no
  verdict* from *a verdict of no*, and the live path defaults to fail-visible:
  the alert reaches the operator marked **UNVERIFIED — TrueSight could not
  decide**, is stored flagged, and is excluded from the incident count so the
  product is not credited for work it did not do. Configurable, because a
  low-stakes site drowning in unverified alerts may reasonably choose otherwise.
- **The mock gate can no longer run unannounced.** `_mock_response()` returns
  `confirmed=True` unconditionally; any config path selecting it would have
  passed every candidate and inverted the product with nothing on screen saying
  so. Every engine entrypoint now refuses to start on `provider="mock"` unless
  `ARGUS_ALLOW_MOCK_GATE=1` is set, and shows a permanent red banner above every
  screen when it is. Constructing a `VerificationGate` directly is unaffected, so
  tests and the eval harness still use it freely.
- **A red test suite can no longer ship an installer.** 32 test files that a
  release build never consulted; tagging a version with failing tests produced
  three signed installers with the bug baked in.
- **Baseline rules are a safety net, not a second opinion** — they no longer
  fire alongside a configured rule for the same event.

---

## [0.9.0] — 2026-08-18

First tagged release: installers for macOS, Windows and Linux published from CI.

### Added

- **Two-stage evaluation harness** (`python -m cvti.eval`) measuring the same
  metrics before and after verification — the drop in false positives with
  recall held is the value the gate adds. Fire measured at 90% → 6.7% false
  alarms with no fires missed (n=39 clips).
- **Gate sensitivity presets** (`sensitive` / `balanced` / `strict`) with the
  cost of each option measured rather than asserted.
- **Alert routing and escalation** by severity, camera and time of day, with
  re-notification of anything left unacknowledged.
- **Watches** — describe a subject in plain English and Argus follows them.
- **Subject bounding boxes on evidence**, so an alert shows *who*, not just
  where.
- **Memory guard** that sheds load deliberately under pressure instead of
  swapping, in a considered order, never dropping the last camera.
- **Feedback loop** — operator labels drive calibration, chronically wrong rules
  stop paging, and every label becomes training data.
- **Per-detector toggles** grouped Security / Safety, and a feed switcher between
  demo clips and live cameras.

### Changed

- **Frames are published by the engine** rather than decoded a second time by the
  UI — decode was the dominant per-camera cost, and the live boxes come free.
- **Gate worker count derives from camera count**, cutting median alert latency
  from 46.5s to 28s at no extra memory cost.

### Fixed

- One incident now produces one alert, always pointing at a subject.
- The gate judges the frames instead of rubber-stamping the detector.
- The eval harness refuses to report numbers when the gate is unavailable, and
  keys checkpoints by gate so runs cannot contaminate each other.
- Detectors that nothing listens to are no longer silently discarded.

---

## Before 0.9.0

Proof-of-concept and build-out, from the first detector through the retail
pipeline, the verification gate, the multi-camera serving engine and the operator
console. Not itemised here — see `git log` and
[docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md), which records the decisions
and why they were made.

[Unreleased]: https://github.com/Ayo-Cyber/cv-threat-intelligence/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/Ayo-Cyber/cv-threat-intelligence/releases/tag/v0.9.0
