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

- **Logging, with rotation and per-module attribution** (`cvti.logging_setup`).
  `get_logger(__name__)` everywhere; a rotating file handler (10 MB × 5) writing
  to `<output_dir>/logs/`, level from `ARGUS_LOG_LEVEL`. The engine and the app
  write separate files — they share an output directory, and two processes
  rotating one handle loses records on POSIX and fails outright on Windows. In
  a packaged build logs go to the per-user application-support directory, since
  the working directory may not be writable and there is no terminal to fall
  back on.
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
