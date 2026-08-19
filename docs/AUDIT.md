# Argus — Holistic System Audit & Prioritised Remediation Backlog

**Audit date:** 19 August 2026
**Auditor perspectives:** Senior Software Engineer · Senior Product Designer · Senior Systems Architect · Senior AI/ML Engineer · Investor · End Users
**Prioritisation by:** Senior Product Manager
**System version:** `v0.9.0` (released), HEAD = `77fa91e`
**Repo state at audit:** working tree clean, 0 unpushed commits

---

## 0. How to read this document

This is a **working document**, not a report card. Every concern below has:

| Field | Meaning |
|---|---|
| **ID** | Stable reference (`SWE-01`, `ARCH-03`…). Use it in commits and PRs. |
| **Severity** | `CRITICAL` / `HIGH` / `MEDIUM` / `LOW` — impact if left unfixed |
| **Evidence** | The measured fact. Every claim here was verified against the repo on audit day, not recalled. |
| **Why it matters** | The consequence in business or user terms, not engineering terms |
| **Definition of Done** | What "fixed" concretely means. No ambiguity. |
| **Points** | Story points (Fibonacci) |

Section 9 consolidates everything into a single ordered backlog. **If you only read one section, read Section 9.**

### Story point scale (calibrated for one full-time developer)

| Points | Effort | Meaning |
|---|---|---|
| 1 | 1–2 hours | Config change, one-line fix, doc |
| 2 | Half a day | Contained change, one file, tests included |
| 3 | 1 day | New small module or a change touching 2–3 files |
| 5 | 2 days | New subsystem, or a change with migration/rollback concerns |
| 8 | 3–4 days | Cross-cutting change, multiple subsystems, needs design |
| 13 | 1–2 weeks | New capability with UI + backend + tests |
| 21 | 3–4 weeks | Architectural change or externally-dependent work (data collection, pilots) |

**Total backlog: 51 concerns, 389 story points ≈ 26 developer-weeks (~6 months for one developer).** That is the honest number. Section 9 explains what to cut.

---

## 1. Executive summary

**The one-line verdict: the engineering *judgement* in this system is better than the *product*, and the *evaluation discipline* is better than the *models*.**

Argus has done the hard, unglamorous thing most computer-vision startups skip: it measures itself honestly on held-out data and publishes results that are unflattering. That is a genuine and rare asset. It has simultaneously skipped several things most production systems do first — authentication, structured logging, data retention, tests in CI.

That asymmetry is the story of this audit. It is also, encouragingly, the *easier* asymmetry to have. Rigour is hard to retrofit; auth is not.

### What is genuinely strong

1. **Measured, held-out results with honest deltas.** Fire detection: raw detectors produce a 90% false-alarm rate; TrueSight cuts it to 6.7% while missing zero fires. That is a crisp, memorable, defensible proof of the core thesis.
2. **Zero TODO/FIXME/HACK markers** across 15,470 lines. Unusual discipline.
3. **All 56 dependencies pinned** to exact versions. Reproducible builds.
4. **Tests encode reasoning, not just behaviour.** `tests/test_memory_guard.py` asserts *"blind is worse than degraded"* — the policy intent is captured, so a future refactor can't silently invert it.
5. **The two-process split** (engine ↔ app) is the correct architectural call. A hung UI cannot stall detection.
6. **The eval harness refuses to lie.** `GateUnavailable` + `preflight()` exist specifically because an earlier run reported "100% suppressed" when Ollama was simply down.

### What will hurt, in order

1. **No authentication, no audit log, no encryption, no data retention.** Four separate hard blockers for any enterprise or regulated sale — and the retention gap is a live GDPR/NDPR exposure, because this system stores identifiable CCTV footage with no deletion path.
2. **260 `print()` calls and zero `logging` imports.** When a customer's box misbehaves you have no diagnosis path.
3. **CI builds and ships installers but never runs the 32 test files.** You can currently release a red build.
4. **Seven of ten detectors are unmeasured.** The product markets ten capabilities and can evidence three.
5. **A malformed VLM response is silently counted as "not a threat"** in the live path, with no counter and no alarm.
6. **Zero pilots, zero real-site hours.** Every number in this system comes from public video clips.

### The three-day reality (audit day → investor demo, Sat 22 Aug)

You have **3 days**. Nothing in this document should be interpreted as "fix it all before Saturday." Section 9.4 identifies **11 story points of pre-demo work** — the items that either (a) protect you from an on-stage failure, or (b) neutralise a question an investor will actually ask. Everything else waits until after.

---

## 2. Evidence base

Every figure in this audit was measured on audit day. Reproduce with the commands below.

### 2.1 Codebase shape

| Metric | Value |
|---|---|
| Production Python (`cvti/`) | **15,470 LOC** across **79 modules** |
| Test code (`tests/`) | **3,380 LOC** across **32 test files** |
| Test-to-source ratio | **21.8%** |
| UI | single **1,429-line** `index.html` |
| Configs | 36 JSON files |
| Direct dependencies | **56**, all pinned (`==`) |
| Heavy dependencies | 5 (torch, ultralytics, transformers, opencv, PyQt6) |

### 2.2 Complexity hotspots

| Module | LOC | Note |
|---|---|---|
| `cvti/detector/core.py` | **2,189** | Monolith — pose, weapons, violence, theft, concealment |
| `cvti/scene/agent_mapper.py` | 872 | |
| `cvti/app/console_backend.py` | 765 | Qt bridge |
| `cvti/training/eval.py` | 671 | |
| `cvti/verification/gate.py` | 588 | The VLM gate |
| `cvti/serving/alert_sink.py` | 511 | |
| `cvti/serving/camera.py` | 480 | Per-camera hot path |
| `cvti/retail/concealment.py` | 470 | |
| `cvti/serving/pipeline.py` | 438 | |

### 2.3 Debt markers

| Marker | Count | Reading |
|---|---|---|
| `TODO` / `FIXME` / `HACK` / `XXX` | **0** | Excellent |
| Bare `except:` | **0** | Excellent |
| Broad `except Exception` | **64** | Deliberate, but see `SWE-02` |
| `# noqa` suppressions | 63 | Mostly paired with the above |
| `print(` calls | **260** | See `SWE-01` |
| Files importing `logging` | **0** | See `SWE-01` |

### 2.4 Security & compliance grep (all zero)

```
auth / login / session / jwt / bcrypt ....... 0 files
audit log .................................. 0 files
retention / purge / prune / max_age_days ... 0 occurrences
encryption (encrypt / Fernet / AES) ........ 0 files
site_id / tenant / organization ............ 0 occurrences
health or metrics endpoint ................. none
secrets committed in configs ............... 0  ✅
```

### 2.5 Test coverage distribution (uneven)

| Package | LOC | Test files |
|---|---|---|
| `serving/` | 3,362 | 1 |
| `detector/` | 2,518 | 2 |
| `app/` | 2,304 | 1 |
| **`training/`** | **1,735** | **0** |
| `scene/` | 877 | 1 |
| `retail/` | 831 | 1 |
| `verification/` | 825 | 1 |
| `eval/` | 623 | 1 |
| `feedback/` | 552 | 1 |
| `cli/` | 374 | **0** |
| `rules/` | 289 | 1 |
| `pipelines/` | 262 | **0** |

### 2.6 Measured detection results (from `runs/eval/*/metrics.json`)

All figures are Stage 1 (raw detectors) → Stage 2 (TrueSight-confirmed) on held-out clips.

| Run | Clips | Stage | Precision | Recall | FPR | Alerts |
|---|---|---|---|---|---|---|
| **Theft — prompt v1** (`baseline-v1`) | 36 | raw | 37.5% | 100% | 55.6% | 201 |
| | | gate | 37.5% | 100% | 55.6% | **191** ⚠️ |
| **Theft — balanced** (`v2-tightened`, **current default**) | 36 | gate | 53.3% | 88.9% | 25.9% | 61 |
| **Theft — strict** (`v3-strict`) | 36 | gate | 63.6% | 77.8% | 14.8% | 28 |
| **Fire** (`fire-v1`, balanced) | 39 | raw | 25.0% | 100% | **90.0%** | 39 |
| | | gate | **81.8%** | **100%** | **6.7%** | 12 |
| **Crowd** (`crowd-v1`, balanced) | 38 | raw | 38.9% | 87.5% | 36.7% | 52 |
| | | gate | 60.0% | 75.0% | 13.3% | 14 |

⚠️ The v1 row is the **rubber-stamp incident**: the gate suppressed only 10 of 201 alerts (5%) and changed no metric. It is retained here deliberately — see `ML-02`.

### 2.7 Measured throughput (MacBook Pro, 18 GB, 5 cameras)

| Metric | Value |
|---|---|
| Sustained per-camera rate | 6.2 fps |
| Detection batch (5 cameras) | 163 ms |
| Gate latency, 1 worker | 46.5 s median |
| Gate latency, 2 workers | 28.0 s median |
| Gate latency, auto-scaled | **18–20 s** median |
| Cameras validated | **5** |

### 2.8 Disk

| Path | Size | Nature |
|---|---|---|
| `runs/` total | **1.6 GB** | |
| ├ `runs/video_finetune` | 658 MB | training checkpoints (not evidence) |
| ├ `runs/video_finetune_stratified` | 329 MB | training checkpoints |
| ├ `runs/classify` | 243 MB | training artefacts |
| ├ `runs/detect` | 175 MB | training artefacts |
| ├ `runs/serving` | 20 MB | **evidence** |
| └ `runs/live` | 13 MB | **evidence** |
| Video files on disk | 254 `.mp4` | |
| Frame stills on disk | 1,400 `.jpg` | |
| `events.db` (each) | 12 KB | 7 copies across repo, dist, packaging |

**Correction to an earlier verbal estimate:** the 1.6 GB is dominated by *training checkpoints*, not evidence. Evidence is currently ~33 MB. The retention concern (`ARCH-01`) stands on its own merits — there is genuinely zero purge logic — but do not quote 1.6 GB as an evidence-growth figure. It would not survive scrutiny.

### 2.9 CI

`.github/workflows/` contains exactly one workflow: `build-app.yml`. It builds macOS/Windows/Linux installers on `v*` tags. **It contains no `pytest`, `unittest`, or test invocation of any kind.**

### 2.10 Reproduce this evidence

```bash
find cvti -name '*.py' | xargs wc -l | tail -1
grep -rn 'except Exception' cvti --include='*.py' | wc -l
grep -rn 'print(' cvti --include='*.py' | wc -l
grep -rln 'import logging' cvti --include='*.py' | wc -l
grep -rln 'authenticate\|jwt\|bcrypt' cvti --include='*.py' | wc -l
grep -rn 'retention\|def purge\|max_age_days' cvti --include='*.py' | wc -l
grep -rn 'pytest\|unittest' .github/workflows/
du -sh runs/* | sort -rh | head
```

---

## 3. Perspective 1 — Senior Software Engineer

> *I am reading this codebase as someone who will be paged at 3am when it breaks, and who has to onboard the second engineer you hire.*

**Overall verdict: unusually disciplined for the stage, with a small number of liabilities that are cheap now and expensive in six months.**

Let me be fair before I'm critical. Zero TODO markers across 15,470 lines is not normal. Zero bare `except:` is not normal. Fully pinned dependencies at a pre-seed stage is not normal. Tests that assert *policy intent* (`test_never_sheds_the_last_camera`) rather than just behaviour are a sign of someone who has been burned before and learned. Whoever wrote this thinks about failure modes.

Now the problems.

---

### SWE-01 — No structured logging anywhere. 260 `print()` calls, 0 `logging` imports.
**Severity: CRITICAL** · **Points: 5**

**Evidence:** `grep -rn 'print(' cvti` → 260. `grep -rln 'import logging' cvti` → **0 files**.

**Why it matters.** This is the single most consequential engineering gap in the system, and it is invisible until the moment you need it most. Today you are the only user and you run the engine in a terminal you can see. The moment a co-founder runs the Windows build, or a pilot site runs it unattended overnight, `print()` gives you:

- No severity levels — you cannot filter noise from a genuine fault
- No timestamps — you cannot correlate a crash with an event
- No module attribution — you cannot tell which of 79 modules emitted a line
- No file output — in the packaged PyInstaller app, stdout goes nowhere the user can retrieve
- No rotation — if it *were* redirected to a file, it would grow unbounded

Concretely: a customer calls and says *"it stopped alerting last night."* Right now you cannot answer that question. You would have to reproduce it on your own machine. That is not a support process, and it is the kind of thing that turns one unhappy pilot into a lost reference customer.

There is a second-order cost. Because there is no logging, the 64 broad exception handlers (`SWE-02`) swallow failures *completely silently*. The two problems compound: the system is designed to survive component failure, but has no way to tell you a component failed. That is how the crowd detector emitted events into the void for two days and looked like "detection is broken."

**Definition of Done:**
- A `cvti/logging_setup.py` providing `get_logger(__name__)` with a rotating file handler (10 MB × 5 files) writing to `<output_dir>/logs/argus.log`, plus a console handler
- Log level controlled by `ARGUS_LOG_LEVEL` env var, default `INFO`
- All 260 `print()` calls in `cvti/` converted to the appropriate level (`debug`/`info`/`warning`/`error`). CLI user-facing output in `cvti/cli/` may remain `print()` — that is legitimately a UI, and should be explicitly exempted with a comment
- Every one of the 64 broad handlers logs at `warning` or `error` with `exc_info=True`
- A "Download diagnostics" button in the app that zips the log directory
- Test asserting the log file is created and that a raised exception inside a detector produces a log record

---

### SWE-02 — 64 broad `except Exception` handlers swallow failures silently.
**Severity: HIGH** · **Points: 3** (after `SWE-01` lands)

**Evidence:** 64 occurrences, 63 paired `# noqa`. Zero bare `except:` (good).

**Why it matters.** The *intent* is correct and I want to defend it: a single misbehaving detector must not kill the camera loop, and one bad clip must not sink an eval run. `cvti/eval/harness.py:179` even documents this — *"one bad clip must not sink the run."* That is right.

The *implementation* is wrong in one specific way: catching is not the same as handling. Right now a swallowed exception produces no record at all. The system degrades invisibly. You lose the ability to distinguish "this detector correctly found nothing" from "this detector has been throwing on every frame for a week."

This has already cost you real time. It is the mechanism behind the crowd-detection bug: events were being discarded, nothing complained, and the symptom presented as a detection-quality problem rather than a wiring problem. You spent debugging effort on the wrong layer.

**Definition of Done:**
- Every broad handler logs with `exc_info=True` (depends on `SWE-01`)
- A per-component error counter surfaced in the app's System panel: *"fire_smoke: 1,204 frames processed, 3 errors"*
- Any component exceeding an error rate threshold (say 10% of invocations) raises a visible degradation warning in the UI
- Handlers in genuinely hot per-frame paths rate-limit their logging (log first occurrence + every Nth) so a persistent failure cannot itself become a disk-filling incident

---

### SWE-03 — CI ships installers but never runs the tests.
**Severity: CRITICAL** · **Points: 1**

**Evidence:** `.github/workflows/` contains only `build-app.yml`. `grep -rn 'pytest\|unittest' .github/workflows/` → no matches.

**Why it matters.** You have 32 test files and 3,380 lines of test code that a release build never consults. Tagging `v1.0.0` on a commit with failing tests produces three signed installers with a bug baked in, distributed to a co-founder and potentially a pilot site. The tests exist; they are simply not wired to the one moment they matter most.

This is the highest value-to-effort item in the entire audit. It is roughly ten minutes of YAML.

**Definition of Done:**
- A `test` job in the workflow, running on every push and every PR, executing the full suite
- The `build` job declares `needs: test` so a red suite blocks installer creation
- Branch protection on `main` requiring the test job to pass
- Badge in `README.md`

---

### SWE-04 — `cvti/detector/core.py` is a 2,189-line monolith.
**Severity: HIGH** · **Points: 8**

**Evidence:** 2,189 LOC — 2.5× the next largest module, 14% of the entire production codebase in one file. It contains pose estimation, weapons, violence, theft, and concealment logic.

**Why it matters.** Three compounding costs:

1. **Every new detector touches this file.** You have shipped ten detectors and plan more. Each addition raises the chance of breaking an unrelated one, because they share module-level state and helpers.
2. **It is the merge-conflict epicentre.** The moment a second engineer joins, you will both be editing this file constantly.
3. **It resists testing.** `detector/` has 2,518 LOC and 2 test files. The size of the unit is *why*. You cannot test a 2,189-line module in pieces; you can only test it end-to-end, which is slow and gives poor failure localisation.

This is where your next serious bug will live, and it will be hard to find.

**Definition of Done:**
- Split into `cvti/detector/` submodules: `pose.py`, `weapons.py`, `violence.py`, `theft.py`, `concealment.py`, with shared helpers in `common.py`
- `core.py` retained as a thin re-exporting facade so no import site outside the package changes — this makes the refactor reviewable and revertible
- Each new submodule under 400 LOC
- At least one dedicated test file per submodule
- No behaviour change: the full eval suite reproduces the archived fire/crowd/theft numbers **exactly** before and after. This is non-negotiable and is what makes an 8-point refactor safe

---

### SWE-05 — `training/` (1,735 LOC), `cli/` (374), `pipelines/` (262) have zero test files.
**Severity: HIGH** · **Points: 5**

**Evidence:** See §2.5. 2,371 LOC — 15% of production code — with no direct test coverage.

**Why it matters.** `training/` is the most consequential of the three, because **your only trained model is produced by untested code.** The VideoMAE theft classifier — one of the three things you can actually evidence — comes out of `video_finetune.py` (348 LOC) and is evaluated by `eval.py` (671 LOC). If there is a subtle bug in the label mapping, the stratification, or the metric computation, every number you have quoted from that model is wrong, and nothing in your test suite would catch it.

That is a quiet but genuine credibility risk. You are presenting measured numbers to investors on Saturday. The measurement code has no tests.

`cli/` is the entry point every user touches. `pipelines/` is orchestration.

**Definition of Done:**
- `tests/test_training_eval.py`: metric computation verified against hand-computed confusion matrices; label mapping verified; stratified split verified to preserve class balance and to never leak a clip across train/test
- `tests/test_cli.py`: every subcommand parses its arguments and dispatches correctly (fast, no model loading)
- `tests/test_pipelines.py`: orchestration wiring
- These three packages reach at least parity with the repo-wide 21.8% ratio

---

### SWE-06 — A malformed VLM response is silently counted as "not a threat".
**Severity: HIGH** · **Points: 3**

**Evidence:** `cvti/verification/gate.py`, `_parse_response()` — the `except Exception` branch returns `VerificationResult(confirmed=False, confidence=0.0, reason=f"Gate parse error: {exc}")`.

**Why it matters.** This is a genuine safety defect and it is subtle enough that I want to be precise about it.

In the **eval harness**, gate failures are handled correctly and deliberately: `harness.py:_confirm()` counts consecutive errors and raises `GateUnavailable` after five, with an explicit comment — *"A gate error is NOT a rejection. Counting it as one would report 'TrueSight suppressed everything' — fake numbers that look real."* That reasoning is exactly right.

**But the live path does not have that protection.** In production, a VLM response that fails to parse — truncated output, a model update changing the response format, an Ollama hiccup — becomes `confirmed=False`, which is indistinguishable from *"TrueSight looked at this and decided it was not a threat."* A real fire could be dropped, and the operator would see nothing at all. No counter, no alarm, no log line (see `SWE-01`).

The fail-direction is also backwards for a safety system. When the verifier cannot render a verdict, the safe default is to surface the candidate to a human marked *unverified*, not to discard it silently.

A related latent risk sits alongside it: `_mock_response()` unconditionally returns `confirmed=True`. If a configuration path ever selects `provider="mock"` in a deployed build, every candidate passes the gate and the product's entire value proposition silently inverts — with no visible signal.

**Definition of Done:**
- Parse failures and transport failures are distinguished from genuine rejections via an explicit `error` field on `VerificationResult`
- Live path applies a configurable fail policy, defaulting to **fail-visible**: the alert reaches the operator flagged `UNVERIFIED — TrueSight could not decide`
- A rolling gate-error counter on the System panel, with a visible banner when the error rate exceeds a threshold
- The engine refuses to start with `provider="mock"` unless `ARGUS_ALLOW_MOCK_GATE=1` is explicitly set, and if allowed, shows a permanent red banner in the UI
- Tests covering: malformed JSON, truncated JSON, empty response, connection refused, and the mock-provider guard

---

### SWE-07 — The UI is a single 1,429-line HTML file.
**Severity: MEDIUM** · **Points: 5**

**Evidence:** `index.html`, 1,429 lines containing markup, styles, and behaviour for all ten navigation surfaces.

**Why it matters.** It is genuinely fine at this size — I would not have flagged it at 800 lines. At 1,429 it is at the edge, and the product roadmap (triage workflow, mobile, weekly reports) will push it past 3,000. At that point CSS collisions become the dominant bug class and every UI change carries regression risk across unrelated panels.

There is no urgency here. There *is* a right moment, and it is before the next major UI feature, not after.

**Definition of Done:**
- Split into per-panel templates plus shared CSS/JS, or adopt a lightweight component approach
- No panel file exceeds 300 lines
- The QWebChannel bridge surface is unchanged (explicit `@pyqtSlot` dispatch preserved)
- A smoke test that loads each panel and asserts the bridge responds

---

### SWE-08 — Seven copies of `events.db` across repo, `dist/`, and `packaging/`.
**Severity: MEDIUM** · **Points: 2**

**Evidence:** `events.db` found at `runs/live`, `runs/site`, `runs/site_vlm`, `runs/test6`, `packaging/demo_data`, `dist/CVTI Console/_internal/demo_data`, `dist/CVTI Console.app/Contents/Resources/demo_data`.

**Why it matters.** Two risks. First, **ambiguity** — when debugging, it is not obvious which database is live, and the `dist/` copies are stale build artefacts that can mislead. Second, **accidental data disclosure**: demo databases are bundled into shipped installers. Today they contain demo events. If a demo database is ever regenerated from a real site during a pilot, real footage metadata ships to every installer recipient.

There is also a naming inconsistency: `dist/` still produces "CVTI Console" though the product renamed to Argus.

**Definition of Done:**
- `dist/` and build artefacts fully git-ignored and removed from tracking
- Exactly one canonical demo database, generated by a script from synthetic data, never copied from a live run
- A packaging assertion that fails the build if the bundled demo database contains events referencing paths outside `demo_data/`
- Product naming aligned to Argus across packaging outputs

---

### SWE-09 — No type checking and no linting in CI.
**Severity: MEDIUM** · **Points: 3**

**Evidence:** 63 `# noqa` suppressions imply a linter is used locally, but no workflow enforces it. No `mypy`/`pyright` configuration.

**Why it matters.** The `# noqa` count tells me you lint locally and care about it. Nothing enforces it for a second contributor. Type checking matters more than usual here because the system passes loosely-typed dictionaries across process boundaries (`payload`, `scene_context`, `event.extra`) — exactly the places where a typo becomes a silent runtime `None` rather than an error. `context_filter` being `eval`'d against `event.extra` makes this sharper still.

**Definition of Done:**
- `ruff` (or existing linter) runs in CI and fails the build
- `mypy` in non-strict mode over `cvti/contracts.py`, `cvti/serving/`, `cvti/verification/` — the interface-carrying modules first
- The dictionaries crossing process boundaries become `TypedDict` or dataclasses
- A documented policy for `# noqa`: each requires a trailing reason comment

---

### SWE-10 — No dependency vulnerability scanning.
**Severity: MEDIUM** · **Points: 1**

**Evidence:** 56 pinned dependencies (pinning is correct and good), no `pip-audit` or Dependabot configuration.

**Why it matters.** Pinning gives reproducibility but freezes vulnerabilities in place — the flip side of a good decision. `torch`, `transformers`, and `opencv` are large native-code surfaces with regular CVEs. Any enterprise security questionnaire asks how you track this, and "we don't" is a failed question.

**Definition of Done:**
- `pip-audit` in CI, failing on HIGH/CRITICAL
- Dependabot or Renovate opening weekly PRs
- A documented monthly review cadence in `README.md`

---

### SWE-11 — No release integrity: installers are unsigned and unhashed.
**Severity: MEDIUM** · **Points: 3**

**Evidence:** `build-app.yml` produces `.dmg` / `.zip` on tag. No code signing, no notarisation, no published checksums.

**Why it matters.** macOS Gatekeeper will block an unsigned `.dmg` with a warning that reads, to a non-technical user, as *"this software is malware."* Windows SmartScreen does the same. For a **security product**, asking a customer to click past an OS security warning during installation is a bad first impression and undermines the trust the product is selling. Published SHA-256 sums are the cheap interim measure.

**Definition of Done:**
- SHA-256 checksums published with every release asset (cheap, do first)
- Apple Developer ID signing + notarisation for macOS
- Authenticode signing for Windows
- `README.md` documents verification steps

---

## 4. Perspective 2 — Senior Product Designer

> *I am asking who uses this at 2am, what job they are trying to finish, and whether the interface helps them finish it. Not whether it looks good.*

**Overall verdict: the demo is strong; the *daily* product has not been designed yet.**

The distinction matters. A demo has one user (you), a known script, and a five-minute lifespan. A product has three different users, no script, and a 200-day lifespan. Argus is currently excellent at the first and unproven at the second. That is the correct order to build in — but it means the product design work is genuinely still ahead of you, not behind.

---

### PD-01 — Ten navigation surfaces for what are really three jobs.
**Severity: HIGH** · **Points: 8**

**Evidence:** Cameras, Alerts, Live, Map, Ask, Learning, Rules, System, plus feed/detector configuration. Ten detector toggles in `RULE_FLAGS`.

**Why it matters.** The information architecture reflects **how the system was built**, not **what people do with it**. Each panel arrived as a feature landed. Nobody has stepped back and asked what the user is actually trying to accomplish.

There are three jobs:

| Job | Frequency | Who | Currently |
|---|---|---|---|
| **Watch** — is everything OK right now? | Continuous | Guard | Split across Live, Cameras, Map |
| **Triage** — what needs me, and what did I do about it? | Per incident | Guard / manager | Alerts (a flat list) |
| **Configure** — set up cameras, rules, zones, detectors | Rare, at install | Installer | Split across Rules, System, Cameras, detector toggles |

The third job happens *once*, at installation, and occupies roughly half the navigation permanently. The second job happens constantly and gets one panel.

This is the clearest signal in the audit that feature delivery has outpaced product definition. It is also very fixable, and it is a *reorganisation*, not a rewrite — the underlying panels mostly work.

**Definition of Done:**
- Navigation restructured to three primary surfaces: **Watch**, **Triage**, **Configure**
- Configuration moved behind a settings surface, out of the primary navigation
- Map and Live merged — they answer the same question at different zoom levels
- A first-run setup flow that walks an installer through cameras → zones → rules → detectors once, then gets out of the way
- Usability-tested with someone who has never seen the system: they reach "I can see my cameras and I understand what an alert means" with no verbal guidance

---

### PD-02 — There is no triage workflow. Alerts are a flat list.
**Severity: CRITICAL** · **Points: 13**

**Evidence:** Alerts are sorted and colour-coded by threat class. Acknowledge / True / False labelling exists (feeding `cvti/feedback/`). Dedup cooldown is 60s so one incident produces one alert.

**Why it matters.** Sorting is not triage. Colour is not triage. The improvements already made are real and they addressed the *noise* problem — but they did not address the *workflow* problem, and those are different.

A guard at 2am does not need a list. They need to know:
- **What needs me right now?** — a single, unambiguous "next action" item
- **What is already being handled, and by whom?** — there is no ownership concept at all, so two guards will both respond to the same alert, or neither will
- **What happened on the previous shift?** — no handover surface exists, so context resets every shift change
- **What did I decide, and can I show it later?** — labels exist for model training, but there is no incident record a manager can review

The consequence is specific and serious: **the product's core claim is that it reduces alert fatigue, and the alert-fatigue reduction is currently delivered entirely by TrueSight (the model), not at all by the interface.** You cut alerts by 86% with a model. The remaining 14% still land in an undifferentiated list. For a site with 20 cameras that is still dozens of alerts a shift with no workflow to process them.

This is the single largest *product* gap in the system, as distinct from the largest engineering gap (`SWE-01`).

**Definition of Done:**
- Alert states modelled explicitly: `NEW → ACKNOWLEDGED (by user X at time T) → RESOLVED (outcome, note)`
- Ownership: acknowledging claims an alert and shows the claimant to everyone else
- A "needs attention now" view showing only unacknowledged alerts above a priority threshold, defaulting to one at a time
- Shift handover: a summary of the last N hours — what fired, what was resolved, what is still open
- An incident record per alert: frames, VLM reasoning, who responded, what they concluded, exportable as PDF
- Resolution outcomes feed `cvti/feedback/` (ties to `ML-06`)

---

### PD-03 — The product does not know who its user is.
**Severity: HIGH** · **Points: 5**

**Evidence:** No roles, no permissions, no per-user views. `grep` for `site_id`/`tenant`/`organization` → 0.

**Why it matters.** The interface assumes the installer, the operator, and the owner are one person. In a real deployment they are three people with almost non-overlapping needs:

- The **owner** wants a weekly email: *"3 incidents, 2 confirmed theft, estimated ₦X prevented."* They will open the app roughly never. Today there is no artefact for them at all, which means the person who signs the cheque has no recurring reason to perceive value.
- The **guard** wants one screen and a phone. They should never see a detector toggle.
- The **installer** wants configuration depth, once.

Exposing all three surfaces to all three people means each finds the product harder than it needs to be, and the owner — the buyer — gets nothing designed for them.

**Definition of Done:**
- Three roles: Owner, Operator, Installer, each with a default landing surface
- Role-appropriate navigation; operators do not see configuration
- A weekly summary artefact for owners (email or PDF), generated automatically
- Depends on `ARCH-02` (authentication) for enforcement, but the *view* separation can ship first

---

### PD-04 — No mobile response path.
**Severity: HIGH** · **Points: 13**

**Evidence:** Telegram delivers alerts to a phone. Every response action — reviewing frames, acknowledging, labelling — requires the desktop app.

**Why it matters.** The notification is mobile; the response is not. That is backwards for the actual job. A guard is by definition *moving* — that is what patrolling is. Alert arrives on the phone; to do anything with it they must return to a desk. In the time that takes, the incident is over.

This inverts the product's core promise. Argus is sold on speed of response, and the response loop has a walk-to-the-office step in it.

The cheap first version is not an app. It is a mobile-responsive web view served by the existing frame publisher, reachable on the local network, showing the alert with its frames and an Acknowledge button. That captures most of the value.

**Definition of Done:**
- Mobile-responsive alert view served by the engine over the local network
- Telegram alerts deep-link into it
- Acknowledge, label true/false, and add a note — all from a phone
- Authenticated (depends on `ARCH-02`); must not be an unauthenticated open endpoint on the site network
- Works offline-first for the local network case; no cloud dependency

---

### PD-05 — "Ask your cameras" is a demo feature, not a product feature.
**Severity: MEDIUM** · **Points: 5**

**Evidence:** The Ask panel supports natural-language queries. No query history, no saved queries, no scheduled queries.

**Why it matters.** It demos beautifully and it is one of the most differentiated things in the product — natural-language interrogation of camera history is exactly what competitors with fixed taxonomies cannot do. But without persistence it is a party trick. A user asks a good question, gets a good answer, and has no way to keep it, repeat it, or be told when the answer changes.

The gap between "impressive in a demo" and "used on day 30" is entirely persistence and scheduling. This is a case where a small amount of work converts a demo asset into a retention asset.

**Definition of Done:**
- Query history, persisted and searchable
- Saved queries, re-runnable in one click
- Scheduled queries that run automatically and alert on a match — this is the feature that turns Ask into a *monitoring* capability rather than a lookup tool
- Results link to the underlying evidence clips

---

### PD-06 — No ROI or value surface for the buyer.
**Severity: HIGH** · **Points: 5**

**Evidence:** The System panel shows noise-suppression statistics. There is no business-value framing anywhere.

**Why it matters.** The noise-suppression counter is, in my assessment, **the strongest single screen in the product** — it is the one place the interface tells the product's whole story in one glance: *"raw detectors would have shown you 201 alerts; TrueSight showed you 28."*

It is currently framed as an engineering metric. The buyer does not care about suppression percentages. They care about: incidents caught, hours of guard attention saved, false alarms avoided, value of goods protected.

The same underlying data, reframed, becomes the renewal conversation. This is a small amount of work with disproportionate commercial leverage — and it doubles as investor demo material on Saturday.

**Definition of Done:**
- A Value surface translating system metrics into business terms: incidents detected, false alarms prevented, estimated attention-hours saved
- Configurable during setup with site-specific values (average shrinkage per incident, guard hourly cost) so the figures are the customer's, not generic
- Included in the weekly owner summary (`PD-03`)
- Every figure traceable to the underlying events — never an unfalsifiable number, since the product's credibility rests on honest measurement

---

### PD-07 — Errors and degradation are not designed.
**Severity: MEDIUM** · **Points: 3**

**Evidence:** `MemoryGuard` degrades deliberately (drops fps, shrinks image size, disables video-action, sheds a camera). The UI has no corresponding vocabulary for communicating this.

**Why it matters.** The engineering here is genuinely thoughtful — the system gives up quality in a considered order rather than swapping or crashing, and `test_never_sheds_the_last_camera` shows the policy was reasoned about. But from the user's side, a camera silently stopping is indistinguishable from a camera *failing*. For a security product, "I thought that camera was being watched and it wasn't" is the worst possible failure, and right now the system has no way to say so.

Degradation that is invisible is equivalent to a lie.

**Definition of Done:**
- A persistent system-health banner with three states: Healthy / Degraded / Critical
- Degraded state names specifically what was given up and why: *"Camera 4 paused — low memory. Detection continues on cameras 1–3."*
- Every `MemoryGuard` mitigation maps to a user-facing sentence
- Camera-offline and gate-unavailable states surfaced the same way (ties to `SWE-06`)
- Recovery is announced, not silent

---

### PD-08 — No onboarding for a non-technical user.
**Severity: HIGH** · **Points: 8**

**Evidence:** Setup requires installing Python and Ollama, pulling a ~3 GB model, and editing JSON for RTSP URLs.

**Why it matters.** Covered from the installation angle in `USR-01`, but the *design* dimension is distinct: even after everything is installed, there is no guided path from "app opens" to "my cameras are being monitored." The user faces ten navigation items and no indicated starting point.

**Definition of Done:**
- A first-run wizard: add camera → verify feed → draw zones → pick a use case template → confirm detectors → test alert
- Use-case templates (Retail, Warehouse/HSE, Office) that preselect sensible detectors and rules — this is where the existing `CustomizationEngine` and baseline-rules work pays off in the interface
- A "send me a test alert" step so the user confirms the notification path before trusting it
- Setup completable by someone non-technical in under 15 minutes, verified by observation

---

## 5. Perspective 3 — Senior Systems Architect

> *I am asking what happens on day 200 at a customer site I cannot physically reach, with a disk that is full and a camera that has been offline for a week.*

**Overall verdict: the core structural decisions are right; nothing currently in place survives contact with an unattended production site.**

The decisions that matter most architecturally have been made well. The two-process split (engine ↔ app over `events.db` + a localhost frame-publisher port) correctly isolates detection from UI — a hung Qt event loop cannot stall the pipeline. The engine-publishes-frames design means the app never double-decodes, which is the right call for both CPU and correctness. Local-first is a genuine strategic and privacy asset. SQLite is the correct choice for a single-box deployment; reaching for Postgres here would have been over-engineering.

What is missing is everything about *operating* the system when you are not in the room.

---

### ARCH-01 — Zero data retention or purge logic. Evidence accumulates forever.
**Severity: CRITICAL** · **Points: 5**

**Evidence:** `grep -rn 'retention\|def purge\|def prune\|max_age_days' cvti` → **0 occurrences**. Current evidence footprint: `runs/serving` 20 MB + `runs/live` 13 MB. 254 `.mp4` and 1,400 `.jpg` files on disk.

*Accuracy note:* the 1.6 GB total in `runs/` is dominated by training checkpoints (658 MB + 329 MB + 243 MB + 175 MB), **not** evidence. Do not cite 1.6 GB as an evidence-growth figure — it would not survive scrutiny. The concern here rests on the confirmed absence of purge logic, which is sufficient on its own.

**Why it matters.** This is simultaneously an operational bug and a legal exposure, and the legal half is the serious one.

*Operationally:* an unattended edge PC with no purge policy fills its disk. When it does, the failure mode is bad — writes fail, SQLite may be unable to commit, and the system stops recording evidence at exactly the moment it is most likely to be needed. Nobody is watching the disk, because there is no monitoring (`ARCH-05`).

*Legally:* this system stores **identifiable images of people** captured by CCTV. Under GDPR (and Nigeria's NDPR, which matters for your likely first market) that is personal data, and storage limitation is not optional — data must be kept no longer than necessary for the stated purpose. A system with no deletion path cannot demonstrate compliance, cannot honour an erasure request, and cannot answer the retention question on any enterprise procurement form. The typical UK/EU CCTV retention norm is 30 days.

There is a real tension to design around: purge must never delete evidence attached to an open incident or an ongoing investigation. Blind time-based deletion would destroy the very records a customer needs.

**Definition of Done:**
- Configurable retention period per site, defaulting to 30 days
- A purge job running on a schedule, deleting evidence past retention — frames, clips, and the corresponding database rows together, so no orphans remain
- **Legal hold**: evidence attached to an unresolved incident, or explicitly flagged, is exempt from purge and visibly marked as retained
- Disk-usage awareness with a warning threshold surfaced in the UI, and an emergency purge-oldest-first path before the disk fills
- A documented retention policy in `README.md` and a customer-facing privacy note
- An export path so a customer can extract evidence before it expires
- Tests: purge deletes what it should, retains legal-hold items, and leaves no orphaned rows or files

---

### ARCH-02 — No authentication. No authorisation. No audit log. No encryption at rest.
**Severity: CRITICAL** · **Points: 13**

**Evidence:** `grep` for `authenticate|jwt|bcrypt|session_token|check_password` → **0 files**. `audit` → **0 files**. `encrypt|Fernet|AES` → **0 files**.

**Why it matters.** These are four distinct gaps, and I am grouping them because they share one root cause — the system was built for a single trusted operator on a machine they own — and because they are almost always evaluated together in procurement.

- **No authentication.** Anyone with physical or network access to the box has complete control: view all cameras, view all recorded evidence, change every rule, disable every detector. For a security product this is a contradiction in terms. The most sensitive failure is not misuse of the cameras — it is that **anyone can silently disable detection**, and there is no record that they did.
- **No authorisation.** Even once identity exists, everyone will be able to do everything, including operators who should never touch detector configuration.
- **No audit log.** There is no record of who viewed which footage, who changed which rule, who disabled which detector, or who resolved which alert. This is doubly damaging: it is a compliance failure, **and** it destroys the evidentiary value of the footage. Video with no chain of custody and no tamper-evident access record is materially weaker if a customer ever needs it in a dispute or a prosecution. Since the product's purpose is producing evidence, this undermines the deliverable itself.
- **No encryption at rest.** A stolen or decommissioned edge PC yields every recorded frame in plaintext, plus the `events.db` metadata.

Any one of these ends an enterprise procurement conversation. Together they mean Argus is currently sellable only to customers who ask no security questions — which is not the customer you want as a reference.

I want to be fair about sequencing: this is correctly *not* pre-seed-demo work. It is unambiguously **pre-pilot** work. The moment footage of real people at a real site lands on that disk, these become live liabilities rather than future ones.

**Definition of Done:**
- Local user accounts with securely hashed passwords (`bcrypt`/`argon2`), enforced at both the app and the frame-publisher HTTP endpoint — the latter must not remain an unauthenticated localhost-open port once `PD-04` exposes it to the network
- Three roles wired to `PD-03`: Owner, Operator, Installer
- Append-only audit log capturing: login attempts, footage access, rule and detector changes, alert resolutions, evidence export, and purge events. Stored separately from `events.db` and never modifiable through the application
- Encryption at rest for evidence and database — OS-level full-disk encryption is an acceptable documented v1, with application-level encryption as the follow-up
- Session timeout, and a forced credential change on first run so no deployment ships with a default password
- A `SECURITY.md` documenting the model, and an answer sheet for standard procurement questionnaires

---

### ARCH-03 — No remote health, monitoring, or fleet management.
**Severity: CRITICAL** · **Points: 13**

**Evidence:** No health or metrics endpoint (`grep '/health\|/metrics\|def health'` → nothing in serving code). No telemetry, no update mechanism.

**Why it matters.** Ask the question directly: **a customer's box dies at 3am on a Saturday. How do you find out?** Today, the answer is that the customer tells you — probably on Monday, probably after an incident was missed, and the conversation starts with them being angry.

This is the concern that most directly limits how many customers one person can support. Without it:
- You cannot detect a dead engine, a dead camera, a full disk, or a stalled VLM
- You cannot ship a fix without walking someone through a manual reinstall
- You cannot tell the difference between "quiet site" and "broken system" — and for a monitoring product those look **identical** from the outside, which is the most dangerous ambiguity in the entire system
- Every support conversation starts from zero, because there are no logs to retrieve (`SWE-01`)

The privacy-preserving design constraint is real and worth stating: telemetry must carry health signals only — uptime, camera status, error rates, disk headroom, model latency — and **never** frames, event content, or anything identifying people. That constraint is compatible with the local-first promise, and being explicit about it is itself a selling point.

**Definition of Done:**
- A local `/health` endpoint reporting per-camera status, gate reachability and latency, disk headroom, memory level, error counters, and uptime
- A heartbeat to a central dashboard, **opt-in and off by default**, carrying health metrics only, with the payload schema documented publicly
- Alerting to you when a site stops heartbeating or reports degradation
- A signed update mechanism so fixes can be delivered without a site visit
- A local "download diagnostics" bundle (logs + health snapshot, evidence excluded) the customer can email during support
- Documented explicitly in the privacy note: what leaves the box, what never does

---

### ARCH-04 — Single-site by construction. No multi-site or multi-tenancy.
**Severity: HIGH** · **Points: 21**

**Evidence:** `grep -rn 'site_id\|tenant\|organization' cvti` → **0 occurrences**. Every path assumes one box, one site, one `events.db`.

**Why it matters.** This is a deliberate and correct scoping decision for the current stage — I am not criticising it. I am flagging that it is load-bearing in a way that gets more expensive to change with every module added.

The commercial consequence is specific: **your most attractive early customers are chains** — a retailer with 12 shops, a facilities manager with 8 sites. Those buyers want one view across all locations. Today you would sell them 12 independent systems with 12 separate interfaces and no consolidated reporting. That is a materially worse product for exactly the highest-value segment, and it caps deal size at single-site pricing.

There is also an architectural trap to avoid: retrofitting multi-tenancy after data exists is significantly harder than designing for it. The mitigation is cheap and should happen early even if the feature does not — introduce a `site_id` concept in the data model *now*, defaulting to a single site, so a future aggregation layer has something to key on.

**Definition of Done:**
- `site_id` present throughout the data model and event schema, defaulting to `"default"` — this alone is 3 points and should be pulled forward
- A site-registry concept, even with one entry
- An aggregation layer that can consolidate multiple sites into one view (this is the 21-point part)
- Cross-site alert routing and reporting
- Explicit tenant isolation guarantees, documented and tested

---

### ARCH-05 — No formal capacity model. Validated to five cameras.
**Severity: HIGH** · **Points: 5**

**Evidence:** Measured on an 18 GB MacBook Pro: 5 cameras at 6.2 fps each, 163 ms per detection batch, gate latency 18–20 s median after worker auto-scaling. `_gate_workers_for()` scales workers as `max(1, min(3, n_cameras // 2))`.

**Why it matters.** Five cameras is a small shop. A warehouse has 20; a mid-size retailer has 40. You currently cannot tell a prospect what hardware they need for their camera count, which means every deal requires a bespoke technical conversation, and you carry the risk of an undersized deployment failing in production.

The gate worker cap of 3 is a hard ceiling on verification throughput that was chosen for a 5-camera machine and has not been revisited for larger deployments. `MemoryGuard` will keep such a system *alive* by shedding load — which is good engineering — but shedding cameras at a 40-camera site means the customer silently loses coverage they paid for. Degrading gracefully is only acceptable when the user is told (`PD-07`).

**Definition of Done:**
- A published capacity table: cameras × hardware tier → sustained fps, expected gate latency, RAM required
- Load-tested at 10, 20, and 40 cameras, with the actual breaking point identified and documented
- A pre-install sizing tool: enter camera count and hardware, get a supported/unsupported verdict
- Gate worker scaling revisited for high camera counts, and the cap of 3 justified or raised with measurements
- Documented minimum and recommended hardware specifications

---

### ARCH-06 — SQLite will become a write bottleneck, and there is no schema migration path.
**Severity: MEDIUM** · **Points: 5**

**Evidence:** `events.db` is the inter-process channel between engine and app. No migration framework present.

**Why it matters.** Two separate issues sharing a file.

*Concurrency:* SQLite permits one writer at a time. At five cameras with 60-second dedup this is comfortably fine, and choosing SQLite was right. At 40 cameras, with the app polling for reads while the engine writes events and the escalation ticker updates state, write contention becomes plausible. This needs measuring under `ARCH-05`'s load tests rather than assuming — but it should be measured before a large deployment, not discovered during one.

*Migrations:* this is the more immediate risk. Customers will soon have `events.db` files containing real evidence. The first schema change — and `site_id` from `ARCH-04`, alert states from `PD-02`, and audit fields from `ARCH-02` all require one — must upgrade those files without data loss. With no migration framework, the options are manual SQL or telling a customer to start fresh. Neither is acceptable once real evidence exists.

**Definition of Done:**
- A schema version table and a migration runner applying migrations in order at startup
- Migrations tested forward from every previously released schema version, including `v0.9.0`
- Automatic timestamped database backup before any migration
- WAL mode enabled and write contention measured under load
- A documented path to Postgres if load testing shows SQLite is the limit — decided by measurement, not preference

---

### ARCH-07 — Inter-process handshake via a `frames.json` port file is fragile.
**Severity: MEDIUM** · **Points: 3**

**Evidence:** The engine writes its frame-publisher port to `<output_dir>/frames.json`; the app reads it to discover the endpoint. Stale-file handling exists.

**Why it matters.** File-based service discovery has well-known failure modes: a stale file after an unclean shutdown, a port reused by another process, a race where the app reads before the engine writes, and permission problems on Windows. Stale handling exists, which shows the failure mode was anticipated — but the mechanism remains more fragile than it needs to be, and every one of these failures presents to the user as "the live view is blank" with no explanation (`SWE-01`, `PD-07`).

**Definition of Done:**
- The port file includes the engine PID and a start timestamp; the app validates the process is alive before trusting it
- A fixed default port with a documented override, so discovery is the exception rather than the norm
- The app retries with backoff and reports a clear, specific state when the engine is unreachable
- Clean shutdown removes the file; startup reclaims a stale one safely
- Tested on Windows, where file locking semantics differ

---

### ARCH-08 — Single point of failure with no backup or recovery path.
**Severity: HIGH** · **Points: 8**

**Evidence:** One box, one disk, one `events.db`. No backup, no redundancy, no documented recovery.

**Why it matters.** If the edge PC's disk fails, the customer loses **all** recorded evidence, all configuration, all zones, all rules, and all operator labels. For a security product, that is a catastrophic and — importantly — *irreversible* customer outcome. Reconfiguring zones and rules for 20 cameras is hours of skilled work; the evidence is simply gone.

Hardware redundancy is genuinely out of scope for an edge product at this stage, and I would not recommend it. But **configuration backup is cheap and its absence is not defensible.** Configuration is small, non-sensitive, and losing it is the most recoverable part of the disaster — provided a backup exists.

**Definition of Done:**
- Automatic configuration backup (cameras, zones, rules, detector settings, routing policy) to a user-chosen location, versioned
- One-click restore onto a fresh install
- Optional evidence backup to an external drive or customer-controlled NAS, respecting the local-first promise — never to your cloud by default
- A documented, tested disaster-recovery procedure with a target recovery time
- `events.db` integrity check at startup with automatic recovery from the last good backup on corruption

---

### ARCH-09 — No graceful handling of long-term camera absence.
**Severity: MEDIUM** · **Points: 3**

**Evidence:** The tamper detector catches sudden brightness/sharpness collapse. There is no distinct concept of a camera being *offline* for an extended period.

**Why it matters.** Tamper detection answers "did someone cover this camera?" It does not answer "has this camera been unreachable for six days?" Those are different events requiring different responses, and the second is more common — RTSP streams drop, cameras lose power, network switches fail.

The dangerous property is the same ambiguity as `ARCH-03`: a camera that is offline produces no alerts, which is indistinguishable from a camera watching a quiet area. The customer believes they have coverage they do not have. In a security product, false confidence is worse than a visible failure.

**Definition of Done:**
- Explicit per-camera connection state: Connected / Reconnecting / Offline, with time-in-state
- Automatic reconnection with exponential backoff, and the attempt history visible
- An offline camera raises its own alert through the normal routing path after a configurable grace period
- Camera status is prominent in the Watch surface — never inferred from an absence of alerts
- Uptime per camera reported in the weekly owner summary (`PD-03`)

---

## 6. Perspective 4 — Senior AI/ML Engineer

> *I am asking what you actually measured, what you are claiming, and whether the gap between those two things will hold up when someone competent pushes on it.*

**Overall verdict: the evaluation discipline is genuinely above average for the stage; the models underneath are thin, and the system's accuracy rests on prompt text that is not under regression control.**

I want to open with real credit, because it is deserved and it is the rarest thing here. Most CV startups at this stage cannot answer "how do you know?" This one can. The two-stage evaluation (raw candidates vs. TrueSight-confirmed) is exactly the right framing because it isolates the product's actual contribution. `GateUnavailable` and `preflight()` exist because an earlier run reported "100% suppressed" when Ollama was down — and rather than shipping that number, the failure was diagnosed and the harness was hardened to make it impossible. Checkpoint files keyed on gate+sensitivity+kind+detectors exist because mock and real results contaminated each other once. Rejected clips are parked in `data/test_clips/_rejected/` with a README explaining why.

That is a person building an honest instrument. It is the most defensible asset in the company. Now the problems.

---

### ML-01 — Only one trained model exists, on 220 clips. Everything else is rules plus prompts.
**Severity: HIGH** · **Points: 21**

**Evidence:** One fine-tuned model: VideoMAE for theft, trained on ~220 clips. The HSE detectors (fire/smoke via HSV flicker, panic-running, crowd-formation, fall via bbox aspect ratio, tamper via brightness/sharpness collapse) are hand-tuned heuristics. Detection is YOLOv8n off-the-shelf.

**Why it matters.** 220 clips is very small for video action recognition — small enough that the model is best understood as a weak prior rather than a reliable classifier, and small enough that its measured performance carries wide uncertainty.

The rule-based detectors are the deeper issue. HSV flicker for fire fires on sunsets, brake lights, reflective surfaces, and television screens — which is precisely why the raw fire false-positive rate is **90%**. Bbox aspect ratio for falls fires on anyone crouching, bending, or sitting. These heuristics work as *candidate generators* feeding a smart gate, and the architecture is honestly well-suited to that. But it means the intelligence in the product is concentrated almost entirely in one 4B-parameter general-purpose VLM and the text of the prompt sent to it.

That is a legitimate and even clever architecture for reaching a working product quickly. It is also a thin technical moat, and it makes `ML-02` the most important item in this section.

**Definition of Done:**
- A data acquisition strategy beyond public clips: proper datasets (RWF-2000 for violence, UR Fall / Le2i for falls, FireNet or D-Fire for fire), plus pilot-site data under explicit customer agreement
- At least three detectors backed by trained models rather than heuristics, prioritised by commercial value
- A documented model card per detector: training data, held-out performance, known failure modes, appropriate use
- Heuristic detectors explicitly labelled as candidate generators in code and documentation, so nobody mistakes them for classifiers

---

### ML-02 — Accuracy depends on prompt text that is not under regression control.
**Severity: CRITICAL** · **Points: 8**

**Evidence:** Three prompt revisions produced these gate results on the identical 36-clip set with identical raw candidates (201 alerts, P=37.5%, R=100%):

| Prompt version | Precision | Recall | FPR | Alerts passed |
|---|---|---|---|---|
| v1 | **37.5%** | 100% | 55.6% | 191 (only 5% suppressed) |
| v2 "balanced" *(current default)* | 53.3% | 88.9% | 25.9% | 61 |
| v3 "strict" | **63.6%** | 77.8% | 14.8% | 28 |

**Why it matters.** Precision moved **26 percentage points** on wording changes alone. The v1 result is the sharpest lesson in the whole project: a prompt instructing the model to treat the detector's flag as a strong prior caused it to confirm 95% of everything — the gate was present, running, and contributing nothing, while appearing to work. A later revision confirmed on *"taking an item from the shelf,"* which is what every shopper does.

The implication is uncomfortable and needs stating plainly: **the product's headline metric is a function of a string literal in `gate.py`, and there is no test that fails when that string changes.** Anyone can edit `_QUESTIONS` or `SENSITIVITY_QUESTIONS`, run the app, see it work, and ship a 26-point precision regression invisibly.

The same exposure exists externally. `gemma3:4b` is not a frozen artefact — a model update, a quantisation difference, or a different Ollama version can shift behaviour with no code change on your side. `SENSITIVITY_MEASURED` hard-codes measured numbers as constants, meaning code can drift out of sync with the reality it claims to describe.

Note also that `min_confidence=0.35` acts as a second, independent accuracy lever with no documented measurement behind that specific value.

This is the highest-leverage ML item in the audit. The eval harness already exists — this is about wiring it to the release process.

**Definition of Done:**
- A **prompt regression suite**: a fast, cached subset of clips run automatically whenever `gate.py` prompt text changes, failing CI if precision or recall moves beyond a defined tolerance
- Prompts versioned with an identifier stamped into every `VerificationResult` and stored with the event, so any archived result can be traced to the exact prompt that produced it
- The VLM model version and quantisation pinned and recorded in eval metadata
- `SENSITIVITY_MEASURED` generated from archived metrics rather than hand-maintained — the pattern already used by `tools/make_numbers_sheet.py`, applied here
- `min_confidence` swept and its default justified by measurement
- A documented procedure for re-validating after any VLM upgrade

---

### ML-03 — Seven of ten detectors are unmeasured. The product markets ten capabilities and can evidence three.
**Severity: CRITICAL** · **Points: 21**

**Evidence:** `RULE_FLAGS` exposes ten detectors: `concealment`, `video_action`, `violence`, `weapons`, `theft`, `tamper`, `fire_smoke`, `running`, `crowd_formation`, `fall`.

| Detector | Status |
|---|---|
| `concealment` + `video_action` (theft) | ✅ Measured — 36 clips |
| `fire_smoke` | ✅ Measured — 39 clips |
| `crowd_formation` | ✅ Measured — 38 clips |
| `violence` | ❌ Unmeasured |
| `weapons` | ❌ Unmeasured |
| `fall` | ❌ Unmeasured |
| `tamper` | ❌ Unmeasured |
| `running` (panic) | ❌ Unmeasured |
| `theft` (standalone) | ❌ Unmeasured |

**Why it matters.** Every unmeasured detector is a claim you cannot support. Two of them — `violence` and `weapons` — are `critical` priority in `configs/baseline_critical_v1.json`, meaning they are **always on, in every customer configuration, and not disableable in pilots.** The two highest-stakes detectors in the system, the ones that would summon an armed response, have never been evaluated. If `weapons` has poor precision, the product generates armed-response callouts for umbrellas. If it has poor recall, it misses the threat it exists for.

`fall` matters commercially because it is the anchor for the HSE use case, and `tamper` matters because it is the system's own integrity check — a tamper detector with unknown recall means the system cannot verify it is not being defeated.

The reason for the gap is documented and legitimate: YouTube search returns *news coverage of* incidents rather than usable CCTV footage, at roughly a 1-in-5 hit rate, and this was correctly written up rather than papered over. The honest conclusion — that proper datasets are required — was reached. It now needs acting on.

**Definition of Done:**
- `weapons` and `violence` measured first — they are always-on and critical priority
- Proper datasets acquired: RWF-2000 (violence), UR Fall / Le2i (falls), a weapons benchmark, synthetic and staged footage for tamper
- `fall`, `tamper`, and `running` measured
- Every detector's measured numbers surfaced **in the UI at the point of configuration**, so a user enabling an unvalidated detector sees that it is unvalidated
- Detectors without measurement marked `EXPERIMENTAL` in the interface and excluded from marketing claims
- `docs/NUMBERS.md` regenerated to cover all ten

---

### ML-04 — Eval sets are 36–39 clips. One clip is worth ~2.6 percentage points.
**Severity: HIGH** · **Points: 8**

**Evidence:** Theft n=36 (9 threat / 27 normal). Fire n=39 (9/30). Crowd n=38 (8/30).

**Why it matters.** These sets are honest and genuinely useful — I would rather have 36 well-labelled clips than 3,000 badly-labelled ones, and the effort spent eyeballing every clip via contact sheets was the right investment. But they are small enough that the confidence intervals are wide, and the reported point estimates carry more apparent precision than they warrant.

With 9 positive clips, recall moves in increments of 11 percentage points — a single clip. Fire's headline **100% recall on 9 positives** has a 95% confidence lower bound near 66%. That is still a good result. It is not "we never miss a fire," and it must not be presented that way, because the first person to check will find the sample size.

The 3:1 negative-to-positive ratio is also unrepresentative. Real deployments run vastly more normal footage than threat footage, so the true false-positive burden per operational day is not directly captured by these sets.

Related: the fire and crowd runs used `max_seconds_per_clip=15.0` while theft used `30.0`. That inconsistency is defensible per-detector but should be recorded explicitly so results are not compared across incompatible settings.

**Definition of Done:**
- Confidence intervals reported alongside every point estimate, in `docs/NUMBERS.md` and in any investor material
- Eval sets grown to at least 100 clips per detector, with a target of 200
- Negative-to-positive ratio brought closer to operational reality, or false alarms reported per hour of footage rather than per clip
- Per-clip results published, not just aggregates, so results are independently checkable
- Eval configuration (clip duration, image size, confidence threshold) recorded in metadata and held constant within a comparison

---

### ML-05 — No drift detection. Accuracy can silently decay after deployment.
**Severity: HIGH** · **Points: 8**

**Evidence:** No monitoring of detection-rate distributions, confirmation rates, or per-camera behaviour over time.

**Why it matters.** Every measured number in this system is a *point-in-time* result on *public clips*. In deployment, conditions change continuously: a camera is nudged during cleaning, a shop rearranges shelving, a new light fixture changes the scene, seasons change the daylight, a new uniform changes what people look like.

Any of these can degrade accuracy substantially, and **nothing in the system would notice.** The failure is silent by construction — a detector that has stopped working produces no alerts, which for a monitoring product is indistinguishable from a safe site. This is the same dangerous ambiguity that appears in `ARCH-03` and `ARCH-09`, and its recurrence across three perspectives is not coincidental: it is the system's characteristic failure mode.

For a product whose entire pitch is measured reliability, "we measured it once, before you bought it" is a weak position under scrutiny.

**Definition of Done:**
- Per-camera baselines established over the first two weeks: candidate rate, confirmation rate, alerts per hour, detection-size distributions
- Statistical drift alerts when current behaviour departs from baseline beyond a threshold
- A "camera may have moved" check comparing periodic scene fingerprints against the reference frame
- Drift status visible in the System panel and included in the weekly owner summary
- A documented re-validation procedure when drift is detected

---

### ML-06 — The reinforcement/feedback loop is built but has never been validated end to end.
**Severity: HIGH** · **Points: 8**

**Evidence:** `cvti/feedback/` (552 LOC) implements label capture, calibration, and retrain triggering. It has one test file. It has never run on real operator labels producing a validated improvement.

**Why it matters.** This is untested product surface that is architecturally load-bearing — it is the mechanism by which the product is supposed to get better at each site, which is a central part of the long-term story and a real differentiator if it works.

Right now nobody knows whether it works, and the risks are not hypothetical. A feedback loop trained on operator labels can *degrade* a system: operators mislabel under time pressure; they label what they remember rather than what the frame shows; a loop that over-fits to one camera's quirks can reduce accuracy elsewhere. Without a held-out evaluation gating each retrain, this feature can silently make the product worse — and it would do so invisibly, because there is no drift detection (`ML-05`) to catch it.

There is a second-order concern: operator labels are the input to model improvement, which makes them a data-integrity surface. With no audit log (`ARCH-02`), there is no record of who labelled what.

**Definition of Done:**
- End-to-end validation on real labels: collect operator labels from a running deployment, retrain, and demonstrate a measured improvement on held-out data
- A retrain **never** ships automatically — every candidate model is evaluated against the frozen held-out set and only promoted if it beats the incumbent
- Model registry with versioning and one-click rollback
- Label quality controls: inter-rater agreement where possible, and a review path for labels that contradict a high-confidence gate verdict
- Operators can see the effect of their labels — this is what makes people keep labelling

---

### ML-07 — No per-detector error analysis. Aggregates hide failure modes.
**Severity: MEDIUM** · **Points: 5**

**Evidence:** Metrics report precision/recall/FPR/F1 per run. No systematic analysis of *which* clips fail and *why*.

**Why it matters.** Aggregate metrics tell you the score, not the reason. Fire at 81.8% precision means 2 false positives out of 11 confirmed — and knowing what those two were is worth more than the percentage, because it tells you whether the next fix is a prompt change, a threshold change, or a model change.

The same applies to the misses. Theft at 88.9% recall means one missed threat clip; that single clip likely characterises an entire failure mode. Right now that information exists in the archived per-clip JSONL but is not analysed.

You already have the raw material — `clip_results_*.jsonl` retains per-clip outcomes. This is analysis work, not collection work.

**Definition of Done:**
- A failure-analysis report per detector: every FP and FN with its frames, the gate's stated reasoning, and a categorised root cause
- Failure taxonomy per detector (e.g. fire: sunset / brake lights / screens / reflections)
- The taxonomy drives targeted eval clips, so each known failure mode has explicit coverage
- Included in `docs/NUMBERS.md` so the honest account of failure modes is part of the public record — this is a credibility asset, not a liability

---

### ML-08 — Latency of 18–20 seconds is unaddressed for time-critical threats.
**Severity: HIGH** · **Points: 8**

**Evidence:** Gate latency measured at 46.5 s (1 worker) → 28.0 s (2 workers) → 18–20 s median with auto-scaled workers. Detection itself is fast: 163 ms per 5-camera batch.

**Why it matters.** Note first that the detection stage is not the problem — YOLO returns in milliseconds. The latency is entirely the VLM verification stage, which is the deliberate trade: you accept latency to buy precision.

For theft and most HSE cases that trade is clearly correct — a shoplifting incident reviewed 20 seconds later is entirely actionable, and the precision gain is worth far more than the delay.

For a weapon or active violence, 20 seconds is a long time, and this is a question you will be asked directly on Saturday. It also interacts badly with `ML-03`: `weapons` and `violence` are always-on critical detectors that are both **unmeasured** and **slowest to act on**.

The architecture already contains the answer, and it should be made explicit rather than left implicit: **for critical threats, alert immediately on the raw detection and let verification update the alert.** The operator gets a provisional warning in under a second, then a confirmation or retraction 20 seconds later. This preserves the precision story for the high-volume cases while removing the objection for the life-safety ones.

**Definition of Done:**
- A two-tier alerting policy: `critical` priority detectors emit an immediate provisional alert, then update in place with the verdict
- Provisional alerts are visually distinct and clearly labelled as unverified
- Retraction is explicit and visible when the gate rejects a provisional alert
- Latency measured and published per priority tier
- Verification queue prioritised so `critical` candidates are never queued behind `medium` ones
- Documented in investor and customer material as a deliberate design decision, with the numbers

---

### ML-09 — Model and dataset provenance is not documented.
**Severity: MEDIUM** · **Points: 3**

**Evidence:** CamNuvem is used for theft training and held-out evaluation. Clips were fetched from YouTube via `tools/fetch_eval_clips.py`. No licence or usage documentation.

**Why it matters.** Two exposures. *Legal:* YouTube-sourced clips carry unclear licensing for commercial evaluation use, and CamNuvem has its own terms. An investor doing technical diligence, or an enterprise customer's legal team, will ask. *Scientific:* without documented provenance, results are not reproducible by a third party, which weakens the credibility that the eval work exists to establish.

There is also a privacy dimension worth being deliberate about — evaluation clips contain identifiable people who did not consent to appearing in a commercial dataset.

**Definition of Done:**
- A dataset card per source: origin, licence, size, class balance, collection date, known biases
- Licence review of every source, with non-permissive sources replaced or removed
- CamNuvem terms confirmed to permit commercial evaluation
- The train/test split for VideoMAE documented and verified to have no clip-level leakage
- A stated policy on retention and use of pilot-site data, agreed with customers in writing before collection

---

## 7. Perspective 5 — Investor

> *I am asking whether this becomes a company, what could kill it, and whether the founder tells me the truth when the numbers are unflattering.*

**Overall verdict: genuinely differentiated with a rare form of credibility, and early on every axis that gets measured at diligence.**

I want to be direct about what is working, because it is unusual.

**The measurement discipline is the strongest signal in the business.** Most CV demos I see cannot answer "how do you know?" This one answers with held-out data, publishes numbers that are unflattering (53.3% precision is not a vanity metric), and documents its own failures — including a written record of a case where the system's own gate was rubber-stamping and contributing nothing. Founders who publish their own negative results are rare, and it is one of the better available proxies for whether the rest of the diligence will hold up. It also means the technical claims here can be *checked*, which most cannot.

**The fire result is the demo.** Raw detectors: 90% false-alarm rate. With TrueSight: 6.7%, with zero fires missed. That is one sentence, it is memorable, it is measured on held-out data, and it demonstrates the entire thesis. Lead with it.

**Local-first is a real strategic wedge**, particularly for the Nigerian market and for EU privacy-sensitive segments. Cloud-based competitors have a structural disadvantage in any deployment where footage cannot leave the premises, and that constraint is becoming more common, not less.

**Plain-English rules are genuinely differentiated.** Competitors ship fixed taxonomies. "Tell it what you care about in a sentence" is a different product, not a better version of the same one.

Now the questions you will be asked, and my honest assessment of each answer.

---

### INV-01 — Zero pilots, zero customers, zero real-site hours.
**Severity: CRITICAL** · **Points: 21**

**Evidence:** All measured results derive from public clips (CamNuvem test split, curated YouTube footage). No deployment data exists.

**Why it matters.** This is the question that dominates the meeting, and it is the one that most cheaply resolves the others. Every concern in this section — defensibility, market validation, pricing, latency tolerance, willingness to pay — either dissolves or sharpens with a single real deployment.

The specific risk of having none is that you are **optimising a product nobody has used in anger.** Ten detectors, an escalation engine, a routing policy, a reinforcement loop — all built to a specification derived from reasoning rather than observation. Some meaningful portion of that work will turn out to solve problems real users do not have, while the problems they do have are not yet visible. That is not a criticism of the engineering; it is the normal cost of building without users, and it compounds with every week.

One pilot converts the story from *"we built a system that works on clips"* to *"we run at a real site and here is what happened."* That difference is worth more than any feature currently on the roadmap.

**Definition of Done:**
- One pilot site live, ideally Deluxe Paints given the existing relationship and the fire use case being your strongest measured result
- A written pilot agreement covering data use, retention, and liability — this is also what makes `ARCH-01` and `ARCH-02` non-optional
- 30 days of continuous operation with uptime recorded
- Real-site metrics: alerts per day, confirmation rate, operator-labelled accuracy, false-alarm rate per shift
- A written case study with the customer's own words on what changed
- A reference customer willing to take a call from an investor

---

### INV-02 — Defensibility is thin. The moat is not where it appears to be.
**Severity: HIGH** · **Points: 13**

**Evidence:** The core intelligence is an off-the-shelf VLM (`gemma3:4b`) plus prompt text plus integration work. One fine-tuned model on 220 clips.

**Why it matters.** Asked directly — *"what stops a funded competitor doing this in a quarter?"* — the honest answer today is: not much of the visible surface. YOLOv8n is public. Gemma is public. Prompts are copyable and, as `ML-02` shows, are where most of the accuracy lives.

But I think the real moat is being under-recognised, and it is worth naming precisely because it changes the pitch:

1. **The evaluation harness and the labelled data.** Competitors will not build this. It is unglamorous, it produces unflattering numbers, and it has no demo value. It is also exactly what turns "our AI is smart" into "here is the measured delta" — and it compounds, because every clip added makes the next model decision better-informed.
2. **The accumulating operator-feedback corpus**, once pilots run — real labelled data from real sites is the asset that cannot be copied.
3. **Local-first deployment expertise** — making this run reliably on cheap edge hardware in a Nigerian retail environment with intermittent power and imperfect networks is real, transferable, hard-won knowledge.

The pitch should shift accordingly: the moat is not the model, it is the measurement loop and the data it generates. That is a more honest claim and, to a technical investor, a more convincing one.

**Definition of Done:**
- A defensibility narrative built on data accumulation and the measurement loop, not model novelty
- The eval harness positioned explicitly as proprietary infrastructure
- A data strategy showing how each pilot makes the product measurably better for the next customer
- Provisional patent review on the two-stage candidate-generation-plus-VLM-verification architecture, if counsel advises it is viable
- Competitive analysis against Verkada, Ambient.ai, and Coram that is specific about where you win (retrofit onto existing cameras, local-first, plain-English rules) and honest about where you do not (scale, funding, integrations, support)

---

### INV-03 — Bus factor of one.
**Severity: CRITICAL** · **Points: 8**

**Evidence:** Single developer. 15,470 LOC. `git log` shows a single author. Substantial architectural context — why the crowd detector counts raw detections, why the baseline must not duplicate configured rules, why gate errors must not count as rejections — exists partly in code comments and partly in one person's head.

**Why it matters.** This is the risk most likely to be raised in partner discussion after you leave the room, and it is unglamorous to fix. If you are unavailable for a month, everything stops: development, support, sales engineering, and the pilot.

There are genuine mitigations already in place and they should be credited — the code comments are unusually good at recording *why* rather than *what*, tests encode policy intent, and `docs/` is extensive at 19 files. That is meaningfully better than typical. But `docs/` is also *sprawling* and partly historical (`48HR_PLAN.md`, `SUNDAY_DEMO.md`, `GET_WEAPON_CHECKPOINT_TODAY.md`), which means a new engineer cannot easily tell current architecture from a superseded plan.

**Definition of Done:**
- `docs/` consolidated: current architecture clearly separated from historical planning, with superseded documents moved to `docs/archive/`
- An `ARCHITECTURE.md` that a competent engineer can read in an hour and then make a safe change
- A documented development environment setup, verified by someone else following it from scratch
- A runbook for the pilot: deploy, diagnose, recover
- A concrete hiring plan with the first engineering hire scoped and budgeted
- Credentials, deployment access, and release signing keys documented and recoverable

---

### INV-04 — Unit economics, pricing, and market sizing are undefined.
**Severity: HIGH** · **Points: 8**

**Evidence:** No pricing model in any documentation. No hardware cost analysis. No stated market sizing.

**Why it matters.** You will be asked what a site costs to serve and what it pays, and the local-first architecture makes this a *good* story that is currently untold. Inference runs on the customer's hardware, so your marginal cost per site approaches zero — that is a structurally better gross-margin profile than cloud-based competitors who pay for GPU inference per camera per month. That is a genuine advantage and right now it is not in the pitch.

The unresolved questions are: who buys the edge PC, what is the minimum viable hardware, is this priced per site or per camera, and what does support cost per site given `ARCH-03` does not yet exist.

**Definition of Done:**
- A pricing model with a stated rationale, tested in conversation with at least three prospects
- Hardware BOM at minimum and recommended tiers, with sourcing and cost
- Unit economics per site: hardware, deployment labour, ongoing support, gross margin
- Market sizing for the initial beachhead — Nigerian retail and industrial HSE — built bottom-up from site counts, not top-down from a global CCTV market figure
- The near-zero marginal inference cost made explicit in the pitch

---

### INV-05 — Regulatory and liability exposure is unaddressed.
**Severity: HIGH** · **Points: 5**

**Evidence:** No privacy policy, no data processing documentation, no terms of service, no liability position. Combined with `ARCH-01` (no retention) and `ARCH-02` (no auth, no audit, no encryption).

**Why it matters.** Two distinct exposures.

*Data protection:* the system processes biometric-adjacent personal data (identifiable images of people) under GDPR and NDPR. With no retention policy, no access control, and no audit trail, a deployment today would be difficult to defend if challenged. The moment a pilot begins, this stops being theoretical.

*Liability:* the harder question is what happens when the system misses a real incident. Recall is measured at 88.9% for theft and 75% for crowd — meaning misses are expected, known, and quantified. That quantification is honest and good practice, but it also means a customer can point to a documented miss rate. Your position must be contractually explicit: Argus is an assistive layer that reduces operator load, not a guarantee of detection, and it does not replace human security.

Getting this wrong once could end the company. Getting it right is a few thousand in legal fees.

**Definition of Done:**
- Privacy policy and DPIA covering both GDPR and NDPR
- Terms of service with an explicit limitation of liability and a clear statement that Argus is assistive, not a guarantee
- Data processing agreement template for pilot customers
- Signage and notification guidance for customer sites, since they carry obligations to inform people being recorded
- Professional indemnity insurance quoted, and obtained before the first pilot
- Legal review completed before any real footage is recorded

---

### INV-06 — No go-to-market motion.
**Severity: HIGH** · **Points: 13**

**Evidence:** No sales collateral, no pricing, no defined ICP, no channel strategy, no pipeline.

**Why it matters.** A demo is not a distribution strategy. The relevant question is how customer two through fifty are reached without your personal involvement in each, and there is currently no answer.

The most likely channel is worth naming: **existing CCTV installers.** They already have the customer relationships, they already visit sites, and Argus is an upsell onto cameras they installed. That is a far more scalable motion than direct sales for a single-founder company, and it aligns with the product's retrofit positioning.

**Definition of Done:**
- A defined ICP: site size, camera count, sector, buyer persona, trigger event
- Sales collateral led by the measured numbers — the fire result is the hook
- A channel strategy with CCTV installers, including margin structure
- A repeatable demo that runs from the installer without your involvement
- A pipeline of at least ten qualified prospects
- Documented CAC and sales-cycle assumptions, to be tested

---

### INV-07 — Product scope has outrun validation.
**Severity: MEDIUM** · **Points: 3**

**Evidence:** Ten detectors, three measured. Escalation, routing, NL watches, reinforcement learning, agent mapper, feed switching, memory guard — all built. Zero users.

**Why it matters.** This pattern — building broadly before validating narrowly — is the most common way pre-seed engineering effort is wasted, and it is visible here. The countervailing point is fair: breadth makes for a better demo and demonstrates technical range, which matters when raising on a thesis rather than traction.

But the honest reading is that some of this work is speculative. The reinforcement loop (`ML-06`) has never run on real labels. Routing and escalation were designed without an operator's input. The correct response is not to remove features; it is to be clear-eyed that validation, not construction, is now the bottleneck, and to stop adding scope until `INV-01` is resolved.

**Definition of Done:**
- A feature-status register: Measured / Built-but-unvalidated / Experimental, published internally and reflected in the UI (`ML-03`)
- A feature freeze on new detectors and new capabilities until one pilot is live
- Post-pilot review identifying which built features were actually used, feeding a deprecation decision
- Marketing claims restricted to the Measured tier

---

## 8. Perspective 6 — The Users

> *Four different people have to live with this. I am asking what each of their days actually looks like.*

**Overall verdict: the buyer has no reason to open the app, the operator cannot act from where they stand, and the installer may not get it running at all.**

---

### USR-01 — Installation is the single largest adoption blocker, and it is not the app.
**Severity: CRITICAL** · **Points: 13**

**Evidence:** Setup requires: install Python, install Ollama, pull a ~3 GB model, edit JSON for RTSP URLs. Installers are unsigned (`SWE-11`), so macOS Gatekeeper and Windows SmartScreen will both warn.

**Why it matters.** I want to state this plainly because it is easy to underrate: **you have shipped `.dmg` and `.zip` installers that cannot actually detect anything on their own.** The application installs; the intelligence does not come with it. A user who downloads the release, clicks past an OS security warning, opens the app, and finds it non-functional has had a complete product failure at first contact — and they will conclude the product does not work, not that a dependency is missing.

The individual steps are each defensible engineering decisions. Together they form a wall that a typical CCTV installer — the exact person who should be deploying this — will not get over. Every step is an opportunity to give up, and the drop-off is multiplicative.

This blocks `INV-06` entirely: you cannot build an installer channel around software that installers cannot install.

**Definition of Done:**
- One installer that bundles everything: application, Python runtime, Ollama or an equivalent embedded inference runtime, and the model weights
- No terminal use required at any point
- Signed and notarised (`SWE-11`) so no OS security warning appears
- If model weights make the download too large, a guided in-app first-run download with progress, resume, and a clear explanation of what is happening
- A self-test on first run that verifies every component and reports precisely what is missing
- Validated by observation: a non-technical person completes installation and sees a live camera in under 30 minutes without assistance

---

### USR-02 — The shop owner (the buyer) has no reason to ever open the app.
**Severity: HIGH** · **Points: 5**

**Evidence:** No summary reporting, no ROI surface (`PD-06`), no scheduled communication.

**Why it matters.** The person who signs the cheque and decides on renewal has no recurring contact with the product's value. They bought it, it runs somewhere in the back office, and their only signal is whether their staff complain. That is the profile of a product that gets cancelled at renewal — not because it failed, but because nobody could articulate what it did.

The fix is small: a weekly summary they receive without asking. *"This week: 47 candidate events, 6 confirmed, 2 required response. Estimated ₦X in prevented loss. All 8 cameras online 99.2% of the time."* That artefact alone changes the renewal conversation, and it is largely a presentation layer over data the system already produces.

**Definition of Done:**
- An automatic weekly summary by email or PDF, no action required to receive it
- Written in business terms: incidents, outcomes, estimated value, camera uptime
- Month-over-month trends
- A monthly deeper report suitable for forwarding to a board or an insurer
- Every figure traceable to underlying events (`PD-06`)

---

### USR-03 — The guard cannot act from where they are.
**Severity: CRITICAL** · **Points: 8**

**Evidence:** Telegram delivers alerts to a phone; all response actions require the desktop app. Median alert latency 18–20 s (`ML-08`). No ownership model (`PD-02`).

**Why it matters.** The guard's day is the product's real test, and it currently has three compounding problems:

1. **The 20-second delay** means the alert arrives after the moment. For theft this is acceptable; for violence it is not (`ML-08`).
2. **They must return to a desk** to see frames, acknowledge, or label (`PD-04`).
3. **No ownership** means with two guards on shift, both respond or neither does (`PD-02`).

Each has a corresponding fix elsewhere in this audit; the point of stating it here is that from the guard's seat these are not three issues, they are one experience — *"I get told about things I can't do anything about from where I am standing."* If that is how the shift feels, the guard stops trusting the alerts, and a monitoring product that the monitor ignores has zero value regardless of its precision.

**Definition of Done:**
- Full response capability from a phone (`PD-04`)
- Immediate provisional alerts for critical threats (`ML-08`)
- Clear alert ownership visible to all operators (`PD-02`)
- Validated with a real guard on a real shift, not simulated

---

### USR-04 — The HSE manager's use case is strong but two-thirds unvalidated.
**Severity: HIGH** · **Points: 8**

**Evidence:** Fire is measured and excellent (81.8% precision, 100% recall, 6.7% FPR). `fall`, `running` (panic), and `crowd_formation` complete the HSE picture; `fall` and `running` are unmeasured, and `crowd_formation` is the weakest measured detector at 60% precision / 75% recall.

**Why it matters.** HSE is arguably your strongest commercial wedge — the fire result is genuinely compelling, the buyer has a compliance budget rather than a discretionary one, and Deluxe Paints is an existing relationship. But HSE buyers ask specific, technical, auditable questions: what is the detection time, what is the miss rate, how is it verified, and can it be included in a safety case.

Fire answers all of these well. Falls — the other anchor HSE use case — cannot be answered at all. And a 75% recall on crowd formation means one in four crowd events is missed, which needs stating plainly to an HSE buyer rather than discovered by them.

**Definition of Done:**
- `fall` and `running` measured on proper datasets (`ML-03`)
- Detection time published per HSE detector
- Documentation aligned to how HSE buyers evaluate systems, including known limitations
- An HSE-specific configuration template (`PD-08`)
- Validated with the Deluxe Paints contact against their actual compliance requirements

---

### USR-05 — The installer cannot deploy without the founder.
**Severity: HIGH** · **Points: 8**

**Evidence:** JSON editing for RTSP URLs, no guided setup (`PD-08`), no self-test, no troubleshooting documentation for field conditions.

**Why it matters.** This is `INV-06`'s blocker restated from the field. The channel strategy depends on third-party installers deploying Argus unaided. Today a deployment requires: hand-editing JSON, knowing RTSP URL formats for the specific camera brand on site, diagnosing why a stream will not open, and understanding zones and rules well enough to configure them sensibly.

Camera discovery is the sharpest specific gap — an installer should not need to know that a Hikvision RTSP path differs from a Dahua one. ONVIF discovery is a well-understood solved problem and its absence forces manual work on the least-tolerant user.

**Definition of Done:**
- Automatic camera discovery on the local network via ONVIF, with manual RTSP entry as a fallback
- A connection test with specific, actionable error messages — not "failed to open stream"
- Guided zone drawing with visual feedback
- Use-case templates so rules do not need to be authored from scratch (`PD-08`)
- An installer's field guide with a troubleshooting decision tree
- Validated by a third-party installer deploying unaided, start to finish

---

### USR-06 — Hardware requirements exclude the likely buyer.
**Severity: HIGH** · **Points: 8**

**Evidence:** Validated on an 18 GB MacBook Pro at 5 cameras, ~6 GB resident. `MemoryGuard` exists specifically because memory pressure is a live operational concern.

**Why it matters.** There is a positioning contradiction that needs resolving. Argus is pitched as a retrofit layer for *existing* CCTV — implying cost-sensitive customers who already spent their budget on cameras — while requiring a machine substantially more capable than the typical back-office PC, which is commonly an 8 GB Windows box.

The fact that `MemoryGuard` was necessary on an 18 GB machine at 5 cameras is the signal here. It is good engineering, and it also indicates the system is operating close to its envelope on hardware well above what a small retailer has.

Until this is characterised (`ARCH-05`), every sales conversation carries an unquantified hardware cost that may exceed the software's price.

**Definition of Done:**
- Minimum viable hardware established by measurement, including whether an 8 GB machine can run 4 cameras usefully
- A published hardware tier table: cameras supported per tier, with expected latency
- A recommended reference build with sourcing and cost for the target market
- Quantised or smaller model variants evaluated for lower-end hardware, with the accuracy trade-off measured, not assumed
- Pricing that accounts for hardware where the customer needs to buy it (`INV-04`)

---

### USR-07 — Nothing tells the user the system is actually watching.
**Severity: MEDIUM** · **Points: 3**

**Evidence:** No heartbeat, no periodic confirmation, no scheduled self-test.

**Why it matters.** Silence is ambiguous, and this is the third appearance of the system's characteristic failure mode. A quiet night and a dead engine produce identical user experiences. Over time, silence erodes trust in either direction — either the user assumes it is working when it is not, or they assume it is broken when it is fine and stop relying on it.

Traditional alarm systems solved this decades ago with a blinking light. The equivalent here is a visible, continuously updating liveness signal plus a periodic proof that the whole chain — camera to detector to gate to notification — still works end to end.

**Definition of Done:**
- A persistent liveness indicator per camera showing last-frame-processed time
- A daily "all systems normal" notification, on by default, with a per-user opt-out
- A scheduled end-to-end self-test that exercises the full path including notification delivery, and alerts on failure
- Camera uptime in the weekly summary (`USR-02`)

---

## 9. Senior Product Manager — Consolidated backlog and prioritisation

> *Six people just told you what is wrong. My job is to decide what you actually do, in what order, and — more importantly — what you deliberately do not do.*

### 9.1 The shape of the problem

**51 concerns. 389 story points. Roughly 26 developer-weeks — about six months for one full-time engineer.**

| Severity | Count | Points |
|---|---|---|
| CRITICAL | 12 | 129 |
| HIGH | 25 | 213 |
| MEDIUM | 14 | 47 |
| **Total** | **51** | **389** |

| Perspective | Concerns | Points |
|---|---|---|
| AI/ML | 9 | 90 |
| Systems Architect | 9 | 76 |
| Investor | 7 | 71 |
| Product Designer | 8 | 60 |
| Users | 7 | 53 |
| Software Engineer | 11 | 39 |

Two observations from that table that should shape your thinking:

**The SWE column is the smallest.** Eleven concerns, 39 points — the lowest total of any perspective despite having the most items. That is a real compliment. The code is in good shape; the fixes are contained and cheap. Your problems are not engineering problems.

**ML and Architecture dominate.** 166 points between them — 43% of the backlog. That is the signature of a system that was built to prove a thesis and now has to become a product that runs somewhere you cannot see it.

### 9.2 The prioritisation principle

I am ordering by **what unblocks the next thing**, not by severity. A CRITICAL item that blocks nothing can wait behind a MEDIUM item that unblocks a pilot. Specifically:

1. **Do not break Saturday.** Three days out, the only acceptable work is work that reduces demo risk or answers an investor question. Nothing else. A refactor that breaks the demo costs more than every point in this document.
2. **Legal and safety gate the pilot.** The moment real footage of real people lands on a disk, `ARCH-01`, `ARCH-02`, and `INV-05` stop being roadmap items and become live liabilities. These are not negotiable and they are not deferrable past first deployment.
3. **The pilot is the highest-value single item in the backlog.** `INV-01` resolves or sharpens roughly a third of everything else. Every week without a pilot is a week of building on assumptions.
4. **Measured beats built.** You have ten detectors and three measurements. Adding an eleventh detector is negative value. Measuring a fourth is positive value.
5. **Nothing ships that the system cannot tell you about.** Logging and health monitoring precede features, because you cannot support what you cannot see.

### 9.3 Priority tiers

| Tier | Milestone | Points | Duration (1 dev) | Gate |
|---|---|---|---|---|
| **P0** | Investor demo ready | **11** | 3 days | Sat 22 Aug 2026 |
| **P1** | Pilot-legal & pilot-supportable | **61** | ~4 weeks | Before real footage is recorded |
| **P2** | Pilot succeeds & produces a case study | **89** | ~6 weeks | Before customer #2 |
| **P3** | Credible & scalable | **122** | ~8 weeks | Before a funded raise closes |
| **P4** | Growth & defensibility | **106** | ~7 weeks | Post-raise |

---

### 9.4 P0 — Before Saturday 22 August (11 points, 3 days)

**Rule: nothing here touches the detection path.** No refactors, no new detectors, no architecture changes. Every item is additive, reversible, and either lowers demo risk or answers a question you will be asked.

| ID | Item | Pts | Why this, now |
|---|---|---|---|
| **SWE-03** | Add test job to CI, `needs: test` on build | **1** | Ten minutes of YAML. "Do you run tests in CI?" is a diligence question with a currently-bad answer. Also stops you shipping a red build in demo week. |
| **SWE-06a** | Refuse to start with `provider="mock"` unless explicitly allowed; add a visible gate-error counter | **2** | **Demo risk.** The mock gate confirms everything. If a config path selects it on stage, your product silently inverts and you would not know. This is insurance. |
| **ML-04a** | Add confidence intervals to `docs/NUMBERS.md`; restate "100% recall" as "100% on 9 positives (95% CI lower bound ~66%)" | **2** | **Credibility protection.** If you claim "we never miss a fire" and a technical investor checks n=9, you lose the room. Stating it correctly yourself converts a weakness into evidence of rigour. |
| **PD-06a** | Reframe the noise-suppression screen in business terms | **3** | Your best screen, currently framed as an engineering metric. Highest-leverage demo change available in the time. |
| **INV-03a** | Move superseded plans to `docs/archive/`; ensure `README.md` reflects current architecture | **2** | If anyone opens the repo, 19 mixed docs including `48HR_PLAN.md` and `SUNDAY_DEMO.md` read as disorganised. Cheap tidy. |
| **SWE-11a** | Publish SHA-256 checksums with release assets | **1** | Partial answer to "is this signed?" until proper signing lands in P1. |

**Explicitly NOT before Saturday:** the `detector/core.py` split, authentication, retention, the installer rework, and any new detector. All are important. None are worth the risk of a broken demo three days out.

**Demo-day guidance from this audit:**
- Lead with fire: *"Raw detectors: 90% false alarms. With TrueSight: 6.7%, zero fires missed."*
- Volunteer the sample sizes before you are asked. It reads as confidence, not weakness.
- Volunteer that 3 of 10 detectors are measured and that measuring the rest is the funded plan. Being caught understating this is far worse than saying it.
- Have an answer ready for latency: *"18–20 seconds, because we chose precision. For life-safety threats we alert immediately and confirm after — that is `ML-08` on our roadmap."*
- Have an answer ready for bus factor. It will come up after you leave.

---

### 9.5 P1 — Pilot-legal and pilot-supportable (61 points, ~4 weeks)

**Gate: none of this is optional once real footage of real people is recorded.** Do not start a pilot before this tier is complete. Starting one without it creates legal exposure you cannot retroactively fix, because the data will already exist.

| Order | ID | Item | Pts | Rationale |
|---|---|---|---|---|
| 1 | **SWE-01** | Structured logging with rotation + diagnostics bundle | 5 | Everything else in this tier is undiagnosable without it. Do it first. |
| 2 | **SWE-02** | All 64 broad handlers log; per-component error counters | 3 | Completes `SWE-01`. Cheap once logging exists. |
| 3 | **SWE-06b** | Gate fail-visible policy; unverified alerts reach the operator | 1 | Safety defect. Finish what P0 started. |
| 4 | **INV-05** | Privacy policy, DPIA, ToS, liability position, insurance | 5 | **Must precede recording.** Lead time on insurance — start immediately, in parallel. |
| 5 | **ARCH-01** | Retention policy, purge job, legal hold, disk monitoring | 5 | GDPR/NDPR storage limitation. Legal hold is the part not to skip. |
| 6 | **ARCH-02** | Auth, roles, audit log, encryption at rest | 13 | Four gaps, one tier. Audit log matters most — it is what makes the footage evidentially useful. |
| 7 | **ARCH-03** | Health endpoint, opt-in heartbeat, alerting, update mechanism | 13 | You cannot operate a site you cannot see. Privacy-preserving by design: health only, never frames. |
| 8 | **ARCH-09** | Camera offline states, reconnection, offline alerts | 3 | "Camera down and nobody noticed" is the worst pilot outcome. |
| 9 | **USR-01** | One installer that bundles everything, signed and notarised | 13 | Includes the rest of `SWE-11`. Without this you deploy the pilot personally — acceptable once, fatal as a pattern. |

**Milestone definition:** a pilot site can be deployed by someone other than you, operates unattended, tells you when it breaks, deletes data on schedule, and is legally defensible.

---

### 9.6 P2 — Make the pilot succeed (89 points, ~6 weeks)

**Gate: this tier turns a running deployment into a reference customer and a case study.** `INV-01` runs concurrently with the rest — the pilot is live while you improve around it, using real feedback.

| Order | ID | Item | Pts | Rationale |
|---|---|---|---|---|
| 1 | **INV-01** | Pilot live: agreement, 30 days operation, case study | 21 | The highest-value item in the entire backlog. Resolves or sharpens a third of everything else. |
| 2 | **ML-02** | Prompt regression suite, versioned prompts, pinned model | 8 | Before pilot data starts influencing anything. Protects the 26-point precision swing from recurring silently. |
| 3 | **PD-02** | Triage workflow: states, ownership, handover, incident record | 13 | The largest product gap. A pilot without triage produces a frustrated operator and a weak case study. |
| 4 | **PD-04** | Mobile response view, authenticated, Telegram deep-links | 13 | With `PD-02`, resolves `USR-03`. The guard's actual day. |
| 5 | **ML-08** | Two-tier alerting: immediate provisional for critical, confirm after | 8 | Removes the latency objection and de-risks the always-on critical detectors. |
| 6 | **USR-03** | Validate the guard experience on a real shift | 8 | Observation, not construction. The validation is the deliverable. |
| 7 | **ARCH-08** | Config backup and one-click restore; DB integrity check | 8 | Losing a pilot customer's configuration would end the pilot. |
| 8 | **USR-02** | Automatic weekly owner summary | 5 | The renewal artefact. Also your case-study raw material. |
| 9 | **PD-06b** | Complete the value surface; site-specific ROI inputs | 2 | Finishes P0's demo work into a real feature. |
| 10 | **USR-07** | Liveness indicators, daily all-normal notification, self-test | 3 | Silence is ambiguous. Closes the system's characteristic failure mode. |

**Milestone definition:** a real customer runs Argus daily, an operator uses it from their phone without complaint, the owner receives weekly value evidence, and you have a written case study and a reference call.

---

### 9.7 P3 — Credible and scalable (122 points, ~8 weeks)

**Gate: this is what makes the claims survive technical diligence and lets you sell beyond one site.**

| ID | Item | Pts |
|---|---|---|
| **ML-03** | Measure the 7 unvalidated detectors — `weapons` and `violence` first (always-on, critical priority) | 21 |
| **ML-04b** | Grow eval sets to 100+ per detector; realistic class balance; per-clip publication | 6 |
| **ML-05** | Per-camera drift detection and re-validation procedure | 8 |
| **ML-07** | Per-detector failure analysis and published failure taxonomy | 5 |
| **ML-09** | Dataset cards, licence review, leakage verification | 3 |
| **ARCH-05** | Capacity model: load-test 10/20/40 cameras, publish hardware tiers | 5 |
| **ARCH-06** | Schema migrations with backup and forward-testing from `v0.9.0` | 5 |
| **ARCH-07** | Harden engine↔app discovery; validate on Windows | 3 |
| **USR-06** | Minimum viable hardware measured; quantised variants evaluated | 8 |
| **USR-05** | ONVIF camera discovery, actionable errors, installer field guide | 8 |
| **USR-04** | HSE validation with Deluxe Paints against real compliance requirements | 8 |
| **PD-01** | Restructure navigation to Watch / Triage / Configure | 8 |
| **PD-03** | Three roles with role-appropriate views | 5 |
| **PD-08** | First-run wizard and use-case templates | 8 |
| **SWE-04** | Split `detector/core.py`; eval must reproduce archived numbers exactly | 8 |
| **SWE-05** | Tests for `training/`, `cli/`, `pipelines/` | 5 |
| **SWE-09** | Lint and type checking in CI; `TypedDict` on boundary payloads | 3 |
| **SWE-08** | One canonical demo DB; `dist/` untracked; Argus naming | 2 |
| **SWE-11b** | Complete code signing and notarisation | 2 |
| **SWE-10** | `pip-audit` in CI, Dependabot | 1 |

**Sequencing note:** `SWE-04` (the monolith split) is deliberately placed *after* the pilot and *after* `ML-03`. It is the riskiest change in the backlog — a behaviour-preserving refactor of the hot path — and it should happen when you have both a full eval suite across ten detectors to verify against and a stable pilot to detect regressions. Doing it earlier trades a manageable problem for an unmanageable one.

**Milestone definition:** every marketed capability has published measured numbers with confidence intervals, hardware requirements are known and stated, and a third-party installer can deploy without you.

---

### 9.8 P4 — Growth and defensibility (106 points, ~7 weeks)

| ID | Item | Pts |
|---|---|---|
| **ML-01** | Trained models replacing heuristics for at least 3 detectors; model cards | 21 |
| **ARCH-04** | Multi-site aggregation (pull `site_id` in the data model forward to P3 — 3 pts — even if the feature waits) | 21 |
| **INV-06** | Go-to-market: ICP, collateral, installer channel, pipeline of 10 | 13 |
| **INV-02** | Defensibility narrative built on the measurement loop and data accumulation | 13 |
| **ML-06** | Validate the feedback loop end-to-end; registry, rollback, promotion gating | 8 |
| **INV-04** | Pricing, unit economics, bottom-up market sizing | 8 |
| **INV-03b** | `ARCHITECTURE.md`, runbook, verified setup, first engineering hire | 6 |
| **PD-05** | Ask: history, saved queries, scheduled monitoring queries | 5 |
| **SWE-07** | Split the 1,429-line UI before it reaches 3,000 | 5 |
| **PD-07** | Degradation vocabulary in the UI mapped to `MemoryGuard` actions | 3 |
| **INV-07** | Feature-status register; freeze on new detectors until pilot is live | 3 |

---

### 9.9 Cross-cutting dependency chains

Three chains must be respected. Violating them means rework.

```
CHAIN 1 — OBSERVABILITY (do first, everything depends on it)
  SWE-01 logging ──► SWE-02 handlers ──► ARCH-03 health ──► ARCH-09 camera state
                 └─► SWE-06 gate errors      └─► ML-05 drift ──► ML-06 feedback loop

CHAIN 2 — LEGAL / TRUST (gates the pilot absolutely)
  INV-05 legal ──► ARCH-01 retention ──► ARCH-02 auth+audit ──► PD-03 roles ──► PD-04 mobile
                                                            └─► ML-06 label integrity

CHAIN 3 — DEPLOYABILITY (gates the channel)
  SWE-11 signing ──► USR-01 one installer ──► USR-05 ONVIF + field guide ──► INV-06 GTM
                                          └─► PD-08 first-run wizard
```

**The single most important ordering constraint:** `PD-04` (mobile) must not ship before `ARCH-02` (auth). A mobile alert view is a network-exposed endpoint. Shipping it unauthenticated would put live camera feeds on an open port on a customer's network — converting a product feature into a security incident, in a security product.

### 9.10 What I would cut

Being a PM means saying what does not get built. My recommendations:

- **`ARCH-04` multi-site (21 pts) — defer, but pull `site_id` forward.** Add the field to the data model in P3 for 3 points. Build the aggregation layer only when a chain customer is actually in the pipeline. Retrofitting the field later is expensive; building the feature speculatively is waste.
- **`ML-01` trained models (21 pts) — defer deliberately.** The heuristic-plus-VLM architecture works and is measured. Replacing heuristics with trained models is the right long-term answer, but it is not what is blocking a sale today. `ML-03` (measuring what exists) delivers more value per point than `ML-01` (improving what exists).
- **`SWE-07` UI split (5 pts) — defer until the next major UI feature.** It is not a problem at 1,429 lines. Do it when `PD-01` and `PD-02` are about to touch the file anyway, and get it nearly free.
- **`PD-05` Ask persistence (5 pts) — defer.** Genuinely differentiated, but nobody has asked for it because nobody uses the product yet. Revisit after the pilot tells you whether Ask gets used at all.

**Total deferred: 52 points — roughly 3.5 developer-weeks recovered.**

### 9.11 What I would refuse to cut

- **`INV-05` legal (5 pts).** Cheap, unglamorous, and the only item here that could end the company outright.
- **`SWE-03` CI tests (1 pt).** The single best value-per-point item in the document.
- **`ML-04a` confidence intervals (2 pts).** Your credibility is the asset. Protect it in the two places it is currently overstated.
- **`ARCH-02` audit log (part of 13 pts).** Without it, footage has no chain of custody — which undermines the product's actual deliverable, not just its compliance posture.

---

## 10. Suggested execution plan

### Sprint 0 — 19–22 August 2026 (3 days, 11 pts) · **Demo readiness**
`SWE-03` → `SWE-06a` → `ML-04a` → `PD-06a` → `INV-03a` → `SWE-11a`
**Exit:** the demo runs, the numbers are stated defensibly, and CI is green on every push.

### Sprint 1–2 — Sept 2026 (4 weeks, 61 pts) · **Pilot-ready**
Chain 1 (observability) and Chain 2 (legal/trust) in parallel; `USR-01` last.
Start `INV-05` on day one — insurance and legal review have external lead times.
**Exit:** you can legally and operationally put Argus on a customer site.

### Sprint 3–5 — Oct–Nov 2026 (6 weeks, 89 pts) · **Pilot live**
`INV-01` starts immediately and runs throughout. Build `PD-02`, `PD-04`, `ML-08` around the live deployment, using real operator feedback rather than assumptions.
**Exit:** a reference customer, a written case study, and 30 days of real uptime data.

### Sprint 6–9 — Dec 2026–Jan 2027 (8 weeks, 122 pts) · **Credible and scalable**
`ML-03` (measure everything) is the anchor. `SWE-04` late, verified against the now-complete eval suite.
**Exit:** every marketed claim is measured, and someone else can deploy it.

### Sprint 10–13 — Feb–Mar 2027 (7 weeks, 106 pts) · **Growth**
GTM, defensibility, trained models, multi-site.
**Exit:** a repeatable sales motion and a defensible technical position.

---

## 11. Closing assessment

The most useful thing I can tell you is the asymmetry, because it tells you what kind of company this is and what kind of work is ahead.

**The engineering judgement here is better than the product, and the evaluation discipline is better than the models.** You built rigour that most teams skip entirely — a harness that refuses to report fake numbers, tests that encode policy intent, honest documentation of what could not be measured and why. You skipped fundamentals that most teams do first — authentication, logging, retention, tests in CI.

That is, on balance, the better asymmetry to have. Rigour and judgement are extremely hard to retrofit into a team or a codebase. Authentication is a fortnight. The things you skipped are known, bounded, and mostly cheap. The things you did well are the things that cannot be bought.

Three sentences to carry forward:

1. **Your moat is the eval harness and the data it will accumulate — not the model.** Pitch it that way; it is both more honest and more convincing.
2. **The pilot is worth more than any feature in this backlog.** Everything you build before it is a guess, and the guesses are getting expensive.
3. **The system's characteristic failure mode is silence** — a dead camera, a dropped alert, a drifted model, and a quiet safe site all look identical. It appeared independently in `SWE-01`, `SWE-06`, `ARCH-03`, `ARCH-09`, `ML-05`, and `USR-07`. Fixing that one pattern fixes six concerns, and it is the difference between a demo and a system somebody can trust when you are not in the room.

---

*Audit performed 19 August 2026 against HEAD `77fa91e`. Every quantitative claim is reproducible with the commands in §2.10. Where an earlier verbal estimate conflicted with measurement — specifically the composition of the 1.6 GB in `runs/` — the measurement is recorded here and the estimate corrected.*
