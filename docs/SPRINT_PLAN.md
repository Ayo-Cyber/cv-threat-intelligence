# Argus — 3-Week Delivery Plan (Epics & Subtasks)

**Owner:** Ayo (solo dev)
**Planning date:** 19 Aug 2026
**Window:** Wed 19 Aug → Fri 11 Sep 2026 · **18 working days**
**Source of truth for concerns:** [`docs/AUDIT.md`](AUDIT.md) — every task below traces to an audit ID.

---

## 0. What "complete" means here

**389 story points is the full audit backlog — roughly 26 developer-weeks. That does not fit in three weeks, and no amount of velocity changes that.**

So I am not planning "finish everything." I am planning to a **specific, defensible finish line**:

> ### 🎯 The Goal
> **A system that can be legally and operationally deployed at a real pilot site, that tells you when it breaks, that a non-technical person can install, and whose claims survive technical scrutiny.**

That is **131 points across 9 epics.** Everything else is deliberately deferred, and §7 lists exactly what and why.

### Why this finish line and not another

Three weeks from now, the single highest-value thing you can own is **a live pilot**. `INV-01` in the audit resolves or sharpens about a third of the entire backlog — pricing, defensibility, latency tolerance, which features actually get used. You cannot start one until four things are true:

| Precondition | Why it's non-negotiable | Epic |
|---|---|---|
| It's legal to record | Real footage of real people = GDPR/NDPR the moment it hits disk | EP-02 |
| Access is controlled and logged | Anyone can currently disable detection with no record | EP-03 |
| It tells you when it breaks | You cannot support a site you cannot see | EP-01, EP-04 |
| Someone else can install it | Deploying it personally is fine once, fatal as a pattern | EP-05 |

This plan builds exactly those four, plus the operator experience that makes the pilot produce a *good* case study rather than just a running box.

### Velocity assumption — read this

131 points in 18 days = **~7.3 points/day**, roughly **2.4× standard solo velocity** (3 pts ≈ 1 day).

I am setting it there because your measured output in this project supports it — you shipped the eval harness, routing, NL watches, the frame publisher, and the memory guard, plus three measured detector evaluations, inside a comparable window. But I want to be straight with you: **this is an aggressive plan with no slack in it.** §7.3 pre-defines what gets cut the moment you slip, so you make that call once, calmly, rather than nightly under pressure.

---

## 1. Epic overview & dependency graph

| Epic | Title | Pts | Days | Blocked by |
|---|---|---|---|---|
| **EP-00** | Demo Readiness | 11 | 1–3 | — |
| **EP-01** | Observability Foundation | 12 | 4–6 | EP-00 |
| **EP-02** | Data Lifecycle & Legal | 10 | 1†, 6–7 | — (legal starts day 1) |
| **EP-03** | Identity, Access & Audit | 13 | 7–9 | EP-01 |
| **EP-04** | Operability & Remote Health | 13 | 9–11 | EP-01 |
| **EP-05** | One-Click Deployment | 15 | 12–15 | EP-01, EP-03 |
| **EP-06** | Alert Triage & Response | 21 | 11–14 | EP-03 |
| **EP-07** | Measurement Integrity | 18 | 4–16 ‡ | EP-00 |
| **EP-08** | Pilot Launch Kit | 18 | 15–18 | EP-01…EP-06 |
| | **TOTAL** | **131** | **18** | |

† EP-02's legal work is an **external dependency with lead time** — kick it off on day 1 and let it run in the background.
‡ EP-07 is **long-running background eval jobs**. Start runs, work on other epics while they execute, harvest results.

### Dependency graph — why the order is the order

```
EP-00 Demo ─────────────────────────────────────────────┐
   │                                                     │
   ▼                                                     ▼
EP-01 OBSERVABILITY ◄──── the foundation           EP-07 MEASUREMENT
   │   (nothing can be supported without logs)      (runs in background
   │                                                 the whole time)
   ├──────────────┬──────────────┐
   ▼              ▼              ▼
EP-03 IDENTITY  EP-04 HEALTH   EP-02 RETENTION
   │  (audit log needs           (legal kicked off
   │   logging infra)             day 1, external)
   │
   ├──────────────┬─────────────►
   ▼              ▼
EP-06 TRIAGE    EP-05 INSTALLER
(ownership needs  (first-run must set
 identity!)        credentials → needs EP-03)
   │              │
   └──────┬───────┘
          ▼
     EP-08 PILOT LAUNCH KIT
```

**The two hard ordering constraints — do not violate these:**

1. **EP-01 before everything.** Every epic after it either emits logs, reports error counters, or is undiagnosable without them. Building EP-03/04/05 first means debugging them blind, and you will lose more time than EP-01 costs.
2. **EP-03 before EP-06's mobile view.** A mobile alert view is a network-exposed endpoint. Shipping it before auth puts **live camera feeds on an open port on a customer's network** — turning a feature into a security incident, in a security product. This is the one sequencing mistake that would be genuinely damaging.

---

## 2. Sprint calendar

| Sprint | Dates | Days | Epics | Pts | Milestone |
|---|---|---|---|---|---|
| **S0** | Wed 19 – Fri 21 Aug | 3 | EP-00 | 11 | 🎤 **Investor demo Sat 22 Aug** |
| **S1** | Mon 24 – Fri 28 Aug | 5 | EP-01, EP-02, EP-03 (start), EP-07 (start) | 35 | 🔍 **System is observable & legal** |
| **S2** | Mon 31 Aug – Fri 4 Sep | 5 | EP-03 (finish), EP-04, EP-06 | 47 | 🔐 **Secure, monitored, operator-ready** |
| **S3** | Mon 7 – Fri 11 Sep | 5 | EP-05, EP-07 (finish), EP-08 | 38 | 🚀 **Pilot-deployable** |

---

## 3. Sprint 0 — Demo Readiness

### 🟥 EP-00 — Demo Readiness · 11 pts · Days 1–3

> **Goal:** Walk into Saturday with a demo that cannot silently fail, numbers that survive a technical investor checking them, and a repo that reads as disciplined.

**Hard rule for this epic: nothing touches the detection path.** No refactors, no new detectors, no architecture changes. Every task is additive and revertible. A broken demo costs more than every point in this plan.

**Definition of Done:** demo runs clean; CI green on every push; every stated number has its sample size attached; no config path can silently disable verification.

| ID | Task | Pts | Audit ref |
|---|---|---|---|
| EP-00-T1 | CI runs the test suite, gates the build | 1 | SWE-03 |
| EP-00-T2 | Mock-gate guard + gate error counter | 2 | SWE-06a |
| EP-00-T3 | Confidence intervals on all published numbers | 2 | ML-04a |
| EP-00-T4 | Business framing for the suppression screen | 3 | PD-06a |
| EP-00-T5 | Docs consolidation | 2 | INV-03a |
| EP-00-T6 | Release checksums | 1 | SWE-11a |

---

**EP-00-T1 · CI runs the test suite and gates the build · 1 pt**

*What:* Add a `test` job to `.github/workflows/build-app.yml` running on every push and PR. Add `needs: test` to the build job.

*Why:* You have 32 test files and 3,380 lines of test code that a release build never consults. Right now, tagging a version with failing tests produces three signed installers with the bug baked in. This is the highest value-per-point task in the entire plan — roughly ten minutes of YAML.

*Acceptance:*
- [ ] `test` job runs the full suite on push + PR
- [ ] `build` declares `needs: test`; a red suite blocks installer creation
- [ ] Branch protection on `main` requires the test job
- [ ] Status badge in `README.md`
- [ ] Verified: push a deliberately failing test, confirm the build is blocked, revert

---

**EP-00-T2 · Mock-gate guard and gate error counter · 2 pts**

*What:* In `cvti/verification/gate.py`, refuse engine start when `provider="mock"` unless `ARGUS_ALLOW_MOCK_GATE=1` is set. When allowed, show a permanent red banner in the app. Add a rolling counter of gate errors surfaced in the System panel.

*Why:* **This is demo insurance.** `_mock_response()` unconditionally returns `confirmed=True`. If any config path selects the mock provider on stage, every candidate passes the gate — your product's entire value proposition inverts, live, and nothing on screen would tell you. The error counter covers the other failure: Ollama being slow or unreachable mid-demo.

*Acceptance:*
- [ ] Engine refuses to start with mock provider unless the env var is explicitly set
- [ ] When explicitly allowed, a permanent red "UNVERIFIED — MOCK GATE" banner shows in the UI
- [ ] Gate error count and last error visible in the System panel
- [ ] Test covering the guard in both states
- [ ] Verified by running the actual demo config end-to-end

---

**EP-00-T3 · Confidence intervals on all published numbers · 2 pts**

*What:* Update `docs/NUMBERS.md` and `tools/make_numbers_sheet.py` to emit Wilson score intervals alongside every point estimate. Restate the fire headline.

*Why:* You currently claim **100% recall on fire**, measured on **9 positive clips**. The 95% CI lower bound is approximately 66%. That is still a good result — but if you say "we never miss a fire" and a technical investor checks the sample size, you lose the room and every other number becomes suspect. Stating the interval yourself converts your single biggest credibility exposure into a demonstration of rigour. This is 2 points that protect the whole pitch.

*Acceptance:*
- [ ] Wilson 95% CI computed and displayed for every precision/recall/FPR figure
- [ ] `n` shown explicitly next to every metric
- [ ] Fire headline restated: *"100% recall on 9 held-out positive clips (95% CI: 66–100%)"*
- [ ] `tools/make_numbers_sheet.py` generates this automatically from archived metrics — never hand-maintained
- [ ] Regenerated `docs/NUMBERS.md` committed

---

**EP-00-T4 · Business framing for the suppression screen · 3 pts**

*What:* Reframe the noise-suppression panel from engineering metrics to business outcomes. Add site-configurable inputs (average incident value, guard hourly cost).

*Why:* This is the strongest single screen in the product — it's the one place the interface tells the whole story in a glance: *"raw detectors would have shown you 201 alerts; TrueSight showed you 28."* It is currently framed as a suppression percentage, which is an engineering metric. The buyer cares about alerts avoided, attention-hours saved, incidents caught. Same data, different frame, materially better demo — and it becomes the renewal artefact later.

*Acceptance:*
- [ ] Panel leads with: incidents detected, false alarms prevented, attention-hours saved
- [ ] Raw-vs-verified comparison shown as the headline
- [ ] Configurable site values feeding the estimates
- [ ] Every figure traceable to underlying events — no unfalsifiable numbers
- [ ] Reads correctly with the demo dataset loaded

---

**EP-00-T5 · Docs consolidation · 2 pts**

*What:* Move superseded planning docs (`48HR_PLAN.md`, `SUNDAY_DEMO.md`, `GET_WEAPON_CHECKPOINT_TODAY.md`, `MVP_POC_PLAN.md`, and similar) into `docs/archive/`. Ensure `README.md` describes current architecture.

*Why:* `docs/` currently holds 19 files mixing live architecture with historical sprint plans. If anyone opens this repo during diligence, they cannot tell current from superseded — it reads as disorganised, which undercuts the impression the code itself makes.

*Acceptance:*
- [ ] `docs/archive/` created with superseded documents moved
- [ ] `docs/README.md` index explaining what each remaining doc is for
- [ ] Root `README.md` accurate to current architecture, quickstart verified
- [ ] `AUDIT.md` and `SPRINT_PLAN.md` linked from the index

---

**EP-00-T6 · Release checksums · 1 pt**

*What:* Emit SHA-256 sums for every release asset in the build workflow; publish them in the release body.

*Why:* Partial answer to "is this signed?" until proper notarisation lands in EP-05. Cheap, and it is the minimum expected of a security product's distribution.

*Acceptance:*
- [ ] SHA-256 computed for each asset during build
- [ ] Published in the GitHub release body
- [ ] `README.md` documents verification steps

---

### 🎤 Demo-day guidance (Sat 22 Aug)

**Lead with fire:** *"Raw detectors give a 90% false-alarm rate. With TrueSight: 6.7%, and zero fires missed."* One sentence, measured, held-out, memorable.

**Volunteer your limits before you're asked** — it reads as confidence, not weakness:
- Sample sizes (n=36/39/38) and confidence intervals
- 3 of 10 detectors measured, with measuring the rest as the funded plan
- Zero pilots — framed as the next milestone, not an omission

**Have these three answers ready:**

| Question | Answer |
|---|---|
| *"20-second latency?"* | "Deliberate — we buy precision with it. For life-safety threats we alert instantly and confirm after. That's `ML-08`, shipping in three weeks." |
| *"What stops a funded competitor?"* | "Not the model — that's public. The moat is the eval harness and the labelled data it accumulates. Competitors won't build it: it's unglamorous and produces unflattering numbers." |
| *"What if you get hit by a bus?"* | Have a real answer. It will come up after you leave the room. |

---

## 4. Sprint 1 — Observable & Legal (35 pts, Days 4–8)

### 🟧 EP-01 — Observability Foundation · 12 pts · Days 4–6

> **Goal:** The system can tell you what it is doing and when it fails. Every subsequent epic depends on this.

**Why this is first.** You have **260 `print()` calls and zero `logging` imports**. Combined with **64 broad `except Exception` handlers**, the system currently degrades completely silently — it is designed to survive component failure but has no way to tell you a component failed. This is the mechanism behind the crowd-detector bug, where events were discarded into the void for two days and presented as a detection-quality problem.

Building EP-03, EP-04, or EP-05 before this means debugging them blind. You will lose more time than this epic costs.

**Definition of Done:** any failure anywhere produces a retrievable, attributed, timestamped record; the app can export a diagnostics bundle; no failure class is silent.

| ID | Task | Pts | Audit ref |
|---|---|---|---|
| EP-01-T1 | Logging infrastructure with rotation | 3 | SWE-01 |
| EP-01-T2 | Convert 260 `print()` calls | 2 | SWE-01 |
| EP-01-T3 | All 64 broad handlers log + error counters | 3 | SWE-02 |
| EP-01-T4 | Gate fail-visible policy | 1 | SWE-06b |
| EP-01-T5 | Camera connection state machine | 3 | ARCH-09 |

---

**EP-01-T1 · Logging infrastructure with rotation · 3 pts**

*What:* Create `cvti/logging_setup.py` exposing `get_logger(__name__)`. Rotating file handler (10 MB × 5) writing to `<output_dir>/logs/argus.log`, plus a console handler. Level from `ARGUS_LOG_LEVEL`, default `INFO`. Add a "Download diagnostics" button that zips the log directory.

*Why:* Today, a customer says *"it stopped alerting last night"* and you cannot answer. You would have to reproduce it on your own machine. That is not a support process. In the packaged PyInstaller build, `print()` output goes nowhere the user can retrieve at all.

*Acceptance:*
- [ ] `get_logger(__name__)` returns a configured logger with module attribution
- [ ] Rotating file handler; logs survive restart; rotation verified
- [ ] `ARGUS_LOG_LEVEL` respected
- [ ] Works inside the PyInstaller bundle (verify explicitly — path resolution differs)
- [ ] "Download diagnostics" produces a zip of logs + health snapshot, **excluding evidence frames**
- [ ] Test asserting the log file is created and an exception in a detector produces a record

---

**EP-01-T2 · Convert 260 `print()` calls · 2 pts**

*What:* Replace every `print()` in `cvti/` with the appropriate level. Explicitly exempt `cvti/cli/` — that is legitimately user-facing UI — with a comment stating why.

*Why:* Mechanical but load-bearing. Until this lands, the logging infrastructure from T1 has nothing flowing through it.

*Acceptance:*
- [ ] `grep -rn 'print(' cvti --include='*.py'` returns only `cvti/cli/` matches
- [ ] Each `cli/` exemption carries a one-line comment
- [ ] Levels chosen sensibly: `debug` for per-frame, `info` for lifecycle, `warning` for degradation, `error` for failure
- [ ] No per-frame path logs at `info` (noise + disk)

---

**EP-01-T3 · Broad handlers log, with per-component error counters · 3 pts**

*What:* Every one of the 64 `except Exception` handlers logs with `exc_info=True`. Add per-component error counters surfaced in the System panel. Rate-limit logging in hot per-frame paths (first occurrence + every Nth).

*Why:* The *intent* of broad catching is correct — one bad detector must not kill the camera loop. The *implementation* is wrong in one way: catching is not handling. Right now you cannot distinguish "this detector correctly found nothing" from "this detector has thrown on every frame for a week."

*Acceptance:*
- [ ] All 64 handlers log with `exc_info=True`
- [ ] System panel shows per-component: frames processed, error count, last error
- [ ] Components exceeding a 10% error rate raise a visible degradation warning
- [ ] Hot paths rate-limit — a persistent failure cannot itself fill the disk
- [ ] Test: an injected detector exception produces a log record *and* increments its counter

---

**EP-01-T4 · Gate fail-visible policy · 1 pt**

*What:* In `gate.py`, distinguish parse/transport failures from genuine rejections via an explicit `error` field on `VerificationResult`. Live path defaults to **fail-visible**: the alert reaches the operator flagged `UNVERIFIED — TrueSight could not decide`.

*Why:* This is a real safety defect. `_parse_response()` currently returns `confirmed=False` on any exception — indistinguishable from TrueSight examining the frame and deciding it is safe. A real fire could be dropped silently. Your **eval harness already handles this correctly** (`GateUnavailable`, with a comment explaining exactly why); the live path does not. Also: when a verifier cannot render a verdict, the safe default for a safety system is to surface it to a human, not discard it.

*Acceptance:*
- [ ] `VerificationResult.error` distinguishes error from rejection
- [ ] Live path surfaces unverified alerts, clearly marked
- [ ] Fail policy configurable; default is fail-visible
- [ ] Tests: malformed JSON, truncated JSON, empty response, connection refused

---

**EP-01-T5 · Camera connection state machine · 3 pts**

*What:* Explicit per-camera state: `Connected` / `Reconnecting` / `Offline`, with time-in-state. Automatic reconnection with exponential backoff. An offline camera raises its own alert through normal routing after a configurable grace period.

*Why:* The tamper detector answers *"did someone cover this camera?"* It does not answer *"has this camera been unreachable for six days?"* An offline camera produces no alerts — which is **indistinguishable from a camera watching a quiet area**. The customer believes they have coverage they do not have. In a security product, false confidence is worse than visible failure.

*Acceptance:*
- [ ] Three explicit states with transitions and time-in-state, logged
- [ ] Reconnection with exponential backoff; attempt history visible
- [ ] Offline alert fires after a configurable grace period
- [ ] Camera status prominent in the UI — never inferred from absence of alerts
- [ ] Tested against a stream that is killed and restored mid-run

---

### 🟧 EP-02 — Data Lifecycle & Legal · 10 pts · Day 1 (kickoff) + Days 6–7

> **Goal:** It is legal to record real people, and evidence does not accumulate forever.

**⚠️ Start EP-02-T1 on Day 1, in parallel with EP-00.** Legal review and insurance have external lead times measured in weeks, not days. If you start this in week 3 it will not be ready, and it gates the pilot absolutely.

**Definition of Done:** you can point a lawyer at the retention policy and the paperwork, and put a camera on a real person without exposure.

| ID | Task | Pts | Audit ref |
|---|---|---|---|
| EP-02-T1 | Legal paperwork (external, starts Day 1) | 5 | INV-05 |
| EP-02-T2 | Retention policy + purge job + legal hold | 5 | ARCH-01 |

---

**EP-02-T1 · Legal paperwork · 5 pts · ⏰ START DAY 1**

*What:* Privacy policy, DPIA (GDPR + NDPR), terms of service with explicit limitation of liability, a DPA template for pilot customers, site signage guidance, and professional indemnity insurance quoted and obtained.

*Why:* Two exposures. **Data protection:** the system processes identifiable images of people; with no retention policy, no access control, and no audit trail, a deployment today would be difficult to defend if challenged. **Liability:** recall is measured at 88.9% (theft) and 75% (crowd) — misses are known, expected, and *quantified by you*. That honesty is good practice, but it means a customer can point at a documented miss rate. Your contractual position must be explicit: Argus is an **assistive layer that reduces operator load, not a guarantee of detection**, and does not replace human security.

Getting this wrong once could end the company. Getting it right is a few thousand in fees.

*Acceptance:*
- [ ] Privacy policy + DPIA covering GDPR and NDPR
- [ ] ToS with limitation of liability and explicit "assistive, not a guarantee" language
- [ ] DPA template for pilot customers
- [ ] Signage/notification guidance for customer sites
- [ ] Professional indemnity insurance **obtained** (not just quoted)
- [ ] Legal review complete **before any real footage is recorded**

---

**EP-02-T2 · Retention, purge, and legal hold · 5 pts**

*What:* Configurable per-site retention (default 30 days). Scheduled purge deleting frames, clips, **and** the corresponding DB rows together. **Legal hold**: evidence attached to an unresolved incident or explicitly flagged is exempt and visibly marked. Disk-usage warning threshold and an emergency purge-oldest-first path.

*Why:* `grep` confirms **zero** retention logic in the codebase. Two consequences. *Operationally:* an unattended edge PC with no purge fills its disk, and when it does, writes fail and evidence stops being recorded at exactly the moment it is most needed — with nobody watching. *Legally:* storage limitation is not optional under GDPR/NDPR. A system with no deletion path cannot honour an erasure request or answer a procurement questionnaire.

**The design tension to respect:** purge must never delete evidence attached to an open incident. Blind time-based deletion would destroy the exact records a customer needs.

*Note:* the 1.6 GB currently in `runs/` is mostly **training checkpoints**, not evidence (evidence is ~33 MB). This task stands on the confirmed absence of purge logic, not on that figure.

*Acceptance:*
- [ ] Retention configurable per site, default 30 days
- [ ] Scheduled purge removing frames, clips, and DB rows atomically — no orphans
- [ ] Legal hold exempts flagged/open-incident evidence, visibly marked in UI
- [ ] Disk warning threshold surfaced; emergency purge before disk-full
- [ ] Export path so customers can extract evidence before expiry
- [ ] Retention policy documented in `README.md` + customer-facing privacy note
- [ ] Tests: purges what it should, retains legal-hold items, leaves no orphaned rows or files

---

## 5. Sprint 2 — Secure, Monitored, Operator-Ready (47 pts, Days 7–14)

### 🟨 EP-03 — Identity, Access & Audit · 13 pts · Days 7–9

> **Goal:** Only authorised people can use the system, and everything consequential leaves a record that cannot be edited from inside the app.

**Why it matters.** Four gaps grouped because they share one root cause — built for a single trusted operator on a machine they own — and because procurement always evaluates them together. `grep` confirms **0 files** with auth, **0** with audit logging, **0** with encryption.

The most sensitive failure is not misuse of cameras. It is that **anyone with access can silently disable detection, and there is no record that they did.** For a security product, that is a contradiction in terms.

The audit log deserves special emphasis: without it, footage has no chain of custody. That doesn't just fail compliance — it **undermines the product's actual deliverable**, since the purpose of the system is producing usable evidence.

**Definition of Done:** identity is required, roles are enforced, every consequential action is recorded in an append-only log, and a stolen disk does not yield plaintext footage.

| ID | Task | Pts | Audit ref |
|---|---|---|---|
| EP-03-T1 | Local accounts + authentication | 4 | ARCH-02 |
| EP-03-T2 | Three roles with enforcement | 3 | ARCH-02, PD-03 |
| EP-03-T3 | Append-only audit log | 4 | ARCH-02 |
| EP-03-T4 | Encryption at rest + `SECURITY.md` | 2 | ARCH-02 |

---

**EP-03-T1 · Local accounts and authentication · 4 pts**

*What:* Local user accounts with `argon2` or `bcrypt` password hashing. Enforced at **both** the app **and** the frame-publisher HTTP endpoint. Session timeout. Forced credential change on first run.

*Why:* Today anyone with physical or network access has complete control: view all cameras, view all evidence, change every rule, disable every detector. Critically, **the frame publisher must be covered too** — EP-06 exposes it to the network for mobile access, and an unauthenticated frame endpoint would put live camera feeds on an open port.

*Acceptance:*
- [ ] Accounts with securely hashed passwords (never plaintext, never reversible)
- [ ] Auth enforced on the app **and** every frame-publisher route
- [ ] Session timeout, configurable
- [ ] First run forces a credential change — no deployment ships with a default password
- [ ] Failed login attempts rate-limited and logged
- [ ] Tests: unauthenticated requests to every endpoint are rejected

---

**EP-03-T2 · Three roles with enforcement · 3 pts**

*What:* `Owner`, `Operator`, `Installer` — each with a default landing surface and enforced permissions. Operators do not see detector configuration.

*Why:* The interface currently assumes installer, operator, and owner are one person. In a real deployment they are three people with near-disjoint needs. Exposing all surfaces to all three makes the product harder for each, and — more importantly — lets an operator accidentally disable a detector.

*Acceptance:*
- [ ] Three roles, enforced server-side (not just hidden in the UI)
- [ ] Role-appropriate default landing surface
- [ ] Operators cannot access configuration; enforcement tested
- [ ] Role changes are audit-logged

---

**EP-03-T3 · Append-only audit log · 4 pts**

*What:* An append-only log capturing: login attempts, footage access, rule/detector changes, alert resolutions, evidence export, purge events, and role changes. Stored **separately from `events.db`** and never modifiable through the application.

*Why:* This is the task that makes footage evidentially useful. Video with no chain of custody and no tamper-evident access record is materially weaker if a customer needs it in a dispute or prosecution. It also answers the single most important security question: *who disabled that detector, and when?*

*Acceptance:*
- [ ] Append-only; no application code path can modify or delete entries
- [ ] Separate store from `events.db`
- [ ] Captures all seven event classes above with actor, timestamp, and target
- [ ] Viewable and exportable by Owner role only
- [ ] Tampering is detectable (hash chain or equivalent)
- [ ] Test asserting no API path can mutate an existing entry

---

**EP-03-T4 · Encryption at rest and `SECURITY.md` · 2 pts**

*What:* Document OS-level full-disk encryption as the v1 requirement, verified during install. Write `SECURITY.md` covering the model, plus an answer sheet for standard procurement questionnaires.

*Why:* A stolen or decommissioned edge PC currently yields every recorded frame in plaintext plus all `events.db` metadata. Full-disk encryption is an acceptable, documented v1 — application-level encryption is the follow-up, not a blocker.

*Acceptance:*
- [ ] Install process verifies and requires disk encryption
- [ ] `SECURITY.md` documenting the security model and threat model
- [ ] Procurement questionnaire answer sheet drafted
- [ ] Application-level encryption noted as a tracked follow-up

---

### 🟨 EP-04 — Operability & Remote Health · 13 pts · Days 9–11

> **Goal:** You find out a customer's box is broken before the customer does.

**Why it matters.** Ask it directly: **a customer's box dies at 3am Saturday. How do you find out?** Today, the answer is that they tell you — probably Monday, probably after a missed incident, and the conversation opens with them angry.

This is the concern that most directly caps how many customers one person can support. And it closes the system's characteristic failure mode: a quiet site and a dead system look **identical** from outside.

**Privacy constraint — non-negotiable:** telemetry carries health signals only. Uptime, camera status, error rates, disk headroom, latency. **Never** frames, event content, or anything identifying a person. This is compatible with the local-first promise, and being explicit about it is itself a selling point.

**Definition of Done:** you have a dashboard showing every deployed site's health, you are alerted when one stops reporting, and you can ship a fix without a site visit.

| ID | Task | Pts | Audit ref |
|---|---|---|---|
| EP-04-T1 | Local `/health` endpoint | 3 | ARCH-03 |
| EP-04-T2 | Opt-in heartbeat + your dashboard | 5 | ARCH-03 |
| EP-04-T3 | Signed update mechanism | 4 | ARCH-03 |
| EP-04-T4 | Liveness + daily self-test | 1 | USR-07 |

---

**EP-04-T1 · Local `/health` endpoint · 3 pts**

*What:* An endpoint reporting per-camera status, gate reachability and latency, disk headroom, memory level, error counters (from EP-01-T3), and uptime.

*Why:* The foundation for everything else in this epic, and immediately useful during the pilot for diagnosing over the phone.

*Acceptance:*
- [ ] `/health` returns structured JSON covering all six signal classes
- [ ] Authenticated (EP-03-T1)
- [ ] Reflects real state — verified by killing a camera and a gate and observing the change
- [ ] Consumed by the app's System panel

---

**EP-04-T2 · Opt-in heartbeat and central dashboard · 5 pts**

*What:* A periodic heartbeat to a central dashboard, **opt-in and off by default**, carrying health metrics only, with a publicly documented payload schema. Alerting to you when a site stops heartbeating or reports degradation.

*Why:* Without this you cannot detect a dead engine, a dead camera, a full disk, or a stalled VLM at a site you cannot physically reach.

*Acceptance:*
- [ ] Heartbeat off by default; explicit opt-in during setup
- [ ] Payload schema documented publicly; contains **no** frames or event content
- [ ] Dashboard showing every site's status
- [ ] Alerts you on missed heartbeat or reported degradation
- [ ] Customer can view exactly what is transmitted, at any time
- [ ] Works through typical NAT/firewall (outbound only)

---

**EP-04-T3 · Signed update mechanism · 4 pts**

*What:* The ability to deliver a signed update without a site visit, with rollback.

*Why:* During a pilot you *will* need to ship fixes. Without this, every fix is a site visit or a phone call walking someone through a manual reinstall — which does not scale past one customer and burns the goodwill the pilot exists to build.

*Acceptance:*
- [ ] Update check, download, verify signature, apply, restart
- [ ] Signature verification mandatory — refuses unsigned updates
- [ ] Rollback to previous version on failure
- [ ] Customer controls timing; never auto-updates mid-shift
- [ ] Tested through a full upgrade **and** a rollback cycle

---

**EP-04-T4 · Liveness indicator and daily self-test · 1 pt**

*What:* A persistent per-camera liveness indicator showing last-frame-processed time. A daily "all systems normal" notification (on by default, opt-out). A scheduled end-to-end self-test exercising the full chain **including notification delivery**.

*Why:* Silence is ambiguous. Traditional alarm systems solved this decades ago with a blinking light. The self-test matters most: it verifies camera → detector → gate → notification actually works, rather than assuming.

*Acceptance:*
- [ ] Per-camera last-frame-processed timestamp always visible
- [ ] Daily all-normal notification, opt-out available
- [ ] Scheduled self-test covering the full path including delivery
- [ ] Self-test failure raises an alert

---

### 🟨 EP-06 — Alert Triage & Response · 21 pts · Days 11–14

> **Goal:** A guard can see what needs them, claim it, act on it from their phone, and hand it over at shift change.

**⚠️ Blocked by EP-03.** Ownership requires identity. The mobile view requires authentication. Do not start T3 before EP-03-T1 is done.

**Why it matters.** This is the **largest product gap in the system**. Sorting is not triage; colour is not triage. The alert-fatigue reduction you've achieved is delivered **entirely by TrueSight (the model) and not at all by the interface**. You cut alerts 86% with a model; the remaining 14% still land in an undifferentiated list.

From the guard's seat the current experience is: *"I get told about things I can't do anything about from where I'm standing."* If that's how the shift feels, they stop trusting the alerts — and a monitoring product the monitor ignores has zero value regardless of its precision.

**Definition of Done:** two guards on a shift never both respond to the same alert, neither ignores one, and the incoming shift knows what happened.

| ID | Task | Pts | Audit ref |
|---|---|---|---|
| EP-06-T1 | Alert state machine + ownership | 5 | PD-02 |
| EP-06-T2 | Incident record + shift handover | 4 | PD-02 |
| EP-06-T3 | Mobile response view | 8 | PD-04, USR-03 |
| EP-06-T4 | Two-tier alerting for critical threats | 4 | ML-08 |

---

**EP-06-T1 · Alert state machine and ownership · 5 pts**

*What:* Model states explicitly: `NEW → ACKNOWLEDGED (by user X at time T) → RESOLVED (outcome, note)`. Acknowledging **claims** the alert and shows the claimant to everyone else. A "needs attention now" view showing only unacknowledged alerts above a priority threshold, defaulting to one at a time.

*Why:* There is currently no ownership concept at all. With two guards on shift, both respond or neither does. The "one at a time" default matters: at 2am a guard needs a single next action, not a list to triage.

*Acceptance:*
- [ ] Three explicit states with enforced transitions
- [ ] Acknowledging claims the alert; claimant visible to all users
- [ ] "Needs attention now" view, one item at a time by default
- [ ] Resolution captures an outcome and a free-text note
- [ ] All transitions audit-logged (EP-03-T3)
- [ ] Resolution outcomes feed `cvti/feedback/`

---

**EP-06-T2 · Incident record and shift handover · 4 pts**

*What:* A per-alert incident record: frames, VLM reasoning, who responded, what they concluded — exportable as PDF. A shift handover summary of the last N hours: what fired, what was resolved, what remains open.

*Why:* Context currently resets at every shift change. And the incident record is what a manager reviews and what a customer would hand to an insurer or the police — it is the product's actual deliverable made tangible.

*Acceptance:*
- [ ] Incident record per alert with frames, reasoning, responder, conclusion
- [ ] PDF export
- [ ] Handover summary for a configurable window
- [ ] Open items carried across shifts and clearly flagged
- [ ] Evidence in an open incident is legal-held (EP-02-T2)

---

**EP-06-T3 · Mobile response view · 8 pts**

*What:* A mobile-responsive alert view served by the engine over the local network. Telegram alerts deep-link into it. Acknowledge, label true/false, add a note — all from a phone. **Authenticated (EP-03-T1).**

*Why:* Today the notification is mobile but the response is not — every action requires returning to a desk. That is backwards for a job defined by movement. This inverts the product's core promise: Argus is sold on speed of response, and the response loop currently contains a walk-to-the-office step.

The cheap version is not an app — it's a responsive web view on the existing frame publisher. That captures most of the value.

*Acceptance:*
- [ ] Responsive view served over the local network, no cloud dependency
- [ ] Telegram deep-links open the specific alert
- [ ] Acknowledge, label, and note — all functional from a phone
- [ ] **Authenticated — verify no unauthenticated route exists** (this is the sequencing risk; test it explicitly)
- [ ] Usable one-handed on a phone screen
- [ ] Validated with a real guard on a real shift, not simulated

---

**EP-06-T4 · Two-tier alerting for critical threats · 4 pts**

*What:* `critical`-priority detectors emit an **immediate provisional alert**, then update in place with the verdict. Provisional alerts are visually distinct and labelled unverified; retraction is explicit and visible. Verification queue prioritised so critical candidates never queue behind medium ones.

*Why:* Detection is fast (163 ms per 5-camera batch) — the 18–20 s latency is entirely VLM verification. For theft that trade is clearly right. For a weapon or active violence it is not, and it interacts badly with the fact that `weapons` and `violence` are **always-on critical detectors** in `baseline_critical_v1.json`.

The architecture already contains the answer; this makes it explicit. The operator gets a provisional warning in under a second, then confirmation or retraction ~20 s later. It preserves the precision story for high-volume cases and removes the objection for life-safety ones.

*Acceptance:*
- [ ] Critical detectors emit provisional alerts in under 1 second
- [ ] Provisional alerts visually distinct and clearly labelled unverified
- [ ] In-place update on verdict; retraction explicit and visible
- [ ] Critical candidates jump the verification queue
- [ ] Latency measured and published per priority tier

---

## 6. Sprint 3 — Pilot-Deployable (38 pts, Days 12–18)

### 🟩 EP-05 — One-Click Deployment · 15 pts · Days 12–15

> **Goal:** Someone who is not you installs Argus and sees a live camera in under 30 minutes, with no terminal.

**⚠️ Blocked by EP-01 (self-test needs diagnostics) and EP-03 (first-run must set credentials).**

**Why it matters.** State this plainly: **you have shipped `.dmg` and `.zip` installers that cannot detect anything on their own.** The app installs; the intelligence does not come with it. Setup currently requires installing Python, installing Ollama, pulling a ~3 GB model, and editing JSON for RTSP URLs — and the unsigned installer triggers an OS security warning that, to a non-technical user, reads as *"this software is malware."*

A user who downloads the release, clicks past a security warning, opens the app, and finds it non-functional has had a **complete product failure at first contact.** They will conclude the product doesn't work, not that a dependency is missing.

Each step is individually defensible. Together they form a wall the typical CCTV installer — exactly the person who should deploy this — will not get over. And the drop-off is multiplicative.

**Definition of Done:** a non-technical person completes installation unaided and sees a live camera, verified by watching someone actually do it.

| ID | Task | Pts | Audit ref |
|---|---|---|---|
| EP-05-T1 | Bundle everything into one installer | 6 | USR-01 |
| EP-05-T2 | Code signing and notarisation | 2 | SWE-11b |
| EP-05-T3 | First-run wizard + self-test | 4 | PD-08, USR-01 |
| EP-05-T4 | ONVIF camera discovery | 3 | USR-05 |

---

**EP-05-T1 · Bundle everything into one installer · 6 pts**

*What:* One installer containing application, Python runtime, Ollama (or equivalent embedded inference runtime), and model weights. No terminal use at any point. If weights make the download too large, a guided in-app first-run download with progress, resume, and a plain-English explanation.

*Why:* This blocks the entire channel strategy — you cannot build an installer-led go-to-market around software that installers cannot install.

*Acceptance:*
- [ ] Single installer per platform; no separate Python or Ollama step
- [ ] Model weights bundled, or downloaded on first run with progress + resume
- [ ] Zero terminal commands in the documented install path
- [ ] Tested on a **clean** machine with no Python and no Ollama — this is the only test that counts
- [ ] Tested on macOS and Windows
- [ ] Naming corrected to Argus throughout (currently still emits "CVTI Console")

---

**EP-05-T2 · Code signing and notarisation · 2 pts**

*What:* Apple Developer ID signing + notarisation for macOS; Authenticode for Windows.

*Why:* For a **security product**, asking a customer to click past an OS malware warning during installation undermines the exact trust the product sells.

*Acceptance:*
- [ ] macOS `.dmg` signed and notarised; installs with no Gatekeeper warning
- [ ] Windows signed; no SmartScreen warning
- [ ] Signing keys documented and recoverable (ties to bus-factor)
- [ ] Verified on a machine that has never seen the app

---

**EP-05-T3 · First-run wizard and self-test · 4 pts**

*What:* A guided flow: set credentials → add camera → verify feed → draw zones → pick a use-case template → confirm detectors → **send a test alert**. Use-case templates (Retail, Warehouse/HSE, Office) preselecting sensible detectors and rules. A self-test verifying every component and reporting precisely what is missing.

*Why:* Even after installation there is currently no path from "app opens" to "my cameras are monitored" — the user faces ten navigation items with no indicated starting point. The templates are where your existing `CustomizationEngine` and baseline-rules work finally pays off in the interface. The test-alert step matters: the user confirms the notification path **before** trusting it with their security.

*Acceptance:*
- [ ] Wizard covers all seven steps in order
- [ ] Three use-case templates with sensible detector/rule defaults
- [ ] Self-test names exactly what is missing, in plain English
- [ ] "Send me a test alert" confirms end-to-end delivery
- [ ] Completable by a non-technical person in under 15 minutes, **verified by observation**

---

**EP-05-T4 · ONVIF camera discovery · 3 pts**

*What:* Automatic camera discovery on the local network via ONVIF, with manual RTSP entry as fallback. Connection test with specific, actionable errors — never "failed to open stream."

*Why:* The sharpest single installer gap. An installer should not need to know that a Hikvision RTSP path differs from a Dahua one. ONVIF discovery is a solved problem, and its absence forces manual work on the least-tolerant user.

*Acceptance:*
- [ ] ONVIF discovery finds cameras on the LAN
- [ ] Manual RTSP entry as fallback
- [ ] Connection test with actionable errors (wrong credentials / unreachable / unsupported codec — each distinct)
- [ ] Tested against at least two camera brands

---

### 🟩 EP-07 — Measurement Integrity · 18 pts · Days 4–16 (background)

> **Goal:** Your numbers cannot silently regress, and the two always-on critical detectors are no longer unmeasured.

**🔄 Run this in parallel throughout.** Eval runs are long-running background jobs — kick one off, work on another epic, harvest results. This epic mostly costs wall-clock, not attention.

**Why it matters.** Two things. First, **your headline metric is a function of a string literal in `gate.py` with no test guarding it.** Three prompt revisions moved precision 37.5% → 53.3% → 63.6% — a **26-point swing on wording alone**. Anyone can edit `_QUESTIONS`, run the app, see it work, and ship a regression invisibly.

Second, `weapons` and `violence` are `critical` priority in `baseline_critical_v1.json` — **always on, in every customer config, not disableable in pilots** — and both are **completely unmeasured**. The two highest-stakes detectors in the system, the ones that would summon an armed response, have never been evaluated. If `weapons` has poor precision you generate armed callouts for umbrellas; poor recall and it misses the thing it exists for.

**Definition of Done:** a prompt change that regresses precision fails CI, and you can state measured numbers for weapons and violence.

| ID | Task | Pts | Audit ref |
|---|---|---|---|
| EP-07-T1 | Prompt regression suite in CI | 5 | ML-02 |
| EP-07-T2 | Prompt + model versioning | 3 | ML-02 |
| EP-07-T3 | Measure `weapons` and `violence` | 8 | ML-03 |
| EP-07-T4 | Detector validation status in UI | 2 | ML-03 |

---

**EP-07-T1 · Prompt regression suite in CI · 5 pts**

*What:* A fast, cached subset of clips that runs automatically whenever prompt text in `gate.py` changes, failing CI if precision or recall moves beyond a defined tolerance.

*Why:* The eval harness already exists — this is about wiring it to the release process. It is the single highest-leverage ML task available.

*Acceptance:*
- [ ] Regression suite runs on any change to `gate.py` prompt constants
- [ ] Fails CI when precision or recall moves beyond tolerance
- [ ] Fast enough for CI (cached candidates; only the gate stage re-runs)
- [ ] Tolerance documented and justified
- [ ] Verified: deliberately weaken a prompt, confirm CI fails, revert

---

**EP-07-T2 · Prompt and model versioning · 3 pts**

*What:* Version prompts with an identifier stamped into every `VerificationResult` and stored with the event. Pin and record VLM model version and quantisation in eval metadata. Generate `SENSITIVITY_MEASURED` from archived metrics rather than hand-maintaining it.

*Why:* `gemma3:4b` is not frozen — a model update or quantisation change can shift behaviour with no code change on your side. And `SENSITIVITY_MEASURED` currently hard-codes numbers as constants, so code can drift out of sync with the reality it claims to describe. You already have the generate-from-archives pattern in `tools/make_numbers_sheet.py`; apply it here.

*Acceptance:*
- [ ] Prompt version stamped on every verdict and stored with the event
- [ ] Model version + quantisation recorded in eval metadata
- [ ] `SENSITIVITY_MEASURED` generated, not hand-written
- [ ] Any archived result traceable to the exact prompt that produced it
- [ ] Documented re-validation procedure for VLM upgrades

---

**EP-07-T3 · Measure `weapons` and `violence` · 8 pts**

*What:* Acquire proper datasets (RWF-2000 for violence; a weapons benchmark) and run the full two-stage evaluation. Publish with confidence intervals.

*Why:* These two are prioritised above the other five unmeasured detectors specifically because they are always-on and `critical` priority. Note the documented lesson: YouTube search returns *news coverage of* incidents rather than usable CCTV, at roughly a 1-in-5 hit rate. Do not repeat that — go to proper datasets.

*Acceptance:*
- [ ] RWF-2000 (or equivalent) acquired, licence reviewed
- [ ] Weapons benchmark acquired, licence reviewed
- [ ] Full two-stage eval for both detectors, ≥50 clips each
- [ ] Results with confidence intervals in `docs/NUMBERS.md`
- [ ] If either performs poorly: **say so, and consider removing it from the always-on baseline** — that is a legitimate and honest outcome of measuring

---

**EP-07-T4 · Detector validation status in the UI · 2 pts**

*What:* Show each detector's measured numbers **at the point of configuration**. Detectors without measurement are marked `EXPERIMENTAL` and excluded from marketing claims.

*Why:* The product currently markets ten capabilities and can evidence three. A user enabling an unvalidated detector should see that it is unvalidated — that is both honest and, given how the rest of this project treats measurement, on-brand.

*Acceptance:*
- [ ] Measured numbers shown next to each detector toggle
- [ ] Unmeasured detectors marked `EXPERIMENTAL`
- [ ] `docs/NUMBERS.md` covers every detector, including "not yet measured"
- [ ] Marketing claims restricted to measured detectors

---

### 🟩 EP-08 — Pilot Launch Kit · 18 pts · Days 15–18

> **Goal:** Everything needed to put Argus at Deluxe Paints and have it produce a case study rather than a support burden.

**Definition of Done:** the pilot can start Monday.

| ID | Task | Pts | Audit ref |
|---|---|---|---|
| EP-08-T1 | Config backup and one-click restore | 5 | ARCH-08 |
| EP-08-T2 | Weekly owner summary | 4 | USR-02, PD-06b |
| EP-08-T3 | Pilot runbook + `ARCHITECTURE.md` | 5 | INV-03b |
| EP-08-T4 | Pilot agreement + success criteria | 4 | INV-01 |

---

**EP-08-T1 · Config backup and one-click restore · 5 pts**

*What:* Automatic versioned backup of cameras, zones, rules, detector settings, and routing policy to a user-chosen location. One-click restore onto a fresh install. `events.db` integrity check at startup with recovery from last good backup.

*Why:* If the edge PC's disk fails, the customer loses all evidence, all configuration, all zones, all rules, all operator labels. Hardware redundancy is legitimately out of scope — but **configuration backup is cheap and its absence is not defensible.** Reconfiguring zones and rules for 20 cameras is hours of skilled work.

*Acceptance:*
- [ ] Automatic versioned config backup
- [ ] One-click restore onto a fresh install, **tested end to end**
- [ ] `events.db` integrity check at startup with automatic recovery
- [ ] Optional evidence backup to external drive or customer NAS — never your cloud by default
- [ ] Documented disaster-recovery procedure with a target recovery time

---

**EP-08-T2 · Weekly owner summary · 4 pts**

*What:* An automatic weekly summary by email or PDF, in business terms, requiring no action to receive. Month-over-month trends. Every figure traceable to underlying events.

*Why:* The person who signs the cheque and decides renewal currently has **no recurring contact with the product's value.** That is the profile of a product that gets cancelled at renewal — not because it failed, but because nobody could articulate what it did. This is largely a presentation layer over data you already produce, and it doubles as your case-study raw material.

*Acceptance:*
- [ ] Automatic weekly delivery, no action required
- [ ] Business terms: incidents, outcomes, estimated value, camera uptime
- [ ] Month-over-month trends
- [ ] Every figure traceable to events — no unfalsifiable numbers
- [ ] Extends EP-00-T4's value surface

---

**EP-08-T3 · Pilot runbook and `ARCHITECTURE.md` · 5 pts**

*What:* A runbook covering deploy, diagnose, recover. An `ARCHITECTURE.md` a competent engineer can read in an hour and then make a safe change. Verified dev-environment setup.

*Why:* Bus factor of one is the risk most likely to be raised in partner discussion after you leave the room. Your code comments are genuinely good at recording *why* rather than *what* — but there is no single document that orients someone new, and `docs/` still mixes current architecture with historical plans.

*Acceptance:*
- [ ] `ARCHITECTURE.md` covering the two-process split, data flow, and key decisions **with their rationale**
- [ ] Pilot runbook: deploy, diagnose, recover
- [ ] Dev setup verified by someone else following it from scratch
- [ ] Credentials, deployment access, and signing keys documented and recoverable

---

**EP-08-T4 · Pilot agreement and success criteria · 4 pts**

*What:* A written pilot agreement (using EP-02-T1's DPA and ToS) with defined success criteria, agreed with the customer **before** starting.

*Why:* `INV-01` is the highest-value item in the entire 389-point backlog — it resolves or sharpens about a third of everything else. But a pilot without agreed success criteria produces an ambiguous outcome and no case study, which wastes the single most valuable thing you can do this quarter.

*Acceptance:*
- [ ] Signed pilot agreement covering data use, retention, and liability
- [ ] Success criteria agreed **in writing, in advance** — uptime, alerts/day, confirmation rate, operator satisfaction
- [ ] 30-day duration with defined review points
- [ ] Customer agrees in advance to act as a reference if criteria are met
- [ ] Baseline captured **before** deployment (current false-alarm burden, current incident rate) — without this you cannot demonstrate improvement

---

## 7. Scope control

### 7.1 What is deliberately NOT in this plan

131 of 389 points are in scope. Here is what the other **258 points** are, and why each waits. Being explicit about this is what makes the plan honest rather than optimistic.

| Deferred | Pts | Why it waits |
|---|---|---|
| **ARCH-04** Multi-site aggregation | 21 | No chain customer in the pipeline yet. **But add the `site_id` field now** (~3 pts, fold into EP-03) — retrofitting the field after real data exists is expensive; building the feature speculatively is waste. |
| **ML-01** Trained models replacing heuristics | 21 | The heuristic + VLM architecture works and is measured. Measuring what exists (EP-07) beats improving what exists, per point, every time. |
| **ML-03** remainder (`fall`, `tamper`, `running`, standalone `theft`) | 13 | EP-07 does `weapons` + `violence` first — they're always-on and critical. The rest follow post-pilot. |
| **INV-06** Go-to-market | 13 | Blocked by EP-05 anyway. You cannot build an installer channel before installers can install it. |
| **INV-02** Defensibility narrative | 13 | Better written *after* the pilot, with real data behind it. |
| **PD-01** Full IA restructure | 8 | Real, but cosmetic next to legal and security exposure. EP-05-T3's wizard captures much of the benefit. |
| **SWE-04** Split `detector/core.py` | 8 | **Deliberately last.** It's the riskiest change in the backlog — a behaviour-preserving refactor of the hot path. Do it when you have a full 10-detector eval suite to verify against and a stable pilot to catch regressions. Doing it now trades a manageable problem for an unmanageable one. |
| **ML-05** Drift detection | 8 | Needs a running pilot to establish baselines. Genuinely cannot be built first. |
| **ML-06** Feedback loop validation | 8 | Needs real operator labels, which need a pilot. |
| **ML-04b** Grow eval sets to 100+ | 6 | EP-00-T3's confidence intervals make current numbers honest. Growing the sets is the follow-up. |
| **SWE-05** Tests for `training/`, `cli/` | 5 | Real gap (your only trained model comes from untested code) but nothing this quarter depends on it. |
| **PD-05** Ask persistence | 5 | Differentiated, but nobody has asked for it — nobody uses the product yet. Let the pilot tell you if Ask gets used. |
| **Everything else** | 129 | Capacity model, migrations, error analysis, provenance, typing, UI split, pricing, hardware tiers… |

**If you disagree with one of these, change it now, not on day 12.** Adding scope mid-sprint is how three-week plans become six-week plans.

### 7.2 What I would refuse to cut, at any velocity

| Item | Pts | Why |
|---|---|---|
| **EP-00-T1** CI tests | 1 | Best value-per-point in the entire plan |
| **EP-00-T3** Confidence intervals | 2 | Your credibility is the company's main asset. Protect it where it's currently overstated. |
| **EP-02-T1** Legal | 5 | The only item here that could end the company outright |
| **EP-03-T3** Audit log | 4 | Without chain of custody, the footage — the actual deliverable — is materially weakened |
| **EP-01-T4** Gate fail-visible | 1 | A live safety defect. One point. |

### 7.3 The cut list — decide once, now

**Pre-commit to this order so you make the call calmly rather than nightly under pressure.** If you're behind at the end of any sprint, cut from the top:

1. **EP-04-T3** signed updates (4) → manual updates during a single pilot are survivable
2. **EP-05-T4** ONVIF discovery (3) → you can configure the pilot's cameras yourself, once
3. **EP-06-T2** shift handover (4) → keep the incident record, drop handover to post-pilot
4. **EP-08-T2** weekly summary (4) → send the first month's report by hand
5. **EP-07-T3** weapons/violence measurement (8) → **if you cut this, remove both from the always-on baseline until measured.** Do not ship unmeasured always-on critical detectors to a real site.

**Cutting all five recovers 23 points — about 3 days.** That is your buffer. It is not large, which is the honest cost of a three-week plan.

**Never cut:** EP-02 (legal), EP-03-T1/T3 (auth, audit), EP-01 (observability). Those four are what make a pilot legal and supportable — the entire point of the exercise.

### 7.4 Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Legal/insurance lead time exceeds 3 weeks | **Blocks the pilot entirely** | **Start EP-02-T1 on day 1.** This is the single most likely schedule-breaker and the only one you cannot solve by working harder. |
| EP-05 bundling fights PyInstaller + Ollama | Slips 2–3 days | Spike it on day 4 (30 min) to find out early. Fallback: guided first-run download instead of full bundling. |
| Dataset acquisition for EP-07-T3 stalls | Weapons/violence stay unmeasured | Pre-agreed: pull them from the always-on baseline rather than ship unmeasured critical detectors |
| EP-06-T3 mobile ships before EP-03-T1 auth | **Security incident** — live feeds on an open port | Hard dependency. Explicit acceptance test: verify no unauthenticated route exists. |
| Velocity is 2.4× standard with zero slack | Everything slips | §7.3 cut list, pre-decided |
| Pilot customer isn't ready by 11 Sep | Plan completes, milestone doesn't | Start the EP-08-T4 conversation with Deluxe Paints **this week**, not in week 3 |

---

## 8. Daily checklist

Print this. Tick it.

```
SPRINT 0 — DEMO (Days 1–3, 11 pts)
  D1  □ EP-02-T1  Email lawyer + insurance broker  ← DO THIS FIRST, EVERY DAY IT SLIPS COSTS A WEEK
      □ EP-00-T1  CI test job + needs:test + branch protection
      □ EP-00-T6  Release checksums
  D2  □ EP-00-T2  Mock-gate guard + gate error counter
      □ EP-00-T3  Confidence intervals in NUMBERS.md
  D3  □ EP-00-T4  Business framing for suppression screen
      □ EP-00-T5  Docs → archive/, README accurate
      □ Full demo dry-run, clean machine, twice
  SAT □ 🎤 INVESTOR DEMO

SPRINT 1 — OBSERVABLE & LEGAL (Days 4–8, 35 pts)
  D4  □ EP-01-T1  logging_setup.py + rotation + diagnostics zip
      □ EP-05 spike: can PyInstaller bundle Ollama? (30 min, de-risks S3)
      □ EP-07-T1  Kick off prompt-regression baseline run (background)
  D5  □ EP-01-T2  Convert 260 print() calls
      □ EP-01-T3  64 handlers log + error counters
  D6  □ EP-01-T4  Gate fail-visible
      □ EP-01-T5  Camera connection state machine
  D7  □ EP-02-T2  Retention + purge + legal hold
      □ EP-03-T1  Local accounts + auth (app AND frame publisher)
  D8  □ EP-03-T2  Three roles, enforced server-side
      □ EP-07-T1  Wire regression suite into CI
  ✅ GATE: system is observable and legally deployable

SPRINT 2 — SECURE & OPERATOR-READY (Days 9–13, 47 pts)
  D9  □ EP-03-T3  Append-only audit log
      □ EP-03-T4  Encryption at rest + SECURITY.md
      □ (fold in ARCH-04's site_id field, ~3 pts — cheap now, expensive later)
  D10 □ EP-04-T1  /health endpoint
      □ EP-04-T2  Opt-in heartbeat + dashboard
  D11 □ EP-04-T3  Signed updates      [cut candidate #1]
      □ EP-04-T4  Liveness + daily self-test
  D12 □ EP-06-T1  Alert states + ownership
      □ EP-06-T2  Incident record + handover   [handover = cut candidate #3]
  D13 □ EP-06-T3  Mobile response view  ⚠️ VERIFY AUTH ON EVERY ROUTE
      □ EP-06-T4  Two-tier alerting for critical
      □ EP-07-T2  Prompt/model versioning
      □ EP-07-T4  Detector status in UI
  ✅ GATE: secure, monitored, and a guard can actually work it

SPRINT 3 — PILOT-DEPLOYABLE (Days 14–18, 38 pts)
  D14 □ EP-05-T1  Bundle everything into one installer
      □ EP-07-T3  Start weapons/violence eval (background)  [cut candidate #5]
  D15 □ EP-05-T2  Signing + notarisation
      □ EP-05-T3  First-run wizard + self-test
  D16 □ EP-05-T4  ONVIF discovery   [cut candidate #2]
      □ EP-05  TEST ON A CLEAN MACHINE — no Python, no Ollama
      □ EP-07-T3  Harvest results → NUMBERS.md
  D17 □ EP-08-T1  Config backup + restore (test the restore!)
      □ EP-08-T2  Weekly owner summary   [cut candidate #4]
  D18 □ EP-08-T3  ARCHITECTURE.md + pilot runbook
      □ EP-08-T4  Pilot agreement + baseline capture
      □ Full clean-machine install + 24h soak
  ✅ GATE: 🚀 PILOT-DEPLOYABLE
```

---

## 9. Definition of Done — the whole plan

Tick every box and Argus is deployable at a real customer site:

- [ ] **Legal** — privacy policy, DPIA, ToS with liability limits, DPA, insurance obtained
- [ ] **Secure** — auth on every surface, three roles enforced, append-only audit log, disk encryption
- [ ] **Observable** — structured logs, per-component error counters, no silent failure class
- [ ] **Monitored** — `/health`, opt-in heartbeat, you're alerted before the customer calls
- [ ] **Bounded** — 30-day retention with purge, legal hold, disk warnings
- [ ] **Installable** — one signed installer, clean-machine tested, non-technical person in <30 min
- [ ] **Operable** — states + ownership, mobile response, incident records, instant critical alerts
- [ ] **Defensible** — prompt regression in CI, versioned prompts, CIs on every number, weapons/violence measured or removed from baseline
- [ ] **Recoverable** — config backup with a *tested* restore, DB integrity checks
- [ ] **Documented** — `ARCHITECTURE.md`, runbook, `SECURITY.md`, current README

**Then start the pilot.** `INV-01` is worth more than every remaining point in the backlog.

---

*Traceable to [`docs/AUDIT.md`](AUDIT.md) — 51 concerns, 389 points. This plan delivers 131 of them, selected to reach one milestone: a pilot that is legal, supportable, installable, and defensible.*
