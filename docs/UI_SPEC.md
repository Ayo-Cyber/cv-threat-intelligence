# Argus — UI specification

**Status: decided.** This is the interface the system follows. Every sprint task
that touches the UI implements *this*, and a change of direction is a change to
*this file first* — not an ad-hoc decision inside a feature branch.

**Decided:** 19 Aug 2026 · **Problem statement:** [DESIGN_BRIEF.md](DESIGN_BRIEF.md)
· **Visual mockups:** [Claude Design canvas](https://claude.ai/design/p/dc5892e5-7a15-4d79-bdbd-5b78136bc39a?file=Argus+Console.dc.html)
(options 1a–1e; this spec records which won and why)

---

## 0. The decision in one paragraph

The shell is the **three-surface architecture** (option 1b): Watch / Triage /
Configure, with configuration demoted behind Settings. Inside Triage, the
default view becomes the **"Now" screen** (option 1a) — one alert at a time,
with ownership — *as soon as EP-06 gives the backend an ownership model*; until
then Triage holds the current list. The **mobile response view** (1c), the
**first-run flow** (1d) and the **state language** (1e) are adopted as designed.
1a and 1b were never alternatives: 1b is the container, 1a is what lives in its
Triage tab.

## 1. Information architecture

### 1.1 Navigation

Nine surfaces collapse to three, plus a settings surface and a footer that is
always present:

| New surface | Absorbs | Who lands here |
|---|---|---|
| **Watch** | Live + Map (one question, two zoom levels: Grid / Floor plan toggle) + camera link state | Installer (default) |
| **Triage** | Alerts (+ the Now screen once EP-06 lands) + shift handover | Operator, Owner (default) |
| **Configure** | Cameras, Rules, zones, detector toggles | Installer |
| **Settings** (behind a gear, not primary nav) | System, Learning, notifications, retention, users, audit trail, diagnostics | Owner |
| **Value** | Value screen + (later) the weekly owner summary | Owner section of nav |

- **Ask** is removed from primary navigation. The audit calls it a demo
  feature; it survives as a secondary affordance inside Watch, not a top-level job.
- **Nav footer, always visible:** liveness line (`● 6 cameras live · last frame
  0.4s ago`), then the signed-in identity (`ayo · owner`, mono). This footer is
  the "blinking light" — it never disappears, on any surface.
- **Role-aware nav** (backend enforcement already exists): operators do not see
  Configure or Settings; installers do not see Triage history or Value. The nav
  renders from `authState.permissions`, never from role names hardcoded in JS.

### 1.2 Landing per role

`landing_for(role)` already exists in the backend: owner → Triage, operator →
Triage, installer → Watch. The UI honours it after sign-in.

## 2. Screens

### 2.1 Sign-in and first run (option 1d) — **BLOCKING, build first**

PR #8 merged: every consequential backend method now requires a session, and no
screen can create one. Until this ships, the app cannot be used at all.

- **First run** (`authState.configured == false`): full-screen 5-step flow.
  Step 1 creates the owner account — copy states plainly: *"Nothing ships with
  a password, so there is nothing to change — you are creating the first
  account now."* Progress dots for steps: account → cameras → zones → what to
  watch for → test alert. Steps 2–5 reuse the existing wizard panels.
- **Sign-in** (`configured == true, signed_in == false`): centered panel, name +
  password, error line on refusal (the backend already returns one message for
  both wrong-password and unknown-user — display it verbatim, never elaborate).
  Lockout message shown as returned.
- **Password fields**: helper line *"At least 8 characters. Stored hashed — it
  cannot be recovered, only reset."*
- After sign-in: route to `authState.landing`.

### 2.2 Watch (from option 1b)

- Grid of camera tiles; each tile carries the **link-state row** beneath it:
  status dot + camera id (mono) + right-aligned freshness (`0.3s ago`) or state
  (`FIRE` in `--sev-crit`, `OFFLINE 6m` in `--sev-crit`, `reconnecting` in
  `--sev-high`).
- An offline camera's tile: dashed `--sev-crit` border, dimmed (~55% opacity),
  `no signal` placeholder. **Never an empty black tile** — black is
  indistinguishable from a working camera watching a dark room.
- Any camera offline past grace ⇒ a full-width banner panel under the grid:
  *"backroom has been unreachable for 6 minutes. Reconnecting — 5 attempts.
  Nothing is being watched there."* + Diagnose button.
- Top bar right: engine truth line (*"All detectors running · last check 2s
  ago"*), from `gate_health.json` freshness — reads *"Engine not running"* when
  stale, never a silent nothing.
- Grid / Floor plan toggle swaps tile layout for the map with the same tiles
  pinned to positions. Same data, two zoom levels.

### 2.3 Triage

**Phase 1 (now):** the existing alert list + detail panel moves here unchanged,
plus the 1e state treatments (below) and the shift strip: a header line
*"Since your shift began: 28 shown · 201 filtered out"* — the Value claim,
surfaced where the operator lives.

**Phase 2 (after EP-06 adds ownership):** the **Now screen** becomes the
default tab:

- One alert, full width: severity pill + rule title + camera/time (mono),
  evidence viewer with frame strip (thumbnails, active frame outlined in
  `--c-accent`), "Why TrueSight confirmed this" panel with reason + confidence
  bar.
- Right rail: **Your call** panel — primary button **"I'm on it"** (claims the
  alert; sub-copy: *"Claims this alert. Everyone else sees your name against
  it."*), then Real / False alarm pair. **Then** panel — the next 2 waiting
  alerts, one line each. Below: *"Sam took 'Weapon · till' 3 min ago"* — other
  people's claims are always visible.
- Queue depth is a number in the top bar (*"2 more waiting · 1 held by Sam"*),
  never a wall of rows.
- **Empty state is a feature, not a blank:** the all-systems-normal panel
  (§2.6) fills the space — a quiet night must look *watched*, not dead.
- Alert states: `NEW → ACKNOWLEDGED(by, at) → RESOLVED(outcome, note)`.
  Handover view: last N hours — what fired, what was resolved by whom, what is
  still open.

### 2.4 Mobile response view (option 1c)

Served by the engine over the local network (frame publisher already
authenticates every route). Telegram alerts deep-link here.

- Single column, 390px design width. Header: status dot + "Argus" +
  `on site network` (mono).
- Severity pill + time/camera, rule title at 22px, swipeable evidence frames,
  reason paragraph.
- **"I'm on it"** full-width at 15px text / ~48px height; Real / False alarm
  below at ≥44px hit targets; a note field for the handover.
- Footer: queue context (*"2 more waiting · Sam has 'Weapon · till'"*).
- Requires sign-in (same accounts); sessions per §1 backend.

### 2.5 Value / owner surface

- The existing Value screen moves under the Owner nav section unchanged.
- The **weekly summary** (EP-08 / USR-02) is generated from the same
  `value_summary()` data: incidents, outcomes, estimated value with the site's
  own rates, camera uptime, month-over-month. Email/PDF, no action required to
  receive. Every figure traceable; money hidden until the site enters a rate.

### 2.6 The state language (option 1e) — applies everywhere

These are system-wide rules, not one screen:

| State | Treatment | Hard rule |
|---|---|---|
| **UNVERIFIED alert** | Dashed border panel; dashed grey pill (`--c-mut`, transparent fill); copy: *"TrueSight could not reach a verdict — [cause]. This has not been checked by anything. Review it yourself."* | **Never a severity colour. No confidence bar — nothing scored it.** Must be visually impossible to confuse with a confirmed alert. |
| **Camera OFFLINE** | `--sev-crit` dot; `OFFLINE <duration>`; banner names the consequence: *"Nothing is being watched there."* | Never inferred from silence; reads `camera_links` state. `unknown` (engine not running) shows as *"not monitoring"*, never as healthy. |
| **Component degraded** | `--sev-high` dot; counts inline: *"failing 1 in 4 frames · 312 ok, 104 errors · last: [error]"* | Numbers, not adjectives. |
| **All systems normal** | `--c-live` dot; *"All systems normal — 6 cameras · last frame 0.4s ago · disk encrypted · 41 days retained"* | The system says so explicitly, on a schedule — silence is never the success signal. |
| **No data yet** | Plain statement of what is missing and why (existing Value-screen pattern) | Zeros are never shown where no measurement exists. |
| **Mock gate** | Existing full-width `--sev-crit` banner, unchanged | Reserved: full-width banners mean "you may not navigate away from this". |

### 2.7 Settings additions (already built backend, no UI)

Inside Settings, three new panels using existing patterns (panel + plabel +
rows): **Users** (list, add, role change — owner only), **Audit trail** (table:
time, actor, action, target; verify-chain status line; export button — owner
only), **Security** (disk-encryption status with the exact
`requirement_message()` text).

## 3. Visual language

Tokens, type, components and copy voice are specified in
[DESIGN_BRIEF.md](DESIGN_BRIEF.md) §Current-visual-language and are **unchanged**
— this redesign is a reorganisation, not a restyle. Binding rules:

- Severity tokens (`--sev-*`) are load-bearing and never repurposed. New
  states argue for a new token; they do not borrow one.
- System font stack only (offline app). Mono for ids, timestamps, hashes.
- 13px base; dense console, not a marketing page. Pills 10.5px; panel labels
  (`.plabel`) 9.5px uppercase.
- Both themes always. Any new colour is defined in both.
- Copy voice: plain, specific, willing to say what the system does not know.
  No exclamation marks, no invented numbers, every measured figure carries its
  n. (*"A number we invented for you is worth nothing to you."*)

## 4. Mapping to the sprint plan

| Sprint task | Implements | Spec section |
|---|---|---|
| **Login/first-run UI** (unblocks merged PR #8 — *do first*) | 1d | §2.2 |
| Nav reorganisation (PD-01, 8 pts — schedule alongside) | 1b shell | §1 |
| EP-04 — Operability & Remote Health | Watch liveness, banners | §2.2 nav footer, §2.6 |
| EP-06 — Alert Triage & Response (21 pts, needs ownership schema) | 1a Now screen, states, handover | §2.3 phase 2 |
| EP-06 mobile view | 1c | §2.4 |
| EP-08 — Pilot Launch Kit | Weekly owner summary | §2.5 |
| Settings: users / audit / security panels (EP-03 follow-through) | — | §2.7 |

## 5. Change control

Mockups live in the Claude Design project; this file records what was decided.
If implementation forces a deviation, update this file in the same PR and say
why — the spec must never drift behind the shipped UI.

### Recorded changes

- **2026-08-20 (EP-09, Sprint UI).** The shell itself now implements §1: nav is
  Watch / Triage / Configure + an Owner section (Value) + Settings behind a
  gear; legacy route names remap to surface tabs; landing is Triage for
  owner/operator, Watch for installer. §2.2: link-state rows under tiles,
  dashed dimmed NO SIGNAL offline tiles, the offline consequence banner, and
  the engine truth line. §2.3 phase 2: the Now screen is Triage's default tab
  (the list survives as "All alerts"); a claimed alert stays with its claimant
  until resolved. §2.7 panels and the §2.6 sweep (UNVERIFIED dashed-grey in
  list, detail and Now; no confidence bar where nothing scored). Two wording
  deviations, both because the spec's copy would have overstated what we
  measure: the footer liveness line reads "N of M cameras live" (per-frame
  freshness is not published outside the engine), and the shift strip reads
  "Last 24 h: N shown · M filtered out" (the backend aggregates by day, and
  claiming "since your shift began" would lie about the window).

- **2026-08-20 (EP-05-T4).** Cameras step gains **Find cameras (ONVIF)** above
  the subnet scan — one tap lists cameras that announce themselves; tapping a
  result fills the address field (manual entry and vendor paths remain the
  fallback). The **Test** button's failures are now named and actionable —
  wrong credentials / unreachable (host:port) / wrong stream path / unsupported
  codec — each with its own fix line; "failed to open stream" no longer exists.
- **2026-08-20 (EP-05-T3).** First-run wizard extended from four steps to the
  epic's seven-step guided flow: credentials (the auth gate, before the wizard)
  → add camera → verify feed (Test button) → **draw zones** → **use-case
  template** (Retail / Warehouse-HSE / Office cards; picking one preselects
  detectors and rules) → **confirm detectors** (grouped toggles, applied to
  every camera) → verification → **send a test alert**. The finish screen now
  runs a **self-test** listing every component ✓/✕/⚠ with a plain-English fix
  line per failure. Extends §2.2's first-run scope; state colours follow §2.6.
- **2026-08-20 (EP-05-T1).** First-run "Local AI verification" step: when the
  Ollama runtime ships inside the bundle (the installed product), the offline
  state no longer sends the user to ollama.com — it shows **Start verifier**,
  which brings up the bundled runtime and pulls the model with progress
  (resume is native to the runtime). The install-Ollama copy remains only for
  dev/source runs without a bundled runtime. Why: the epic's zero-terminal
  acceptance — telling an installer to go install a dependency is the exact
  wall EP-05 removes.
