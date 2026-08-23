# Argus — Architecture

*For the competent engineer who has an hour and then needs to make a safe
change. Rationale is stated with every decision, because the "why" is what
keeps changes safe. History and phase numbers live in [plan.md](plan.md) and
[docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md); this file is the present.*

## The one-paragraph version

Argus is an on-premises CCTV threat-detection product. Cheap, always-on
detectors (YOLO + heuristics + a fine-tuned VideoMAE) watch every frame and
propose *candidates*; a local vision-language model (**TrueSight**, gemma3:4b
via Ollama) *verifies* each candidate before a human is disturbed. Everything
runs on one edge machine; nothing leaves the building unless the site opts in
(heartbeat) or points a notifier outward (Telegram etc.).

## The two-process split

```
┌─ Argus (GUI console) ────────────┐      ┌─ argus-engine ───────────────────────┐
│ PyQt6/QtWebEngine shell          │      │ python -m cvti.serving.pipeline      │
│ one index.html (no build step)   │      │                                      │
│ ConsoleBackend  ── spawns ─────────────▶│ decode → YOLO batch → per-camera     │
│  · site/zones/rules editing      │      │ state (track/zones/detectors/rules)  │
│  · triage, users, audit, value   │      │ → AlertQueue → GatePool → TrueSight  │
│  · reads events.db + health JSON │◀──── │ → AlertSink (events.db, evidence,    │
└──────────────────────────────────┘ files│   notifier, mobile deep-links)       │
                                          │ + FramePublisher, MobileServer,      │
                                          │   Heartbeat, Retention, Assurance    │
                                          └──────────────────────────────────────┘
```

**Why two processes.** The engine imports torch/ultralytics/transformers and
must survive GUI crashes (and vice versa); the console must stay responsive
while models load for 30s. They share nothing but **files in one output
directory** (`events.db`, `gate_health.json`, `frames.json`) — chosen over a
socket API deliberately: files are inspectable after the fact, survive either
process dying, and make "what did the engine think at 3am" answerable.

**Contract points** (all under the site's output dir, default `runs/site/`):
- `events.db` — SQLite; the engine writes alerts, the console writes triage
  state (claim/resolve), both via short transactions. One writer per column
  family; SQLite's locking is sufficient at this scale (≤ tens of writes/min).
- `gate_health.json` — the engine's `/health` document, rewritten every few
  seconds. **Staleness is the liveness signal**: >30s old = engine not
  running. Never inferred from process tables.
- `frames.json` — `{port, token}` of the engine's frame publisher. The token
  travels with the port; every frame route authenticates (see Security).

## Data flow, end to end

1. **Decode** (`serving/streams.py`) — one thread per camera, RTSP/file/webcam,
   with reconnect + explicit link state (`connected/reconnecting/offline`).
   Offline is *declared*, never inferred from silence.
2. **Detect** (`serving/pipeline.py` + `serving/camera.py`) — batched YOLO at
   `--target-fps`; per-camera `PerCameraState` runs tracking, zones, and the
   enabled detectors (concealment, violence, weapons, fire, fall, running,
   crowd, tamper, VideoMAE video-action).
3. **Rules** (`rules/customization.py`) — customer config + compound recipes +
   the per-camera plain-English rule (`gate_question`), merged over
   `baseline_critical_v1.json`, which is **always on and not disableable** —
   the baseline is a safety net, not a second opinion.
4. **Queue** (`serving/alert_queue.py`) — heap; criticals drain first.
   Critical candidates also take the **fast path**: a provisional alert row +
   notification in <1s, settled in place by the verdict (confirmed, or kept
   and marked RETRACTED — "why did my phone buzz?" always has an answer).
5. **Verify** (`verification/gate.py` + `serving/gate_pool.py`) — TrueSight
   answers the rule's question about the evidence frames. **Fail-visible**: a
   transport/parse failure is an UNVERIFIED alert shown to a human, never a
   silent drop, and never dressed as a severity it didn't earn.
6. **Sink** (`serving/alert_sink.py`) — persist + evidence frames + notify
   (site notifier, or routing rules for matches only) + mobile deep-link.
7. **Respond** — console Triage (Now screen: one alert, one action) and the
   phone view (`serving/mobile.py`), both through the same state machine
   (`triage.py`: NEW → ACKNOWLEDGED(owner) → RESOLVED(outcome)) and audit log.

## Key decisions and why

| Decision | Rationale |
|---|---|
| **Local VLM (Ollama/gemma3:4b), not a cloud API** | The product's core promises are privacy (footage never leaves) and offline operation. ~12s/verdict is the accepted cost; the two-tier fast path exists because that latency is wrong for weapons/fire. |
| **Fail-visible verification** | `confirmed=False` (a verdict) and `error` (absence of a verdict) are different fields. A connection failure once looked identical to "TrueSight examined a fire and said safe". Everything downstream preserves the distinction — UNVERIFIED never wears a severity colour. |
| **One `index.html`, no frontend build** | The bundle must build with PyInstaller on three OSes; every dependency is a build risk. Inline JS with `node --check` in CI is the trade. |
| **Prompt versioning + regression trip-wire** | Three prompt rewords moved precision 26 points. Every verdict carries a prompt fingerprint; CI fails when wording changes without a re-measurement (`tools/prompt_regression.py`). `SENSITIVITY_MEASURED` is generated from archived runs, never hand-edited. |
| **Measured vs EXPERIMENTAL, in the UI** | Detector toggles show their evidence (`✓ 88.9% caught (n=9)`) or a dashed EXPERIMENTAL badge, held equal to `docs/NUMBERS.md` by test. The interface may not outrun the evidence. |
| **Tokens on every frame route** | Frame servers bind localhost but the mobile server doesn't; one uniform rule ("no unauthenticated route to a frame") survives topology changes. The token rides in `frames.json` beside the port. |
| **Append-only audit with a hash chain** | Operator actions are evidence. SQLite triggers forbid UPDATE/DELETE; `verify()` says "intact" or "treat as tampered". |
| **Site file is the single config surface** | Cameras, detector flags, English rules, notify, retention, value rates, heartbeat — one JSON the engine watches (mtime): notify/retention/heartbeat apply **live**; camera/zone/model changes require Start monitoring (they rebuild models, which *is* a restart). |
| **Retention purges settled evidence only** | GDPR storage-limitation by default (30d), but legal holds and open incidents are never purged — deleting the evidence of an unresolved incident is worse than keeping it. |
| **Signed updates (Ed25519) separate from OS signing** | OS signing proves the installer's origin to the OS; the update key proves updates' origin to Argus. Fail-closed without the `cryptography` lib; rollback works offline including to as-shipped. |

## Where things live

| Path | What |
|---|---|
| `cvti/serving/` | engine: pipeline, streams, camera state, queue, gate pool, sink, mobile, heartbeat, retention, frame publisher, onboarding |
| `cvti/verification/` | TrueSight gate, prompts, Ollama helpers |
| `cvti/rules/` | customization engine (customer rules, recipes, English rules) |
| `cvti/app/` | console: Qt shell, bridge (QWebChannel slots), ConsoleBackend, web/index.html |
| `cvti/security/` | accounts (scrypt/PBKDF2), permissions, audit chain, disk encryption checks |
| `cvti/eval/` + `tools/` | datasets, golden sets, prompt regression, critical-detector measurement, numbers-sheet generators |
| `packaging/` | `argus.spec` (one bundle, two executables), build.py, entitlements |
| `configs/` | site files, presets, baseline, zones/rules per camera (generated), routing |
| `docs/` | NUMBERS (claims sheet), SPRINT_PLAN/AUDIT (history), RUNBOOK, SECURITY, SIGNING, UI_SPEC |

## Making a safe change

1. `python -m pytest -q` first — 530+ tests, ~60s, green before you start.
2. UI: edit `index.html`, keep `node --check` clean; async callbacks that
   paint `$("screen")` must capture and re-check the render token.
3. Engine contracts: anything the console reads (`events.db` columns,
   `gate_health.json` keys) is migrated additively — the sink's `ALTER TABLE`
   loop and the health doc's legacy keys show the pattern.
4. Prompts: any wording change fails CI until re-measured — that is the
   feature. `python tools/prompt_regression.py run` against a live Ollama.
5. New claims (a measured number, a coverage statement) go to
   `docs/NUMBERS.md` **and** the code table beside it; tests enforce equality.
