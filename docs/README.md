# Documentation index

What each document is for, and which ones are current. If a doc isn't listed
here it's in [`archive/`](archive/) — superseded, kept for history, not to be
followed.

## Start here

| Doc | What it's for |
|---|---|
| [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md) | How the system actually works, end to end. The one to read first. |
| [architecture.md](architecture.md) | The three-layer design and why it's shaped that way. |
| [USER_GUIDE.md](USER_GUIDE.md) | Operating the desktop app, written for the person using it, not building it. |
| [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) | Exact commands to show the system working, and what each part proves. |

## What we can defend

| Doc | What it's for |
|---|---|
| [NUMBERS.md](NUMBERS.md) | Every measured figure, with its sample size and 95% interval. **Generated** — run `python tools/make_numbers_sheet.py`, never hand-edit. |
| [GATE_MODEL_BAKEOFF.md](GATE_MODEL_BAKEOFF.md) | How the on-device gate model was chosen, and what it scored. |
| [prompt_baseline.json](prompt_baseline.json) | The gate prompts' measured precision/recall, and the fingerprint CI checks them against. **Generated** — `tools/prompt_regression.py run --update-baseline`. |
| [AUDIT.md](AUDIT.md) | Independent audit of the whole system. The backlog everything else traces to. |
| [BACKLOG.md](BACKLOG.md) | Work that is started, blocked or deferred — what state it is in and what would unblock it. |
| [UI_SPEC.md](UI_SPEC.md) | **The decided UI direction.** Every UI-touching task implements this; deviations update this file first. |
| [DESIGN_BRIEF.md](DESIGN_BRIEF.md) | The interface problem stated for a designer: the three jobs, the missing triage workflow, and what is built but undesigned. |
| [SPRINT_PLAN.md](SPRINT_PLAN.md) | The current three-week delivery plan. Every task cites an audit ID. |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | The system in one hour: two-process split, data flow, every key decision with its rationale. |
| [RUNBOOK.md](RUNBOOK.md) | **Pilot operations**: deploy, diagnose, recover — with the recovery-time target and the key inventory. |
| [PILOT_AGREEMENT.md](PILOT_AGREEMENT.md) | The pilot template: data terms, success criteria agreed in advance, reference clause. |
| [PILOT_BASELINE.md](PILOT_BASELINE.md) | The before-numbers worksheet — filled in the week before install. |

## Running and extending it

| Doc | What it's for |
|---|---|
| [../SECURITY.md](../SECURITY.md) | Security model, threat model, and the procurement answer sheet. What is defended and what is not. |
| [HEARTBEAT.md](HEARTBEAT.md) | Exactly what an opted-in site transmits, field by field, and how to run the receiver. Public on purpose. |
| [DATA_RETENTION.md](DATA_RETENTION.md) | What is stored, for how long, what outlives its expiry and why. Customer-facing — written to be handed to a DPO. |
| [OFFLINE_VLM.md](OFFLINE_VLM.md) | Running verification fully offline against a local model. |
| [SOFTWARE_WIRING.md](SOFTWARE_WIRING.md) | Which pipeline is which, and what calls what. |
| [TRAINING.md](TRAINING.md) | Fine-tuning the video-action model. |
| [TRAIN_WEAPON_MODEL.md](TRAIN_WEAPON_MODEL.md) | Training custom weapon weights. |

## History

| Doc | What it's for |
|---|---|
| [../plan.md](../plan.md) | The backend V1 plan. Module docstrings cite its phase numbers (`plan.md Phase 8.3`), so it stays at the repo root. |
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Running chronological log of decisions and why they were made. Append-only; earlier entries describe the state at the time, not now. |
| [handoffs/](handoffs/) | Point-in-time session handoffs. Snapshots, not living documents — read the newest to get caught up. |
| [archive/](archive/) | Superseded plans and one-off briefs. Kept so decisions can be traced; **do not follow them.** |
