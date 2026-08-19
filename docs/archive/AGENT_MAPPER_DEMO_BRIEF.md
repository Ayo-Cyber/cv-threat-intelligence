# Agent Mapper — Demo Brief

A talking-points document for the live demo. Covers the architectural shift, what is built, the design correction we just made, the chosen model, the hosting path, and the next layer.

---

## One-paragraph framing

We are moving the product from a detector with hardcoded alerts to a context-aware threat intelligence platform. The new architecture has four layers — Agent Mapper, Detection Core, Customization Engine, Verification Gate. The Agent Mapper is the first of those new layers and it is now built. It uses a vision-language model to understand the environment each camera is watching, so downstream layers know whether they are looking at a retail shop, an estate gate, or a parking lot. That context is what allows the rest of the system to interpret detections correctly instead of firing on everything.

---

## Why this matters — the structural false-positive fix

The hardest problem in CCTV intelligence is not detection. It is false positives. A generic AI tries to interpret "is this dangerous?" against every possible threat in the world — that is why it over-fires. Our approach is the opposite: the user explicitly defines what counts as a threat in their specific business context, and the system only fires on those declared rules. The Customization Engine reads the user's `user_config.json`, and that file is the source of truth. The AI assists; the user decides. This is a structural fix, not a tuning exercise.

The Agent Mapper feeds into this by giving the user a smart starting point. It identifies the environment so we can suggest the right rule preset out of the box — Estate Guard for an estate gate, Retail Watch for a shop floor, Office Sentinel for an office lobby. The user then accepts or edits that suggestion, and the result becomes their authoritative config.

---

## What is built

### The four-layer architecture is documented and locked
- `architecture.md` — the technical north star, with data contracts for every inter-layer interface
- `docs/AGENT_MAPPER_PLAN.md` — the build plan for this specific layer
- `PROJECT_CONTEXT.md` — running checkpoint log of every decision and shift

### Agent Mapper v1 — built and verified
- `agent_mapper.py` — full implementation, ~750 lines
- `schemas/scene_context.schema.json` — locked JSON contract for the output, with bounded enums for environment types and zone roles
- `prompts/agent_mapper_prompt.txt` — bounded-vocabulary, JSON-only VLM prompt
- Supports webcam, RTSP, video file, and image file inputs
- Three provider backends: `mock` for offline tests, `anthropic` for Claude Vision, `openai_compatible` for any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, etc.)
- Smart frame sampling — picks the best representative frame from a video clip using brightness and blur heuristics so the VLM gets a usable image
- Defensive output parsing — recovers from VLMs that wrap JSON in prose, snaps invalid enums back to known values, never crashes on a malformed response
- Outputs `runs/context/<camera_id>/scene_context.json` for downstream consumption

### Test harness — built and verified
- `tests/agent_mapper/eval.py` — multi-model evaluation runner
- Imports `agent_mapper.py` directly so it tests the actual production code path
- Measures: valid-JSON rate, environment-type accuracy, latency, and a hard threat-vocabulary leak check (more on that below)
- Outputs per-clip CSV plus a per-model summary
- End-to-end smoke-tested

---

## The design correction — descriptive-only Mapper

A real architectural improvement landed today after feedback from Ayo, the co-founder.

The original Agent Mapper schema included two fields that were structurally wrong: `risk_hints` (a list of suggested threats) and `suggested_preset` (a recommended rule pack). The problem with putting these in the Mapper:

1. **Data quality** — a VLM asked "describe what you see" is reliable. A VLM asked "what threats could happen here" is speculating. That is the wrong place for the noisiest call in the pipeline.
2. **Layer coupling** — `risk_hints` mirrored the GTM-12 rule library, meaning every change to the rule set would force a change to the Mapper schema, prompt, and code.
3. **It pre-empted the user's authority** — even labelled as "suggestions," these fields would bias the UI and shape what the user picks. That violates the source-of-truth principle.

We stripped both fields. The Agent Mapper is now deliberately descriptive-only. It tells you what the scene is; it does not tell you what to worry about. Threat policy lives in a separate deterministic component (Preset Recommender) that maps `environment_type` to a preset and a default rule pack, which the user then accepts or edits.

The test harness includes a hard threat-vocabulary leak check — it scans the model's output for words like "threat", "danger", "loiter", "suspicious", "intrud", "weapon", etc. Any leak is a structural contract violation, not a soft signal.

---

## Model selection — Gemma 4 26B-A4B-it

We evaluated the current open-source VLM frontier and selected **Google Gemma 4 26B-A4B-it**.

- Mixture-of-Experts architecture: 26B total parameters, only ~3.8B active per token. Runs at near-4B-dense inference speed while leveraging the full 26B knowledge.
- Native multimodal (text + image + video input)
- 256K context window
- 140+ languages
- Apache 2.0 license — unrestricted commercial use
- Released April 2026

Empirically validated against alternative candidates (Qwen 3-VL, Qwen 3.5-35B-A3B, GLM-5.1) on Hugging Face image tests. Gemma 4 26B-A4B was the best fit for the descriptive scene-understanding task at this parameter scale.

---

## Hosting path — OpenRouter

We are running Gemma 4 26B-A4B through **OpenRouter**, which hosts the model with both free and paid tiers.

**Why this works for us:**
- OpenAI-compatible API — our existing `openai_compatible` provider works with zero code changes
- Free tier covers the dev and eval phase entirely
- Paid tier is $0.06 / million input tokens, $0.33 / million output tokens
- ~$0.0003 per Mapper call — $0.30 per 1,000 calls
- No infrastructure to manage, no GPU to provision
- 11 backend providers, high uptime

**Cost projection:** at 1 call per camera every 5 minutes across 100 cameras, full-day cost is roughly $9. For our dev and pilot phase the cost is effectively zero.

**The longer-term hosting path** when we outgrow OpenRouter: serverless GPU (Modal, Replicate) for pay-per-call economics at single-deployment scale, or a dedicated vLLM instance on RunPod/Lambda for multi-camera production throughput. We do not need to make that decision yet.

---

## Current constraints

Being honest about what is and is not in place — these are known and managed, not surprises.

- **Local hardware is the bottleneck.** The current dev machine has 16 GB system RAM and a 2017-era Nvidia MX150 with only 2 GB of VRAM. Gemma 4 26B-A4B cannot run on it at any quantization. This is why OpenRouter is the right answer now — it removes the hardware blocker entirely. Any future shift to self-hosting requires either a 24 GB consumer GPU (RTX 4090 or used 3090/A5000 class) or a rented cloud GPU instance.

- **We do not yet have a labeled clip set.** The evaluation harness is built and verified, but it has not been run against real footage. We do not yet have empirical numbers for how Gemma 4 performs on our specific environment distribution — that data comes from the next session. Building the test set across all 15 environment types is a manual sourcing exercise.

- **`scene_context.json` is currently an orphan artifact.** The Mapper produces clean structured output, but nothing downstream consumes it yet. The Preset Recommender and Customization Engine are the next two layers and they are not built. The output is ready for them; they are not yet ready for the output.

- **OpenRouter free-tier has rate limits.** Fine for development and evaluation, will not survive any real pilot traffic. The paid tier is cheap (~$0.30 per 1,000 calls) and easily absorbed once we have real cameras, but the migration from free to paid is a future operational step.

- **Detection Core is split across branches.** Local `main` is at a POC level. The collaborator branch `ayo/main` has materially stronger detection logic — ByteTrack tracking, a temporal violence gate, a theft state machine, and an evaluation harness. These need to be reconciled into one trunk before the architecture is end-to-end coherent.

- **Verification Gate does not exist yet.** Even a perfect Agent Mapper plus a perfect Customization Engine leaves one source of false positives: edge-case detector misfires that match a configured rule. The Verification Gate (VLM confirms or rejects alerts before they escalate) is the second structural FPR fix, and it has not been built. It is in the architecture and on the roadmap, but the user should not expect it in the V1 ship.

- **No fine-tuning yet.** Gemma 4 is being used out of the box with prompt engineering only. Generic VLM training data underweights Nigerian-specific scenes — warehouse layouts, motorcycle commercial areas, generator-area edge cases. If accuracy on those specific scenes proves weak in evaluation, the longer-term answer is a small fine-tuning effort on our own collected footage. That is Phase 2 data strategy, not blocking the V1 launch.

- **Schema validator is currently loaded but not enforced at runtime.** Validation is done by hand-rolled normalizers in the code. A VLM returning extra fields would have them silently persisted. This is acceptable for the current phase and easy to tighten with a real JSON Schema validator when we lock the contract for external consumers.

---

## What is next

In priority order, the immediate roadmap:

1. **Run the live VLM evaluation** — drop labeled clips into `tests/agent_mapper/clips/<env>/`, run the harness against Gemma 4 via OpenRouter, confirm the model holds the descriptive-only contract and classifies environments accurately on real footage.

2. **Build the Preset Recommender** — small deterministic component that consumes `scene_context.json` and a static `presets.json` lookup, producing a suggested `user_config.json` draft. No VLM, just policy mapping. This is what bridges the Mapper to the user.

3. **Build the Customization Engine** — the heart of the product. Reads the authoritative `user_config.json`, evaluates rule triggers against raw detection events, produces candidate alerts. This is what makes the same backend sellable across verticals.

4. **Lock `user_config.schema.json`** — the rule contract. Should mirror the discipline of the scene context schema: bounded vocabulary, no ambiguity.

5. **Implement the GTM-12 rules** — loitering, perimeter intrusion, after-hours presence, crowd formation, running, tailgating, abandoned object, unauthorized vehicle, camera tampering, person down, mask during business hours, power outage + motion.

6. **Build the Verification Gate** — VLM confirms or rejects candidate alerts against the specific rule that fired, not against a vague "is this bad" judgment. This is the second structural FPR fix, complementing the user-narrowing fix from the Customization Engine.

---

## What you can show in the demo

- The four-layer diagram in `architecture.md`
- The locked schema at `schemas/scene_context.schema.json`
- The descriptive-only prompt at `prompts/agent_mapper_prompt.txt`
- A live Mapper run against a real scene image using Gemma 4 via OpenRouter — descriptive JSON output in roughly five seconds
- The test harness measuring valid-JSON rate, environment accuracy, and threat-vocabulary leakage across a clip set
- The build plan at `docs/AGENT_MAPPER_PLAN.md`

---

## The one-line summary

We have a working, descriptive-only Agent Mapper running on a current-frontier open-source vision model, with a structurally clean separation between scene understanding and threat policy, and a hosting setup that costs nothing during development and pennies in production. The next layer up is the Customization Engine, which is where the product becomes sellable across verticals.
