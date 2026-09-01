# Agent Mapper Operations

## Purpose

The Agent Mapper describes each camera before alert processing starts. It
produces a bounded scene context containing the environment, a description,
possible actors, and suggested zones. It does not decide whether an event is a
threat and it does not activate suggested zones automatically.

The production flow is:

```text
camera source (every camera maps independently)
  -> persistent one-worker mapping queue
  -> camera context + deterministic area/site proposals
  -> grouped operator review
  -> detector and customer rules
  -> deterministic context compatibility
  -> TrueSight VLM verification
```

This prevents category errors before the gate. For example, a built-in
shoplifting candidate is suppressed in a reviewed `parking_lot` context unless
the customer explicitly configured an applicable merchandise or checkout zone.
Critical baseline safety rules such as fire, weapon, serious violence, person
down, and camera tampering are never blocked by scene compatibility.

## Policies

Set `scene_context_policy` at the top level of a site JSON:

- `require_reviewed`: map missing/stale cameras and run them in
  `critical_only` mode until an owner or installer approves the result.
- `auto`: use a valid unreviewed mapping immediately and show its review state.
  This is the compatibility default when an old site has no policy field.
- `manual`: never invoke the mapper; every camera must have a human-authored
  `scene_description` and may also specify `environment_type` and
  `expected_actors`.

The first-run Argus wizard writes `require_reviewed` for genuinely new sites.
Reopening the wizard does not rewrite an existing legacy site's policy.

## Local Ollama Setup

From the repository root:

```bash
cd "/Users/macbook/Desktop/CV Threat Intelligence/cv-threat-intelligence"
source .venv/bin/activate
ollama list
curl -s http://127.0.0.1:11434/api/tags
```

If the `curl` command returns model data, Ollama is already running. Do not
start a second `ollama serve` process. The mapper and TrueSight should use the
same local vision model to avoid loading a second large model.

## Review-Gated End-to-End Test

Create a disposable site from the two-camera test config and enable mandatory
review:

```bash
cp configs/site_full_demo.json /private/tmp/argus_mapper_review.json
./.venv/bin/python -c 'import json; p="/private/tmp/argus_mapper_review.json"; d=json.load(open(p)); d["scene_context_policy"]="require_reviewed"; open(p,"w").write(json.dumps(d,indent=2))'
```

Run mapping preflight with local Ollama:

```bash
MPLCONFIGDIR=/private/tmp OLLAMA_API_KEY=ollama ./.venv/bin/python -m cvti.serving.pipeline \
  --site-config /private/tmp/argus_mapper_review.json \
  --gate-provider ollama \
  --gate-model gemma3:4b \
  --mapper-provider ollama \
  --mapper-model gemma3:4b \
  --mapper-base-url http://127.0.0.1:11434/v1 \
  --notify console \
  --output-dir runs/agent_mapper_review \
  --target-fps 4 \
  --seconds 30 \
  --gate-drain 30
```

On the first run, newly mapped cameras should be `ready_unreviewed`. They remain
live for critical baselines (fire, weapon, serious violence, person down and
tampering), while context-sensitive customer rules remain paused.

Open the Argus console against the same site and output directory:

```bash
./.venv/bin/python -m cvti.app.shell \
  --site-config /private/tmp/argus_mapper_review.json \
  --db runs/agent_mapper_review/events.db
```

Sign in as an owner or installer. The grouped **Scene review** opens after
onboarding and presents one physical area at a time, with every independent
camera view. Confirm, edit and confirm, reject/remap, or keep an area paused.
The Rules screen remains the camera-specific maintenance path. Suggested zones
remain inert until **Accept zone** is selected explicitly.

Approval is picked up by the running detector process within its health cycle;
the affected camera switches to full rules without rebuilding models or
restarting unrelated cameras.

## Automatic Compatibility Test

For a quick one-command run, use a site with `scene_context_policy: "auto"`.
The mapper will create context and the camera may start immediately. This mode
is useful for development, but it is weaker operationally because the mapping
has not been reviewed.

To reproduce the reported category-error case, create a one-camera config whose
source is a genuine car-park clip and whose detector config includes the retail
shoplifting rules. A successful result has all three properties:

1. The saved context says `parking_lot`, or an operator corrects it to that.
2. A shoplifting rule match is recorded as context-suppressed before TrueSight.
3. Critical safety candidates still reach TrueSight.

Do not use `--gate-provider mock` to judge mapping or hallucination quality.
Mock confirms every alert and only proves plumbing. A real parking clip and the
local Ollama VLM are required for this evaluation.

## Artifacts

Each camera writes under the selected output directory:

```text
runs/<run>/context/<camera_id>/
  scene_context.json
  mapping_status.json
  source_frame.jpg
  raw_response.txt       # only when debug persistence is enabled
runs/<run>/context/_areas/<area_id>/
  area_context.proposal.json
  area_context.json      # after approval
runs/<run>/context/_site/
  site_context.proposal.json
  site_context.json      # after approval
runs/<run>/context/mapping_queue.json
```

`mapping_status.json` uses these states:

- `pending`: mapping has not completed.
- `ready_unreviewed`: valid mapper output exists but no human approved it.
- `ready_reviewed`: approved context can start under every policy.
- `stale`: the source fingerprint changed or remapping was requested.
- `failed`: mapping or required manual context failed.

Overall status is also published in `runs/<run>/gate_health.json`. A failure or
review block affects only that camera; other usable cameras continue.

## Recovery

- **Ollama unavailable:** confirm `curl` works, confirm `ollama list` contains
  the model tag, then rerun. The camera remains visibly failed rather than
  silently using generic context.
- **Wrong context:** edit and approve it in Rules, or select Remap. Remap marks
  the current artifact stale; it does not erase the evidence immediately.
- **Changed source:** rerun the pipeline. A changed credential-redacted source
  fingerprint invalidates the old cache.
- **Camera limited after mapping:** approve its area or camera context. It will
  leave `critical_only` mode without a pipeline restart.
- **RTSP privacy:** fingerprints are hashes of credential-redacted source
  identity. Passwords must never appear in context, health, logs, or UI.

## Large Sites

Every camera still makes an independent visual inference, even when several
cameras are assigned to the same physical area. Area assignment is a prior,
never a copied visual label. Camera observations are reconciled
deterministically; high-confidence disagreement becomes a visible conflict and
disables bulk confirmation. The persistent queue defaults to one local Ollama
request at a time, survives restarts, and was integration-tested with 100
cameras grouped into 20 areas.
