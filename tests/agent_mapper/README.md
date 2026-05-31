# Agent Mapper VLM evaluation

Compares open-source VLMs on the descriptive-only `scene_context.json` task. The
harness reuses the same functions `agent_mapper.py` calls at runtime, so it tests
what we actually ship.

## Layout

```
tests/agent_mapper/
  clips/<env>/<clip_id>.mp4     # ground-truth clips (gitignored)
  labels.json                   # one entry per clip with expected environment_type
  eval.py                       # runs N models against M clips, writes a CSV
  results/eval_<run_id>.csv     # one row per (model, clip) — gitignored
```

## One-time setup

### 1. Ollama

Ollama is already installed on the dev machine. The OpenAI-compatible endpoint
runs on `http://localhost:11434/v1`. Confirm with `ollama list`.

### 2. Pull the candidate VLMs

The right starting set as of 2026-05 — Ollama-supported, vision-capable, MoE for
inference efficiency where possible. Pick based on available VRAM.

```powershell
# Qwen3-VL — dedicated vision-language line, vision is properly wired in Ollama.
# 4B is the laptop-friendly tier (~2.5 GB Q4); 8B is the practical sweet spot
# (~5-6 GB Q4, 12-16 GB VRAM); 32B is the strong tier (~24 GB VRAM).
ollama pull qwen3-vl:4b
ollama pull qwen3-vl:8b
# ollama pull qwen3-vl:32b   # only if you have >=24 GB VRAM

# Gemma 4 — Google's April 2026 multimodal release. The 26B-A4B is an MoE with
# only ~3.8B active params (128 experts), 256K context, vision-native. Runs at
# near-4B-dense speed but the weights footprint is ~13-15 GB at Q4.
ollama pull gemma4:26b
# Edge variant for lower VRAM budgets:
# ollama pull gemma4:e4b
```

Verify they downloaded:

```powershell
ollama list
```

**Do not use `qwen3.6` for Mapper testing.** Vision is broken in Ollama for that
model right now — the `mmproj` projector ships as a separate file and Ollama's
GGUF flow does not wire it up. Text-only would work, but image input fails. Stick
to `qwen3-vl:*` and `gemma4:*` for anything that takes a frame.

Models in the user's Ollama list that are **not** Mapper candidates:
- `kimi-k2.6:cloud`, `glm-5.1:cloud`, `nemotron-3-super:cloud`, `gemma4:31b-cloud` — cloud-only, defeats the "no paid API" goal of this phase
- `glm-5.1` — 754B, coding/agentic-focused, not vision-first

### 3. Set a dummy API key

Ollama's OpenAI-compatible endpoint doesn't validate the key, but the harness
expects one to be set in the env:

```powershell
$env:OLLAMA_API_KEY = "ollama"
```

(For a persistent setting, use `[System.Environment]::SetEnvironmentVariable("OLLAMA_API_KEY", "ollama", "User")`.)

## Add clips and labels

1. Drop labeled clips into `tests/agent_mapper/clips/<environment_type>/` — for
   example `tests/agent_mapper/clips/estate_gate/lekki_phase1_gate_01.mp4`.
2. Add a matching entry to `labels.json`:

```json
{
  "clip_path": "tests/agent_mapper/clips/estate_gate/lekki_phase1_gate_01.mp4",
  "expected_environment_type": "estate_gate",
  "acceptable_environment_types": ["estate_gate", "estate_street"],
  "notes": "Wide shot of estate vehicle gate during daytime."
}
```

Aim for ~3 clips per environment_type so the per-model accuracy numbers are
meaningful. Use `schemas/scene_context.schema.json` for the canonical list of
allowed environment values.

## Run the eval

```powershell
python tests\agent_mapper\eval.py --models qwen3-vl:8b,gemma4:26b
```

Useful flags:

- `--filter-env estate_gate` — only run clips whose expected env matches
- `--sample-count 5` — sample more frames per clip before picking the best one
- `--api-base-url http://localhost:11434/v1` — change if running vLLM / LM Studio
- `--provider mock` — sanity-check the harness end-to-end without a live model

The console shows per-clip results live, then a per-model summary at the end.
Full results land in `tests/agent_mapper/results/eval_<run_id>.csv`.

## What the harness measures

For each (model, clip) pair:

| Column | Meaning |
|---|---|
| `valid_json` | The provider response parsed without throwing. |
| `env_match` | Predicted `environment_type` equals the expected one exactly. |
| `env_acceptable` | Predicted env falls in the `acceptable_environment_types` list (close alternates count). |
| `latency_s` | Wall-clock time for the provider call only (not frame sampling). |
| `leaked_terms_count` | How many threat-vocabulary terms appeared in `scene_description` or `notes`. Should be **0** for a clean descriptive Mapper. |
| `leaked_terms` | The actual leaked tokens (e.g. `loiter,suspicious`) — useful for prompt tuning. |
| `scene_description` | Verbatim description text — eyeball this for quality. |

Leak detection is a structural correctness check, not just a soft signal: per
the 2026-05-24 refactor, the Mapper is descriptive-only and should not produce
threat semantics. A high leak count means the model is ignoring the prompt and
the prompt likely needs tightening.

## Picking a winner

The smallest model that crosses your quality bar wins. For Agent Mapper the
call is infrequent (once per session / every ~5 min per camera), so raw latency
is forgiving. Prioritize in this order:

1. `valid_json` rate — must be ~100%, this is the fundamental contract.
2. `leaked_terms_count` — must be ~0 across the test set.
3. `env_acceptable` rate — should be high; exact `env_match` is the stretch goal.
4. `scene_description` quality — eyeball check on a sample.
5. Latency and VRAM — only as tiebreakers between models that pass the above.

Don't default to the biggest model.
