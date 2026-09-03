# Agent Mapper Initial Baseline: Gemma 3 4B

Run date: 2026-09-03  
Model: `gemma3:4b` via local Ollama  
Manifest: `tests/agent_mapper/labels.json` (`initial-15-v1`)  
Raw rows: `baseline_gemma3_4b_initial15_20260903.csv`

## Result

| Metric | Result |
|---|---:|
| Valid schema output | 15/15 (100%) |
| Exact environment match | 7/15 (46.7%) |
| Acceptable environment match | 10/15 (66.7%) |
| Clips with threat-language leakage | 0/15 (0%) |
| Hard provider/parser errors | 0/15 (0%) |
| Average provider latency | 16.515 s |

Per expected environment:

| Environment | Exact | Acceptable |
|---|---:|---:|
| `retail_shop` | 3/3 | 3/3 |
| `parking_lot` | 1/3 | 2/3 |
| `estate_street` | 1/3 | 1/3 |
| `office_floor` | 2/3 | 3/3 |
| `residential_exterior` | 0/3 | 1/3 |

## Verdict

Keep `gemma3:4b` as the temporary local Mapper model because it produced
schema-valid, descriptive-only output on every clip and was strong on the
retail and office scenes. Do not auto-trust its environment label: outdoor
granularity is weak, so the existing human scene-review gate remains required.
Finish the planned 45-clip/15-environment set before selecting a permanent
model, then run the same manifest against the installed `qwen2.5vl:3b` as the
first small-model comparison.

## Interpretation limits

- This is a smoke baseline, not the completed M3 benchmark.
- The 15 clips are a small convenience sample from existing public/locally held
  collections; they are not representative of Nigerian pilot sites.
- Clips `10.mp4` and `398.mp4` were labeled `estate_street` but prominently
  include gates. Their `estate_gate` predictions expose a ground-truth
  adjudication question. This first result is preserved unchanged; relabeling
  requires an independent review before the next run.
- Smoke obscures scene context in two residential clips. Those are useful hard
  cases, but three clean residential negatives must be added to separate
  environment-recognition quality from visibility degradation.
- Media redistribution rights remain unresolved. Videos stay gitignored; the
  manifest records SHA-256 checksums for exact-file verification.

## Reproduction

```bash
OLLAMA_API_KEY=ollama python tests/agent_mapper/eval.py \
  --models gemma3:4b \
  --provider openai_compatible \
  --api-base-url http://127.0.0.1:11434/v1 \
  --clips-dir "/path/to/checkout-containing-data"
```
