# Gate VLM Bake-Off — On-Device Model Selection (v1)

**Goal:** the Verification Gate's VLM must run **locally on the edge device** (offline, fits the 4–5 GB budget). This document compares candidate small VLMs (run locally via Ollama) on *our* task to pick one — and to produce a validated accuracy/FPR number per the "no model ships without a validated FPR" rule.

## What the gate is (scope of this test)
The gate is the **false-positive filter**: given a candidate event's frame(s) + the fired rule, it **confirms or rejects** before an alert is raised. This bake-off tests the gate on the **concealment / shoplifting** decision specifically — the *hardest, most ambiguous* gate call. It does **not** yet test violence or weapons (those are more visually obvious; the harness can be extended to them). It is **not** the full detector — the full system (`detector.py`) runs violence + weapons + theft + concealment + zones together.

## Method
- **6 labeled clips:** 3 concealment-positive (gate *should* confirm) + 3 normal (gate *should* reject).
- For each clip the **peak-concealment frame** is auto-selected and the **same frame is sent to every model** (fair comparison).
- Each model judges through the local Ollama gate. We score: **recall** (caught the positives), **specificity** (correctly cleared the normals), **accuracy**, **valid-JSON rate**, **latency**.
- Run with: `python tools/gate_bakeoff.py --models gemma3:4b,qwen2.5vl:3b --verbose`
- ⚠️ **Tiny set — directional, not final.** Expand the labeled set before quoting an FPR.

## Results (2026-06)

| model | recall | specificity | accuracy | valid JSON | latency/call |
|---|---|---|---|---|---|
| **gemma3:4b** | 33% (1/3) | **100% (3/3)** | 67% | 6/6 | ~7.5 s |
| **qwen2.5vl:3b** | 100% (3/3) | **0% (0/3)** | 50% | 6/6 | ~8 s |
| moondream | _not tested_ | | | | |

## Interpretation
- **`gemma3:4b` — conservative & well-grounded (the better gate).** It rejected **every** normal clip — including the hard one where our own concealment detector over-fired to 1.0 — and correctly called the empty warehouse *"empty, no people."* On a clear shoplifting frame it confirmed at **0.95** (*"concealing a bottle in their clothing"*). Its misses are largely **frame-selection** (e.g. `theft_yt_01`'s chosen frame showed a *vehicle and ATM*, not the theft moment). **Profile = exactly what a gate needs: it does not cry wolf.**
- **`qwen2.5vl:3b` — rubber-stamps "yes" (unsuitable as a gate).** It confirmed **all 6**, including normal street scenes **and an empty warehouse where it hallucinated a person** (*"a person moving quickly"*). **0% specificity** means it provides **no filtering** — as a gate it would confirm every false alarm. Its "100% recall" is meaningless because it says yes to everything.

## Recommendation
**Use `gemma3:4b` (or the team's `Gemma-4-E4B` once confirmed on Ollama) as the gate.** For the gate's job, **specificity + grounding** are what matter (the candidate generator provides recall; the gate provides precision). Gemma discriminates and grounds; Qwen does not. This also matches the team's prior bias toward Gemma for vision.

## Caveats & next steps
1. **Expand the labeled set** (add `data/anomaly/` robbery clips as positives, more normals) → trustworthy recall/specificity = the **validated FPR** for the model registry.
2. **Test the team's exact model:** this run used **Gemma 3 4B**; the shortlist was **Gemma-4-E4B-it**. Try `ollama pull gemma4:e4b`.
3. **Frame selection depresses recall** — raise `--max-frames` (or `0` for full clips) and test **multi-frame** (`--gate-frames 3`); the same frame is given to all models so the *comparison* is fair, but absolute recall is understated.
4. **Extend the gate test to violence + weapons** for a full-gate eval.
5. **Edge note:** ~8 s/call on a Mac is fine because the gate runs **infrequently** (only on candidate events); expect it slower but still acceptable on the Jetson.
