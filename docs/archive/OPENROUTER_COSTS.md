# OpenRouter Cost Implications — Agent Mapper

Captured 2026-06-06. Pricing pulled live from OpenRouter's `/api/v1/models`
endpoint for the model the Agent Mapper uses: `google/gemma-4-26b-a4b-it`.

## Two tiers

### Free tier — `google/gemma-4-26b-a4b-it:free`
- **Cost: $0.00.** Genuinely free.
- The only friction is occasional "too busy" responses (HTTP 429) from the
  shared free pool. The Mapper's retry/backoff in `agent_mapper.py` rides
  through these, so they don't break runs.
- There is a **daily usage cap** on free models. It's low until you've added a
  small amount of credit (historically ~$10), which raises the daily allowance
  substantially. Check OpenRouter's current limits page for exact numbers.
- **This is the dev/eval path. Credits are NOT required to build or test.**

### Paid tier — `google/gemma-4-26b-a4b-it`
Only needed if you want zero flakiness.

| Direction | Price |
|---|---|
| Input  | **$0.06 per million tokens** (`$0.00000006`/token) |
| Output | **$0.33 per million tokens** (`$0.00000033`/token) |

No separate per-image or per-request fee — the image counts as input tokens.

## What one Agent Mapper call costs (paid tier)

A token is roughly ¾ of a word; an image counts as a few hundred-to-thousand
tokens. One Mapper call is approximately:

- Input: prompt + one image ≈ ~3,000 tokens → 3,000 × $0.00000006 = **$0.00018**
- Output: the small JSON it returns ≈ ~300 tokens → 300 × $0.00000033 = **$0.0000099**
- **Total ≈ $0.0002 per call** — about two-hundredths of one cent.

## In real terms

| Usage | Paid-tier cost |
|---|---|
| 1 call | ~$0.0002 (0.02¢) |
| 1,000 calls | ~$0.20 |
| A **$5** top-up | ~20,000–25,000 calls before it runs out |
| Production: 100 cameras, 1 call every 5 min, 24/7 | ~28,800 calls/day ≈ **~$6–9/day** (~$200–260/month) |

## Bottom line

- **Building + testing now:** $0. Free tier covers everything; we just retry on
  busy signals.
- **To kill the flakiness:** ~$5 lasts basically the whole project. Gemma 4 is
  one of the cheapest capable vision models available.
- **At real scale (hundreds of live cameras):** still single-digit dollars per
  day, because the Mapper runs *infrequently* (once every few minutes per
  camera, not every frame). That infrequency is what makes this layer cheap by
  design.

## Caveat

These prices are for **Gemma 4** specifically. Switching to a heavier model
(GPT-class, Claude, Gemini) can be **10×–100× more expensive** per call. The
cost story above holds as long as you stay on Gemma 4 or a similarly small
open model.
