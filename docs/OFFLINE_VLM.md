# Offline VLM (local verification)

CVTI's verification gate and scene mapper can run against a **local** vision-language
model instead of a cloud API. No API key, no internet, no per-call cost. This is the
`local` provider (`ollama` in the Agent Map panel).

## How it works

The gate/mapper talk to an [Ollama](https://ollama.com) server over its
OpenAI-compatible endpoint (`http://localhost:11434/v1`). The gate is **throttled** —
the VLM only runs on a *new* alert, not every frame — and Ollama unloads an idle model
after ~5 minutes, so RAM is only used in short bursts during verification.

## One-time setup

1. Install Ollama: https://ollama.com/download (or `brew install ollama` on macOS).
2. Start the server (the app also tries to start it for you): `ollama serve`
3. Pull a model (the app offers to do this on first Start): `ollama pull gemma3:4b-it-qat`

## Choosing a model

Ollama models are **already 4-bit quantized** (Q4_K_M) by default — you don't quantize
anything yourself. Approximate memory *while a verification call is running* (added on
top of the ~1.5–2.5 GB the YOLO + Qt + Python stack uses):

| Model (Ollama tag)      | Params | Loaded RAM | Notes |
|-------------------------|--------|-----------|-------|
| `gemma3:4b-it-qat`      | 4B     | ~3.3 GB   | **Default.** Quantization-aware Gemma 3 — BF16-level quality at int4. ~6 GB peak. |
| `gemma3:4b`             | 4B     | ~3.3 GB   | Plain Q4 Gemma 3. Same size, slightly lower quality than QAT. |
| `moondream`             | 1.9B   | ~1.6–2 GB | Low-RAM fallback (~4 GB total) — weaker on complex scenes, higher error rate. |
| `qwen2.5vl:7b`          | 7B     | ~6 GB     | Strongest small VLM; heavier. |
| `llama3.2-vision`       | 11B    | ~8 GB     | Highest quality; tight alongside everything else. |

> **Gemma vision floor:** `gemma3:270m` and `gemma3:1b` are **text-only** — they cannot
> read frames. `gemma3:4b` is the *smallest* Gemma 3 with vision, so there is no way to
> run a Gemma vision model under a true ~4 GB total budget. Use the QAT build
> (`gemma3:4b-it-qat`) for the best quality at that ~3.3 GB footprint.

On an 18 GB Apple Silicon machine running YOLO + the Qt app, **`gemma3:4b-it-qat`** is
the default (~6 GB peak, well within budget). `moondream` is only a last-resort fallback
if you must stay near ~4 GB total — expect lower verification accuracy.

## Using it

**Desktop app:** set `Gate` → `local`, pick a `Model`, press Start. If the model
isn't present the app offers to download it and streams progress in the status bar.
For scene mapping, open the **Agent Map** tab and choose provider `ollama`.

**CLI:**

```bash
# retail pipeline with local verification
python -m cvti.pipelines.retail_pipeline \
  --source data/test_clips/theft_shop_01.mp4 \
  --config configs/retail_pipeline_v1.json \
  --gate-provider local --gate-model moondream

# point at a non-default Ollama host
python -m cvti.pipelines.retail_pipeline ... \
  --gate-provider local --gate-base-url http://localhost:11434/v1
```

## Packaging (bundled Ollama, model on first run)

The build **bundles the Ollama runtime** inside the app (on macOS this is ~430 MB
unpacked / ~130 MB added to the compressed installer — it ships CPU runners for many
archs plus the MLX/Metal runners; smaller on Linux/Windows), so the client does **not**
install Ollama separately. The ~3.3 GB model is **not** embedded — it downloads
automatically on first launch (needs internet once). This keeps the installer small
while removing all manual setup.

Build flow:

1. `scripts/fetch_ollama.sh` (or `.bat` on Windows) downloads the Ollama binary into
   `vendor/ollama/<platform>/`. The platform build scripts call this automatically.
2. `pyinstaller cvti.spec` bundles that binary (and the YOLO weights, configs, prompts,
   schemas as before). If `vendor/ollama/` is absent, the bundle is built without it and
   the app falls back to an Ollama already on the machine.
3. On first launch with `Gate → local`, the app starts the bundled Ollama server and
   offers to pull the model, streaming progress in the status bar.

At runtime the app prefers the bundled binary and falls back to a system `ollama` on
`PATH` (see `cvti/verification/ollama.py: ollama_binary()`).

> Fully-offline variant: to also embed the model (zero internet, ~4 GB installer), pre-place
> the pulled model blobs and point `OLLAMA_MODELS` at a writable copy on first run. Not
> wired up here — the default is bundle-binary + pull-on-first-run.
