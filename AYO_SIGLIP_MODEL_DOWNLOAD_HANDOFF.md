# Ayo Handoff: Include SigLIP in the AI Download

## The Request

Ayo, can we include the object-recognition model in the same customer-facing
AI download flow as Gemma? When a customer clicks **Download AI**, Argus should
install both the verification model and the object-recognition model, with no
terminal commands or manual model-folder setup.

The model's name in our code is **SigLIP**, not SigCLIP.

## Why We Need This

Shipping the object-recognition code alone is not enough. The code also needs
the trained model weights and image-processing configuration on the customer's
machine. A model working on a developer's laptop does not mean it is present
on a fresh customer installation.

Gemma and SigLIP do different jobs:

| Component | Purpose |
| --- | --- |
| Object detector/proposal provider | Finds candidate objects or regions to inspect. |
| SigLIP | Converts reference photos and camera crops into numerical representations so we can compare visual similarity. |
| Gemma via the VLM verification gate | Assesses visual evidence and context for configured alert-verification workflows. |

Downloading Gemma does **not** also install SigLIP or satisfy its dependencies.
For example, a customer uploads photos of a CHI product. SigLIP is the matching
component that helps compare those references with candidate camera crops.
It does not guarantee exact product identification, read every label, or prove
theft. Camera quality, similar packaging, reference examples and matching
thresholds still need testing.

## What the Inspected Code Already Does

Based on the local checkout at commit `84b3418`; this is not a fresh audit of
any later changes Ayo may have pushed.

- `cvti/object_watch/embeddings.py`: the production `siglip` backend loads a
  local model directory with `local_files_only=True`. It does not automatically
  fetch missing weights when recognition starts.
- `cvti/object_watch/runtime_config.py`: readiness checks require `config.json`,
  `preprocessor_config.json` and model weight artifacts. Missing files make the
  feature unavailable. Structural readiness alone is not a successful inference test.
- The current configuration validator supports original SigLIP, and explicitly
  rejects SigLIP2 configurations. Pick a checkpoint compatible with this loader.
- `cvti/app/console_backend.py`: object-watch configuration requires SigLIP;
  the hash backend is test-only, not a production replacement.

## Recommended Delivery

**Bundle it into the download experience, not necessarily into the installer.**
Keep the desktop installer separate from the large model assets. The existing
AI setup flow should orchestrate the downloads as separate, versioned components:

1. Install or verify the Gemma model and its runtime.
2. Install or verify the selected SigLIP checkpoint and all required processor,
   configuration, weight and index files.
3. Store SigLIP under a persistent, writable application-data model directory
   and configure object watch to use that path automatically.
4. Verify file checksums, load the model and run a small embedding smoke test.
5. Show each component's real status: downloading, ready, failed or retry required.

This does not mean loading SigLIP through Ollama. Keep its current
Transformers/PyTorch loader; share the installer/download UI, not the inference
runtime. Ensure those Python dependencies are packaged in the engine too.

Downloads should be resumable or retryable, check available disk space, and only
activate a model after verification completes. Do not redownload it on each
launch. Ordinary local matching should work without an internet connection once
installation is complete; reference photos need not be uploaded to a cloud service.

If object recognition is included in the customer's setup, include SigLIP in
that setup's readiness requirements. A SigLIP failure should clearly disable
object recognition, not prevent unrelated camera viewing or monitoring features
from working. Never silently substitute the test backend.

## Details to Agree Before Implementation

- Pin the exact compatible checkpoint and revision; verify its redistribution
  terms before hosting or shipping its assets. Download size depends on that choice.
- Package every required model for the chosen proposal provider as well.
  SigLIP alone does not install YOLO or optional YOLO-World dependencies.
- Keep reference embeddings tied to the model fingerprint and preprocessing
  version. On a model upgrade, rebuild incompatible reference embeddings rather
  than comparing vectors from different model versions.
- Treat appearance matching as evidence, not calibrated certainty. Preserve the
  existing candidate filtering and configured alert-verification policy.

## Acceptance Checks

- On a clean Mac and Windows installation with no developer model cache, the
  customer completes AI download without manually copying weights or setting paths.
- Reference-photo enrollment and camera-crop matching work after disconnecting
  the internet and after restarting Argus.
- Interrupted or corrupt downloads produce a clear retry path, not a false Ready state.
- Missing SigLIP is reported specifically as an object-recognition dependency.
- A model upgrade cannot silently reuse incompatible reference embeddings.
- Validate recognition on representative CHI products and confusing non-matches;
  installing the model is not the same as meeting the recognition KPI.

## Short Message

“Ayo, can we add SigLIP to the same Download AI flow as Gemma? Gemma handles VLM
verification, while SigLIP matches objects against the customer's reference
photos. Our recognition code needs SigLIP's files locally, so a fresh customer
installation needs to receive them automatically. They can remain separate
model packages, but from the customer's side it should be one guided setup.”

This document proposes packaging/download work; it does not implement it.
