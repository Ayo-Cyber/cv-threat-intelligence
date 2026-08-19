# CVTI — System Guide

**CV Threat Intelligence** is a single, unified detection tool. Fast per-frame
detectors watch the video; when one flags a suspicious moment, an optional
video-action classifier adds a temporal opinion, a config-driven rules engine
decides whether *you* care about it, and a VLM (vision-language model) gate
confirms or rejects the alert before it is raised. Everything runs in one
pipeline, sharing one event stream.

Status (verified 2026-07-11): the core pipeline runs end-to-end on CPU and
produces confirmed, saved alerts. 33 unit tests pass. The optional VideoMAE
layer requires one extra install step (below).

---

## 1. How it works (the pipeline)

Every frame flows through the same loop in
[`cvti/detector/core.py`](../cvti/detector/core.py):

```
                         ┌─────────────────────────────────────────────┐
   video frame ─────────▶│  DETECTION  (runs on every frame)           │
                         │  • YOLO objects / weapons                    │
                         │  • pose  → violence heuristics               │
                         │  • theft / concealment / zones (opt-in)      │
                         └───────────────┬─────────────────────────────┘
                                         │ a detector fires = "event center"
                                         ▼
                         ┌─────────────────────────────────────────────┐
        (optional)       │  VIDEO ACTION  (runs only when triggered)    │
                         │  • VideoMAE samples ~16 frames around the    │
                         │    moment → weak temporal labels             │
                         └───────────────┬─────────────────────────────┘
                                         │ all detectors emit RawEvents
                                         ▼
                         ┌─────────────────────────────────────────────┐
                         │  CUSTOMIZATION ENGINE  (your JSON rules)     │
                         │  keeps only events your config cares about   │
                         │  → CandidateAlert                            │
                         └───────────────┬─────────────────────────────┘
                                         │ top alert
                                         ▼
                         ┌─────────────────────────────────────────────┐
                         │  VERIFICATION GATE  (VLM)                    │
                         │  sees the alert + frames → confirm / reject  │
                         └───────────────┬─────────────────────────────┘
                                         │ confirmed
                                         ▼
                              raise + save event (clip.mp4, frame.jpg, metadata.json)
```

This maps 1:1 onto the design:

1. YOLO / pose / concealment run on each frame.
2. If a detector sees a suspicious moment, that frame becomes the **event center**.
3. VideoMAE samples frames *around* that center (window of N seconds).
4. VideoMAE returns **weak** temporal labels (e.g. "punching person").
5. Useful labels become weak `RawEvent`s (`detector="video_action"`).
6. The `CustomizationEngine` checks whether your config cares about that event.
7. The `VerificationGate` (VLM) gets the alert + visual evidence and confirms/rejects.
8. If confirmed, the system raises and saves the alert.

> **Design note.** VideoMAE is a *supporting temporal witness*, not the final
> judge. It only runs *after* a per-frame detector raises a moment, and its
> output is a weak signal — the VLM gate is what actually confirms an alert.

---

## 2. Components

| Component | Module | Role |
|---|---|---|
| Detector core | `cvti/detector/core.py` | Per-frame YOLO + pose + violence/theft; owns the main loop; entry point `cvti-detect` |
| Concealment | `cvti/retail/concealment.py` | Pose-based concealment (pocket/bag = threat, trolley = safe) |
| Zones | `cvti/retail/zones.py` | Restricted-zone / dwell (loitering, after-hours) |
| Video action | `cvti/video_action_{model,hybrid,runtime}.py` | VideoMAE/X3D weak temporal classifier |
| Event adapters | `cvti/event_adapters.py` | Turn detector state into shared `RawEvent`s |
| Customization | `cvti/rules/customization.py` | Config-driven rule matching → `CandidateAlert` |
| Verification gate | `cvti/verification/gate.py` | VLM confirm/reject (mock / anthropic / openrouter / ollama / local) |
| Contracts | `cvti/contracts.py` | Shared `RawEvent`, `CandidateAlert`, `VerificationResult` |
| Retail pipeline | `cvti/pipelines/retail_pipeline.py` | Retail-focused orchestration; entry point `cvti-retail` |

**Entry points** (from `pyproject.toml`): `cvti-detect`, `cvti-retail`,
`cvti-train`, `cvti-eval`, `cvti-infer`, `cvti-suite`, `cvti-app`.

---

## 3. Installation

```bash
# Core detection (YOLO, pose, gate, customization)
pip install -e .

# Optional: the VideoMAE / X3D video-action layer
pip install -r requirements-video.txt   # transformers, accelerate, safetensors, pytorchvideo
```

**Model weights** live in `models/`. `yolov8n.pt` and `yolov8n-pose.pt` are
auto-downloaded by ultralytics on first run (needs internet once). A custom
weapon model (`models/weapon_best.pt`) is included.

---

## 4. Running it

### Quick smoke test (mock gate, capped frames)

```bash
python3 -m cvti.cli.detect \
  --source data/test_clips/violence_suspected.mp4 \
  --config configs/all_threats_v1.json \
  --gate-provider ollama \
  --concealment \
  --max-frames 40
```

Expected output (verified):

```
[CustomizationEngine] Loaded 4 rules for use-case 'all_threats_v1'
[Concealment] Pose-based concealment detector ON
[VerificationGate] Provider: mock (all alerts auto-confirmed).
[CONFIRMED] violence (CRITICAL) — VIOLENCE SUSPECTED | confidence=0.72 | ...
Threat event saved to: runs/detect/event_YYYYMMDD_HHMMSS_000
```

Each saved event folder contains `clip.mp4`, `frame.jpg`, and `metadata.json`.

### With the VideoMAE layer

```bash
python3 -m cvti.cli.detect \
  --source data/test_clips/violence_suspected.mp4 \
  --config configs/hybrid_video_action_v1.json \
  --video-action-backend videomae \
  --video-action-window-seconds 4 \
  --video-action-frames 16 \
  --gate-provider ollama
```

### With a real VLM gate

```bash
export OPENROUTER_API_KEY=sk-...
python3 -m cvti.cli.detect \
  --source <clip-or-rtsp> \
  --config configs/all_threats_v1.json \
  --gate-provider openrouter          # uses google/gemma-4-26b-a4b-it:free
```

### Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `0` | Camera index, RTSP URL, or video file |
| `--config` | — | Customization rules JSON (no config → no alerts) |
| `--concealment` | off | Enable pose-based concealment detector |
| `--zones` | off | Enable restricted-zone / dwell detection |
| `--max-frames` | 0 (all) | Stop after N frames (useful for testing) |
| `--gate-provider` | `ollama` | `ollama` / `local` (on-device) · `anthropic` / `openrouter` (cloud) · `mock` (refused unless `ARGUS_ALLOW_MOCK_GATE=1`) |
| `--video-action-backend` | `none` | `none` / `videomae` / `x3d` |
| `--video-action-window-seconds` | `4.0` | Seconds of video analyzed around a triggered moment |
| `--video-action-frames` | `16` | Frames sampled from that window |
| `--video-action-cooldown` | `2.0` | Min seconds between video-action runs |

> **Provider note.** `cvti-detect` exposes `mock / anthropic / openrouter`. The
> `VerificationGate` class *also* supports `ollama` and `local` (on-device
> OpenAI-compatible endpoints); those are reachable via the API and the retail
> pipeline. See §7.

### Window tuning (video action)

Keep 16 frames; experiment with the window first:

- `--video-action-window-seconds 2` — tighter motion, better for fighting/stabbing
- `--video-action-window-seconds 4` — balanced default
- `--video-action-window-seconds 6` — more context, but motion becomes sparse

---

## 5. Customization — the heart of the tool

The `CustomizationEngine` is the single, config-driven decision layer. *Every*
detector — weapons, violence, theft, concealment, zones, and `video_action` —
emits into one `RawEvent` stream, and your JSON decides which events matter.
**One JSON = one deployment/vertical.** No code changes to add or drop a
capability per site.

### Rule shape

```json
{
  "name": "shoplifting",
  "trigger":        { "detector": "concealment", "state": "" },
  "time_filter":    "18:00-06:00",
  "context_filter": "adjusted_confidence >= 0.3",
  "priority":       "high"
}
```

| Field | What it matches |
|---|---|
| `trigger.detector` | which detector fired: `weapons`, `violence`, `theft`, `concealment`, `zones`, `video_action` |
| `trigger.state` | optional detector sub-state (e.g. theft `DEPART`) |
| `time_filter` | `HH:MM-HH:MM` window (e.g. after-hours only) |
| `context_filter` | expression over the event's `extra` (e.g. `signal_type == 'violence_candidate' and adjusted_confidence >= 0.02`) |
| `priority` | `low` / `medium` / `high` / `critical` |

Only events matching a rule become `CandidateAlert`s; the gate then verifies the
top one.

### Shipped configs

- [`configs/all_threats_v1.json`](../configs/all_threats_v1.json) — one rule per
  detector (weapons, violence, theft, concealment). Good full-system baseline.
- [`configs/hybrid_video_action_v1.json`](../configs/hybrid_video_action_v1.json)
  — shows how to consume the weak `video_action` signal via `context_filter`.

### Example: per-vertical configs

- **Jewelry store** → `weapons` (critical) + `concealment` (high)
- **Bar / venue** → `violence` (critical) + hybrid `video_action` violence candidate (medium)
- **Warehouse** → `zones` with a `time_filter` of after-hours only

---

## 6. Testing

```bash
python3 -m pytest tests/ -q          # 33 tests, ~1s (no GPU/network needed)
```

Coverage includes: video-action model sampling & windows, hybrid
label→event mapping, runtime, verification-gate artifacts, concealment, retail
zones, zone customization.

**End-to-end run** (the real proof) — see §4 quick smoke test. It exercises
detection → rules → gate → saved event on a bundled clip.

---

## 7. The Verification Gate — providers

The gate is only called when a rule fires (cost-controlled: ~$0.001/alert, not
per frame). Supported providers in `VerificationGate`:

| Provider | Default model | Key env | Notes |
|---|---|---|---|
| `mock` | — | — | Auto-confirms; for testing only |
| `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | Claude Vision |
| `openrouter` | `google/gemma-4-26b-a4b-it:free` | `OPENROUTER_API_KEY` | Gemma / other vision models |
| `ollama` | `gemma3:4b` | `OLLAMA_API_KEY` | Local Ollama server |
| `local` | `gemma3:4b-it-qat` | — | On-device OpenAI-compatible endpoint (`http://localhost:11434/v1`) |
| `openai_compatible` | `gpt-4.1-mini` | `ANTHROPIC_API_KEY`→override | Any OpenAI-compatible cloud, via `base_url` |

Features: multi-frame verification (send the N clearest frames), optional
chain-of-thought prompting (`cot=True`), scene-context priming.

See [`docs/GATE_MODEL_BAKEOFF.md`](GATE_MODEL_BAKEOFF.md) for model comparison
results.

---

## 8. Current status & limitations

- ✅ Core pipeline (detection → customization → gate → save) runs on CPU, verified end-to-end.
- ✅ Retail pipeline, zones, concealment integrated into the same event stream.
- ✅ VideoMAE video-action layer merged and unit-tested; requires
  `requirements-video.txt` and is opt-in via `--video-action-backend`.
- ⚠️ VideoMAE labels are **weak** by design — always gate them behind the VLM
  before alerting.
- ⚠️ CPU inference on full clips is slow; use `--max-frames` for quick checks,
  or a GPU/`--video-action-device` for real workloads.
- ✅ *(resolved 19 Aug 2026)* `cvti-detect --gate-provider` now offers every
  provider the gate supports, and defaults to on-device `ollama`. It previously
  offered only `mock/anthropic/openrouter`, which meant the only offline option
  was the one that confirms everything without looking.

---

## 9. Repository map

```
cvti/
  detector/core.py          # main loop + per-frame detectors  (cvti-detect)
  retail/{concealment,zones}.py
  video_action_{model,hybrid,runtime}.py
  rules/customization.py    # the rules engine
  verification/gate.py      # VLM gate
  event_adapters.py         # detector state → RawEvent
  contracts.py              # shared dataclasses
  pipelines/retail_pipeline.py   # cvti-retail
  cli/                      # thin entry points
configs/                    # customization JSONs
data/test_clips/            # sample videos
tools/                      # probes & bake-off harnesses
tests/                      # 33 unit tests
docs/                       # this guide, project context, bake-off
```
