# CV Threat Intelligence Backend V1 Plan

## Overview

This project is a context-aware AI security intelligence layer for existing CCTV, IP camera, RTSP, and video-file feeds. The product goal is not simply to detect one class like violence or shoplifting. The goal is to help customers define what a "threat" means in their own environment, run local computer vision continuously, verify candidate alerts with context, and notify stakeholders with evidence when a real threat is occurring.

The product thesis is:

```text
Threat meaning is customer-specific.
The customer defines threat policy.
The backend detects possible events.
The verification layer confirms whether the event matches the configured threat.
The feedback loop makes the system sharper per camera, per site, and per customer.
```

The current GTM direction is property/security operators first: premium gated estates, malls, offices, banks, and adjacent commercial real-estate environments. Retail shoplifting remains a useful technical proving ground because it tests temporal action understanding, ambiguity, and false-positive control.

## Architecture We Are Building Toward

```text
Camera feed
  -> Agent Mapper
       descriptive scene context per camera
       environment type, expected actors, zones
  -> Detection Core
       cheap local signals every frame or short clip
       people, objects, pose, zones, dwell, motion, weapons, concealment, violence
  -> Event Adapters
       convert raw detector outputs into normalized RawEvents
  -> Customization Engine
       applies user_config.json threat policy
       returns CandidateAlerts
  -> Verification Gate
       local/cloud VLM or later specialist verifier
       confirms/rejects candidate alerts using scene context
  -> Alert/Evidence Layer
       saves clips, frames, JSON, model verdicts, human feedback labels
```

The important boundary: Agent Mapper describes the world; it does not decide threat policy. Customization Engine applies the user's definition of threat. Detection Core produces possible signals. Verification Gate confirms a specific candidate alert.

## What Exists Today

### Detection Core

`detector.py` supports:

- webcam, RTSP, and video-file sources
- YOLO/RT-DETR-style object detection through Ultralytics-compatible paths
- legacy YOLOv5 weapon checkpoint loading via `--weapon-loader yolov5`
- person/weapon model separation
- pose extraction with YOLO pose
- ByteTrack-style tracking fallback controls
- rule-based weapon/armed-person/assault assessment
- pose-based violence heuristics
- theft state machine
- optional classifier override through `--classifier-weights`
- optional retail zones through `--zones`
- optional concealment detector through `--concealment`
- `--mode {all,theft,violence,weapons}` for narrowed experiments

### Customization and Rule Layer

`customization.py` supports:

- `RawEvent` and `CandidateAlert` dataclasses
- loading `user_config.json` style rule files
- matching on detector, state, level, time windows, and simple context filters
- converting weapons, violence, theft, zone presence, and concealment outputs into RawEvents
- sorting alerts by priority

Important current limitation: `CustomizationEngine.evaluate()` returns all matching alerts, but runtime paths often select only `candidate_alerts[0]`.

### Scene Context / Agent Mapper

`agent_mapper.py` supports:

- image, video, webcam, and RTSP sampling
- representative-frame selection
- mock, Anthropic, and OpenAI-compatible providers
- local Ollama-compatible calls through the OpenAI-compatible endpoint
- schema-shaped scene context output
- descriptive-only mapper behavior: environment, description, expected actors, zones

The mapper should seed per-camera context and later help generate preset suggestions. It should not list risks or decide threats.

### Retail/Zone/Concealment Modules

`retail_zones.py` supports:

- polygon zones
- per-track zone membership
- dwell-time accumulation
- sticky dwell across brief track drops
- person plausibility filtering to reduce mannequin/display false positives
- zone annotations

`concealment.py` supports:

- pose-sequence concealment heuristic
- waist/pocket destination
- personal-bag destination
- explicit trolley/basket-safe behavior
- per-track rolling state
- synthetic tests
- a seam for replacing heuristic scoring with a trained temporal classifier later

### Verification Gate

`verification_gate.py` supports:

- mock, Anthropic, OpenRouter, and local Ollama providers
- single-frame or multi-frame verification
- optional CoT-style prompt
- rule-specific questions for shoplifting, violence, weapon sighting, robbery, banking, and similar alerts
- artifact saving: `alert.json`, `verification.json`, `raw_response.txt`, and sent frames
- stale frame cleanup after the 2026-07-04 artifact-leak fix

### Evaluation and Experiments

`tools/gate_bakeoff.py` supports:

- local model comparison through Ollama
- rule-aware labels: concealment, violence, weapon
- frame selection by detector score
- `--gate-frames` multi-frame spans
- `--event-moment` motion-peak anchoring
- `--crop` suspect/interaction crop, now treated as an A/B toggle because it hurt some runs
- `--use-agent-mapper`
- `--cot`
- per-model recall, specificity, accuracy, JSON rate, latency, and per-rule accuracy

`tools/batch_anomaly.py` supports:

- running unified signals over `data/anomaly`
- recall-style coverage across weapon, violence, theft, and concealment

`train_classifier.py` currently trains a frame classifier, not a true temporal video model.

### Data Present in the Repo

- `data/test_clips/`: small curated demo/eval clips
- `data/anomaly/`: robbery/anomaly clips
- `data/ground_truth.csv`: existing labels
- `download_training_data.py`: YouTube downloader for theft/violence/normal bootstrap data
- `runs/detect/*`: saved event artifacts that can become feedback-loop data after human review
- `runs/bakeoff_frames*`: gate/eval artifacts

## What Is Ticked

- Context-aware architecture is clear.
- Existing-camera input path exists.
- Local/edge-first direction is practical.
- Agent Mapper exists as a descriptive context layer.
- Customization Engine exists as the customer threat-policy layer.
- Detection Core can run multiple primitive signals.
- Zones and dwell are implemented and tested.
- Concealment candidate generation is implemented and tested.
- VLM gate is implemented and can run local Ollama models.
- Gate bakeoff tooling exists and exposed real behavior differences between Gemma 3 and Gemma 4.
- Artifact leakage bug in gate frame saving was identified and fixed.
- A first 9-clip quick eval file exists: `tools/gate_bakeoff_labels_9.json`.
- Ayo's branch proves a package/app/offline-Ollama direction is viable, though it does not include all ML experimentation features from `unify-detector`.

## What Is Missing

### Backend Intelligence Gaps

- The backend is still too narrow in practice. It mostly evaluates weapons, violence, theft, zones, and concealment.
- Robbery is not a first-class compound event. It may only be caught if weapon, violence, or panic-like signals fire.
- The system does not yet implement the GTM 12-rule V1 set end to end.
- There is no always-on critical safety baseline for events like visible weapons, person down, fire/smoke, and camera tampering independent of customer-specific rules.
- Multiple simultaneous alerts are not handled properly. The runtime often chooses only the highest-priority/top candidate.
- The Verification Gate in `detector.py` still receives the current frame, not a curated rolling event clip.
- Per-rule gate configuration is not implemented in runtime. Experiments show motion threats and object threats need different frame counts/settings.
- Weapon/gun detection is weak on low-resolution CCTV, small objects, blur, and occlusion.
- Current classifier is frame-based and cannot truly understand temporal action.

### Product Rule Gaps

The GTM V1 wants 12 rules:

1. Loitering detection
2. Perimeter intrusion / fence climbing
3. After-hours presence
4. Crowd formation
5. Running detection
6. Tailgating
7. Abandoned object
8. Unauthorized vehicle / wrong-way movement
9. Camera tampering / obstruction
10. Person down / fall detection
11. Mask / face-covering during business hours
12. Power-outage + motion combo

Currently, only parts of this are present:

- Loitering: partially via zones/dwell
- After-hours presence: partially via zone/time configs
- Running: primitive motion possible, not productized
- Weapon/violence: present but brittle
- Concealment/shoplifting: present as retail-specific candidate generator

Not yet productized:

- perimeter intrusion/fence climbing
- crowd formation
- tailgating
- abandoned object
- unauthorized vehicle/wrong-way movement
- camera tampering/obstruction
- person down/fall
- mask/face-covering during business hours
- power-outage + motion combo

### Evaluation Gaps

- No validated FPR yet.
- Current labeled sets are too small and skew positive.
- Need hard negatives per rule.
- Need normal retail/estate/office footage.
- Need per-rule, per-camera evaluation.
- Need final metrics split into candidate-generator recall, gate precision, final recall, final FPR, latency, and model size.

### Data/Training Gaps

- Existing data is not enough for a reliable video model.
- Online/western CCTV data can bootstrap, but will not fully capture Nigerian/security-property context.
- No true video-model fine-tuning pipeline exists yet.
- No reviewed feedback loop exists yet. Event artifacts exist, but they need human labels before training.

## Edge Cases and Failure Modes Discussed

### Multiple Simultaneous Threats

Example: shoplifting occurs in one aisle while armed robbery starts at the entrance. The desired backend should emit and verify multiple concurrent alerts. Current runtime generally selects one top alert, so this is not solved yet.

### Narrow User Config Hiding Critical Threats

If a user config only enables violence, the system may ignore concealment or weapons at the policy layer even if the detectors saw them. V1 should separate:

```text
customer-specific threat policy
always-on critical safety baseline
```

Always-on baseline should include at least weapon/armed robbery, serious violence, person down, fire/smoke, and camera tampering.

### Robbery as Compound Event

Robbery should not rely only on gun detection. Low-resolution cameras often miss guns. Robbery should be a threat recipe combining signals such as weapon candidate, masked entry, counter rush, violence, running/panic, person down, and crowd dispersal.

### VLM Frame Blindness

The VLM sees still frames, not continuous video. Multi-frame helps some motion threats but can hurt object threats. Frame selection matters. Cropping can help focus but can also remove context and reduce accuracy, as observed in Gemma 3/Gemma 4 tests.

### Artifact Leakage

Older multi-frame artifacts were left in reused `gate_XXXX` folders and made later single-frame runs look mixed. This was fixed by deleting stale `frame*.jpg` before saving new gate artifacts.

### Low-Resolution Weapon Detection

Small weapons in CCTV are hard. If the pixels are poor, neither object models nor VLMs can reliably infer the weapon. Robbery detection should use compound signals and better temporal/context cues, not gun detection alone.

### Track Fragmentation

ByteTrack ID fragmentation can break dwell, concealment windows, and per-person event grouping. Sticky dwell helps, but stronger tracking/reassociation may be needed.

### Trolley vs Bag

Putting goods into a trolley/basket is normal shopping. Putting goods into a personal bag/pocket is suspicious. `concealment.py` currently models this distinction by treating backpack/handbag/suitcase as concealment destinations and not treating trolley/basket as a personal bag.

### Agent Mapper Scope

Agent Mapper must stay descriptive. If it starts making threat judgments, it weakens the product principle that the customer defines threats.

## Recommended V1 Definition

Forget WhatsApp/UI for backend readiness. The minimum viable backend V1 should be:

```text
For one property-security preset, the backend can run locally on video/RTSP,
produce multiple candidate alerts, verify each important candidate with local VLM or rule-specific verifier,
save evidence and machine-readable artifacts, and report recall/FPR on a labeled test set.
```

The first V1 preset should be either:

### Option A: Estate Guard First

Best GTM alignment with the real-estate recommendation.

Minimum rules:

- after-hours presence in restricted/perimeter zones
- loitering at gate/perimeter
- perimeter intrusion/fence-climbing candidate
- weapon/armed-robbery critical baseline
- person down/fall candidate
- camera tampering/obstruction

### Option B: Retail Watch First

Best technical continuity with existing code and clips.

Minimum rules:

- concealment/shoplifting
- shelf loitering
- weapon/armed-robbery critical baseline
- violence/assault
- person down/fall candidate
- crowd/running/panic candidate

Recommendation: use Retail Watch for short-term model experimentation because the repo already has retail clips and concealment code, but define backend V1 around Estate/Retail-compatible primitives so the product does not become "just shoplifting."

## Roadmap

### Phase 0: Stabilize Current Ground Truth

1. Keep `unify-detector` as the ML experimentation branch.
2. Preserve Ayo's package/app branch as the productionization branch.
3. Create a single source of truth for backend status in `plan.md` and `PROJECT_CONTEXT.md`.
4. Run current tests:
   - `python tests/test_concealment.py`
   - `python tests/test_retail_zones.py`
   - `python tests/test_zone_customization.py`
   - `python tests/test_verification_gate_artifacts.py`

### Phase 1: Fix Multi-Threat Backend Semantics

Goal: move from "top alert" to "alert queue."

Tasks:

- Replace `top_alert = candidate_alerts[0]` runtime behavior with a throttled queue of all important candidate alerts.
- Add per-alert signature using rule, detector, person_id, object_label, zone, and time bucket.
- Verify multiple concurrent candidates without flooding the VLM.
- Save one artifact bundle per verified candidate.
- Add tests proving two simultaneous RawEvents create two CandidateAlerts and both can be processed.

### Phase 2: Add Always-On Critical Baseline

Goal: customer-specific rules should not hide universal critical threats.

Tasks:

- Create `configs/baseline_critical_v1.json`.
- Include weapon/armed robbery, violence/assault, person down/fall, camera tampering, fire/smoke once implemented.
- Merge baseline rules with customer config at runtime.
- Allow customer severity tuning later, but do not allow disabling critical safety rules in early pilots without an explicit operator override.

### Phase 3: Formalize Threat Recipes

Goal: move from one detector per threat to compound threat definitions.

Example:

```json
{
  "name": "armed_robbery",
  "priority": "critical",
  "signals": ["weapon_candidate", "violence", "masked_entry", "counter_rush", "running", "person_down"],
  "logic": "one_high_or_two_medium",
  "gate_question": "Do these frames show an armed robbery or coercive threat in progress?"
}
```

Tasks:

- Extend RawEvent with confidence, zone, camera_id, track_id/person_id, and signal_type.
- Add compound rule evaluator.
- Build first compound recipes for armed robbery, shoplifting, loitering, and after-hours intrusion.

### Phase 4: Per-Rule Frame/Clip Selection

Goal: stop using one global gate strategy.

Rule-specific defaults:

- weapon: single best full frame, no CoT by default
- violence: 3-4 frames over motion peak
- concealment: 3-4 frames over reach/retract moment, full frame first, crop only as A/B
- robbery: 4-6 frames around compound event
- zone/time rules: one frame plus event metadata may be enough

Tasks:

- Port `tools/gate_bakeoff.py` frame-selection logic into runtime utility code.
- Keep `--crop` experimental, not default.
- Save selected frame indices and strategy in artifacts.

### Phase 5: Build a Real Evaluation Set

Goal: validated FPR before claiming V1.

Dataset buckets:

- positives per rule
- normal negatives
- hard negatives per rule
- low-resolution CCTV clips
- Nigerian/customer-collected clips as soon as pilots start

Metrics:

- candidate recall
- gate precision
- final recall
- final FPR
- latency
- memory/model size
- per-rule confusion matrix

### Phase 6: Fine-Tune Video Models

Goal: make temporal threats smarter than VLM frame guessing.

Training strategy:

- Use online/public CCTV data now for bootstrapping.
- Treat online/western data as pretraining/initial fine-tune, not final truth.
- Use human-reviewed pilot events later for active learning and supervised fine-tuning.
- Do not call this reinforcement learning yet. The practical loop is:

```text
candidate event -> human review -> labeled positive/negative/hard-negative -> periodic supervised fine-tune
```

First video-model targets:

1. concealment/no-concealment, because it is Veesion-like and already has a heuristic seam
2. violence/assault/no-violence, because current heuristics are brittle
3. person-down/fall, because it is high-value for property-security V1

Recommended first experiment:

- Fine-tune a small pretrained video model on 100-300 labeled clips.
- Use cloud GPU for serious training.
- Keep local machine for data prep and small smoke tests.
- Compare trained video model against current heuristic + VLM gate.

Current implementation checkpoint:

- `video_action_model.py` provides VideoMAE and X3D wrappers plus frame-window helpers.
- `tools/video_action_probe.py` can test a clip with:
  - `--window-mode single`;
  - `--window-mode segments`;
  - `--window-mode event`;
  - `--save-frames-dir` to inspect exactly what the model saw.
- `video_action_hybrid.py` converts useful video-model labels into weak `RawEvent`s.
- `video_action_runtime.py` keeps a rolling frame buffer for live detection.
- `detector.py` can now run VideoMAE around detector-triggered moments using `--video-action-backend videomae`.

The hybrid runtime path is:

```text
YOLO / pose / theft / concealment detects suspicious frame
-> current frame becomes event center
-> VideoMAE samples 16 frames around that moment
-> VideoMAE emits weak temporal action labels
-> useful labels become down-weighted RawEvents
-> CustomizationEngine applies customer config
-> VerificationGate/VLM confirms or rejects
```

Current recommendation:

- Use VideoMAE first, not X3D, because VideoMAE produced more useful weak violence signals in early tests.
- Keep VideoMAE optional and off by default until latency/FPR are measured.
- Use VideoMAE as a weak temporal witness, never as the final alert judge.
- Start with violence/person-down/motion threats before shoplifting, because pretrained VideoMAE does not understand concealment directly.

Manual smoke command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/private/tmp ./.venv/bin/python detector.py \
  --source data/test_clips/violence_suspected.mp4 \
  --config configs/hybrid_video_action_v1.json \
  --video-action-backend videomae \
  --video-action-window-seconds 2 \
  --video-action-cooldown 0 \
  --max-frames 7 \
  --gate-provider mock \
  --no-track \
  --show-all-detections
```

Validated behavior:

```text
[VideoAction] ... top=punching person (boxing)
[CONFIRMED] video_action_violence_candidate (MEDIUM)
```

### Phase 7: Feedback Loop and Model Registry

Goal: turn pilots into the moat.

Every alert artifact should support:

- machine verdict
- human verdict: true threat, false alarm, missed threat, ambiguous
- rule name
- camera_id
- environment_type
- zone
- source model versions
- selected frames/clip
- notes

The registry should track:

- dataset version
- model version
- metrics including FPR
- deployment target
- date trained
- rules supported

No model should ship without a validated FPR attached.

### Phase 8: Serving & Throughput (multi-stream, near real-time)

Goal: run many camera streams on one box in near real-time. Phases 0–7 cover
*what* to detect; this phase covers *how to serve it at scale*, which nothing
above addresses.

Target (confirmed 2026-07-13): **one edge box per site, NVIDIA GPU, 5–16
cameras, 1–2s glass-to-alert latency.**

Key insights that shape the design:

- The 1–2s budget means we do **not** need 30 FPS/camera. Detection at ~5 FPS
  is enough (threats and dwell/zone events are multi-second). 16 cams × 5 =
  ~80 detection-frames/s total.
- **Detection is not the bottleneck.** Baseline bench (yolov8n @640, MPS dev
  box, fp16 off) already sustains ~180 fps batched → ~36 cams at 5 FPS. A real
  NVIDIA GPU with fp16/TensorRT is multiples faster.
- **The local VLM gate is the true ceiling** (~1–3s per verify, ~single-thread
  on one GPU). Across all 16 cams combined you can afford roughly one verified
  alert every 1–3s. You cannot gate every candidate — so Phase 1 (alert
  queue + dedup) and Phase 4 (per-rule frames) become load-bearing for scale.
- **Batch the stateless detector across cameras; keep the stateful tracker and
  rules per camera.** ByteTrack holds per-camera state, so batch detection then
  associate per stream.

Target architecture (one process, shared models for a camera group):

```text
16 RTSP decode threads ─┐  keep LATEST frame only (drop stale) + skip to ~5 FPS
                        ├─▶ BATCHER (~150ms window) ─▶ YOLO.predict(batch, half=True)
                        │        └─▶ scatter detections back per camera
                        ├─▶ per-camera: ByteTrack assoc + pose(batched) + zones/rules
                        ├─▶ ALERT QUEUE (Phase 1): dedup (camera,rule,track,time-bucket) + throttle
                        └─▶ async LOCAL-VLM pool (1–2 workers, rate-limited) ─▶ evidence writer
```

Sub-phases (sequenced so the hot loop is rewritten once):

- **8.0 Measure** — `tools/throughput_bench.py`: per-model ms/frame at batch
  1/8/16 (+fp16) and a multi-stream capacity projection. Run on the target GPU.
- **8.1 Pipeline refactor** — decode threads (latest-frame-drop) → batcher →
  batched detect → per-camera track/rules; single process, shared models.
  **Fold Phase 1 (top-alert → alert queue) in here.**
- **8.2 Frame governor** — per-camera target-FPS subsampling.
- **8.3 Async gate + baseline-critical** — non-blocking VLM pool, hard dedup,
  prioritize Phase 2 baseline-critical over customer rules. Tames the VLM ceiling.
- **8.4 GPU optimize** — export YOLO→TensorRT (fp16), tune imgsz; then NVDEC decode.
- **8.5 Resilience** — per-camera supervision/reconnect (one dead RTSP feed must
  not kill the box).

Interlock with earlier phases: 8.1 *is where* Phase 1 lands (same loop). 8.3
depends on Phase 2 and consumes Phase 4. Compound recipes (Phase 3), eval
(Phase 5), and video fine-tuning (Phase 6) run in parallel, unaffected.

## Minimum Bar To Call This Backend V1

Backend V1 is ready when:

- One preset is selected as the first pilot preset.
- At least 5 rules work end to end for that preset.
- Always-on critical baseline exists.
- Multiple simultaneous candidate alerts can be queued, verified, and saved.
- Agent Mapper context is consumed by gate/rules where relevant.
- Per-rule gate frame selection is implemented.
- Local VLM gate runs with the selected edge model.
- A labeled eval set exists with positives, normals, and hard negatives.
- Metrics are reported per rule: recall, FPR, latency, model size.
- Evidence artifacts include enough metadata for human review and future training.

## Ayo Branch Technical Comparison

Compared branch: `ayo/main` at `4364ba4`.

Common base with current `unify-detector`: `413da78`.

### Features Present in Current `unify-detector` But Not Ayo's Branch

- Unified general `detector.py` has optional `--zones`.
- Unified general `detector.py` has optional `--concealment`.
- General detector can merge weapons, violence, theft, zones, and concealment into the same event stream.
- `verification_gate.py` has provider name `ollama`.
- `verification_gate.py` accepts a list of frames for multi-frame verification.
- `verification_gate.py` supports CoT prompt mode.
- `verification_gate.py` supports OpenRouter provider.
- `tools/gate_bakeoff.py` exists.
- `tools/gate_bakeoff.py` supports `--gate-frames`, `--cot`, `--use-agent-mapper`, `--crop`, and `--event-moment`.
- `tools/batch_anomaly.py` exists.
- `configs/all_threats_v1.json` exists.
- `docs/GATE_MODEL_BAKEOFF.md` exists on this branch.

### Features Present in Ayo's Branch But Not Current `unify-detector`

- Installable `cvti` package structure.
- Desktop app under `cvti/app`.
- `cvti/verification/ollama.py` operational helpers: server check, model check, pull model, bundled binary handling.
- `local` verification provider with default `gemma3:4b-it-qat`.
- Build scripts and PyInstaller spec.
- Docs for offline VLM, user guide, and software wiring.

### Important Technical Differences

- Ayo's `cvti/detector/core.py` does not include the unified detector's `--zones` or `--concealment` flags.
- Ayo's general detector gate supports `mock` and `anthropic`, not the full local/Ollama/OpenRouter path in this branch's general detector.
- Ayo's `cvti/verification/gate.py` is single-frame only. It does not accept multi-frame lists.
- Ayo's gate does not include CoT mode.
- Ayo's branch does not include the gate bakeoff harness, event-moment frame selection, crop A/B toggle, or local model comparison tooling.
- Both branches still use a top-alert runtime pattern in key paths: `alerts[0]` or `candidate_alerts[0]`.

### Practical Conclusion

Ayo's branch is better for packaging and offline app operations. This branch is better for ML/backend intelligence experiments and unified detector work. The eventual merge should port the ML features from `unify-detector` into Ayo's `cvti` package structure, not discard them.
