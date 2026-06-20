# CV Threat Intelligence

AI-powered threat detection pipeline for live camera systems.  
Built for real-world deployment in Nigeria and similar markets — proactive surveillance, not passive recording.

---

## Table of Contents

- [What This Is](#what-this-is)
- [Current Pipeline Status](#current-pipeline-status)
- [Quick Start](#quick-start)
- [Running the Detector](#running-the-detector)
- [Threat States Explained](#threat-states-explained)
- [False Positive Problem — Root Cause Analysis](#false-positive-problem--root-cause-analysis)
- [State-of-the-Art Model Recommendations](#state-of-the-art-model-recommendations)
- [Architecture: Online vs Edge](#architecture-online-vs-edge)
- [2-Day Demo Plan](#2-day-demo-plan)
- [Phased Roadmap](#phased-roadmap)
- [Evidence Output](#evidence-output)
- [Open Questions](#open-questions)

---

## What This Is

Traditional CCTV records footage. A human reviews it later.

This system makes cameras proactive:
- monitors video feeds automatically
- detects suspicious or dangerous events in near real time
- produces alerts immediately
- saves evidence clips for review

Current focus: a live demo proving the concept is worth building further.  
Long-term vision: a surveillance intelligence layer that sits on top of existing camera infrastructure, purpose-built for Nigeria and similar markets.

---

## Current Pipeline Status

The pipeline is working. The weak link is the weapon model, not the code.

| Layer | Status | Notes |
|---|---|---|
| Camera ingestion (webcam, RTSP, file) | Working | Stable across sources |
| Person detection (YOLOv8n) | Working | Reliable in tested scenes |
| Weapon detection (YOLOv5 knife/gun) | Working but noisy | Fires on background objects |
| Pose-based violence heuristics | Working | Rule-based, not learned |
| Threat assessment engine | Working | Heuristic, not model-based |
| Evidence recording | Working | Annotated frames + clips saved |

What it can do right now:
- detect a visible weapon when the checkpoint recognizes it
- detect people reliably
- infer `ARMED PERSON` when a weapon appears spatially attached to a person
- infer `POSSIBLE ASSAULT` when an armed person is close to another person
- save annotated evidence clips per threat event

What it cannot do yet:
- reliably detect fighting or general violence without a weapon present
- distinguish a kitchen knife from a threat knife
- understand scene context (home vs warehouse vs street)

---

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r external/yolov5/requirements.txt
```

Smoke test with webcam:

```powershell
python detector.py --source 0 --weights yolov8n.pt --threat-classes person --show
```

---

## Running the Detector

### Person only (pipeline smoke test)

```powershell
python detector.py --source 0 --weights yolov8n.pt --threat-classes person --show
```

### Dual-model: person + weapons

```powershell
python detector.py \
  --source 0 \
  --weights yolov8n.pt \
  --person-weights yolov8n.pt \
  --weapon-weights "models\weapon_best.pt" \
  --weapon-loader yolov5 \
  --person-classes person \
  --weapon-classes knife,gun \
  --threat-classes knife,gun \
  --show
```

### With violence heuristics + strict tuning (recommended for demo)

```powershell
python detector.py \
  --source 0 \
  --weights yolov8n.pt \
  --person-weights yolov8n.pt \
  --weapon-weights "models\weapon_best.pt" \
  --weapon-loader yolov5 \
  --pose-weights yolov8n-pose.pt \
  --person-classes person \
  --weapon-classes knife,gun \
  --threat-classes knife,gun \
  --weapon-conf 0.80 \
  --min-threat-frames 3 \
  --violence-min-frames 4 \
  --debug-weapon \
  --debug-violence \
  --show
```

### RTSP stream

```powershell
python detector.py \
  --source "rtsp://username:password@camera-ip:554/stream" \
  --weights yolov8n.pt \
  --threat-classes person \
  --show
```

### Video file (safe fallback for demo)

```powershell
python detector.py \
  --source "demo.mp4" \
  --weapon-weights "models\weapon_best.pt" \
  --weapon-loader yolov5 \
  --weapon-conf 0.80 \
  --min-threat-frames 3 \
  --show
```

### Key tuning flags

| Flag | Purpose |
|---|---|
| `--weapon-conf 0.80` | Raise this to reduce weapon false positives |
| `--min-threat-frames 3` | Require N consecutive frames before raising a threat |
| `--violence-min-frames 4` | Same gate for violence heuristics |
| `--debug-weapon` | Print exact weapon detections and confidence live |
| `--debug-violence` | Print pose-based violence reasoning live |
| `--show-all-detections` | Draw all raw detections including irrelevant classes |
| `--weapon-min-area-ratio` | Reject weapon boxes that are too small |
| `--weapon-max-area-ratio` | Reject weapon boxes that are too large |
| `--weapon-border-margin-ratio` | Reject weapons hugging the frame edge |

---

## Threat States Explained

| State | What it means |
|---|---|
| `DANGEROUS OBJECT` | A weapon-class object is detected |
| `ARMED PERSON` | A weapon is spatially attached to a detected person |
| `POSSIBLE ASSAULT` | An armed person is within proximity of another person |
| `VERIFYING THREAT` | Detection is accumulating frames before confirming |
| `VIOLENCE SUSPECTED` | Pose heuristics: high wrist speed + close persons |
| `POSSIBLE STABBING` | Weapon near wrist + arm extension + close proximity |
| `POSSIBLE ARMED ASSAULT` | Armed person + aggressive pose + proximity |

All states above `VERIFYING` are heuristic. They are not learned action-recognition outputs.

---

## False Positive Problem — Root Cause Analysis

The current false positives are not a tuning problem. They are a structural problem with three causes.

### 1. The weapon checkpoint is weak

`models/weapon_best.pt` is a hobbyist YOLOv5 model trained on limited data.  
It fires on door edges, frame borders, bed sheets, and anything roughly knife/gun-shaped.  
Raising `--weapon-conf` helps, but does not fix the underlying model quality.

### 2. No temporal or contextual understanding

Even with `--min-threat-frames`, the model sees a single bounding box.  
It cannot understand:
- is the object being held by someone?
- is the person moving aggressively?
- is this a kitchen counter or a street corner?

A knife on a table and a knife in an attacker's hand look identical to it.

### 3. No scene context

The same logic applies to a living room, a warehouse, and a commercial street.  
The environment is completely ignored, so the false positive rate multiplies by scene noise.

### The fix — in priority order

1. Replace the weapon checkpoint with one trained on [Open Images v7](https://storage.googleapis.com/openimages/web/index.html) — it has 10,000+ labeled `Knife`, `Gun`, `Handgun` examples with diversity the current checkpoint lacks.
2. Add **Grounding DINO** as a second-opinion gate. Query it with `"person holding a knife"` not just `"knife"`. This kills contextless detections structurally.
3. Add **ByteTrack or BoT-SORT** tracking. A weapon that persists across 15+ frames on the same tracked person ID is structurally different from a one-frame hit on a background object.
4. Replace pose heuristics with a real video action model (see next section).
5. Collect hard negatives from the actual deployment environment and add them to training.

---

## State-of-the-Art Model Recommendations

No lock-in to YOLOv8. These are the current SOTA options per task.

### Object Detection

| Model | Strengths | Best Use |
|---|---|---|
| **Grounding DINO** | Open-vocabulary, text-prompted, zero-shot | Second-opinion verification gate, eliminates contextless FPs |
| **RT-DETR** (Baidu) | Transformer-based, no NMS, better spatial reasoning | Drop-in YOLOv8 replacement, better at context |
| **OWLv2** (Google) | CLIP backbone, natural language queries | Zero-shot weapon detection by description |
| **YOLOv9 / YOLOv10 / YOLO11** | Programmable gradient information, anchor-free | Fastest upgrade path from current setup |

**Recommended:** RT-DETR for detection backbone. Grounding DINO as the verification gate.

### Violence and Action Recognition

| Model | Strengths | Best Use |
|---|---|---|
| **VideoMAEv2** (Meta/FAIR) | SOTA on Kinetics-400/600/700, understands 16-frame temporal motion | Cloud violence detection, fine-tune on RWF-2000 dataset |
| **SlowFast** (FAIR) | Dual-path: spatial detail + rapid motion | Detecting sudden aggressive movements |
| **MViTv2** | Efficient video transformer, strong temporal reasoning | Cloud or high-end edge (Jetson AGX) |
| **X3D-M / X3D-S** | Very efficient 3D CNN, quantizable | Edge deployment on Jetson Orin NX, 15–25fps |

**Recommended:** VideoMAEv2 fine-tuned on RWF-2000 (real-world fighting) for cloud. X3D-S for Jetson edge.

The key insight: these models look at 8–16 frame clips natively, not single frames. They eliminate pose heuristic noise structurally.

### Pose Estimation

| Model | Strengths | Best Use |
|---|---|---|
| **RTMPose** (OpenMMLab) | Real-time, multi-person, 2–3x faster than YOLOv8-pose | Edge deployment, drop-in upgrade |
| **ViTPose** | Transformer-based, highest accuracy | Cloud use cases |
| **DWPose** | Distilled from ViTPose, balanced accuracy/speed | General use |

**Recommended:** RTMPose as the direct upgrade from YOLOv8n-pose.

### Two-Stage Verification Architecture (the real false positive fix)

```
Fast detector (RT-DETR)
        ↓  fires on candidate only
Grounding DINO or VLM verification
(query: "person holding knife" / "person pointing gun")
        ↓  confirmed only
Threat raised + evidence saved
```

This is the architectural pattern that separates research demos from production-grade systems.  
The VLM gate costs API calls but the fast detector pre-filters so only genuine candidates reach it.

---

## Architecture: Online vs Edge

### Online (Cloud/Server)

Best for: multi-camera city deployment, control room monitoring, warehouse fleets, high-value sites.

```
RTSP cameras (multiple)
        ↓
  Stream broker (Kafka / Redis Streams)
        ↓
  Ingest workers (per-camera, frame sampling 3–5fps)
        ↓
  Detection layer
    ├── RT-DETR or Grounding DINO  (objects + weapons)
    ├── VideoMAEv2 or SlowFast     (violence, 16-frame clips)
    └── RTMPose or ViTPose         (pose features)
        ↓
  Threat fusion engine
    ├── rule layer (armed person, proximity, temporal persistence)
    └── VLM verification gate (Claude / GPT-4V) for final confirmation
        ↓
  Alert service (WebSocket → dashboard, SMS, push notification)
        ↓
  Evidence store (S3 / GCS, annotated clips + metadata JSON)
        ↓
  Operator dashboard (live map, alert queue, review interface)
```

Inference hardware: T4 or L4 GPU handles 8–16 cameras at 5fps with RT-DETR-sized models.

### Edge (On-Premise / Offline-First)

Best for: motorcycle snatching detection, market stalls, warehouses, locations with poor internet, privacy-sensitive deployments.

```
IP Camera (RTSP, local)
        ↓
  Jetson Orin NX 16GB (~$500) or AGX Orin (~$899)
        ↓
  TensorRT INT8 optimized models:
    ├── YOLO11n or RT-DETR-tiny    (person + weapon)
    ├── X3D-S                      (action, 8-frame clips)
    └── RTMPose-tiny               (real-time pose)
        ↓
  Local threat fusion (fully offline capable)
        ↓
  Local alert: on-screen overlay, buzzer, LED, local storage
  + Buffered sync: push events to cloud when connectivity available
```

| Hardware | Cost | Cameras | Notes |
|---|---|---|---|
| Hailo-8 + Raspberry Pi 5 | ~$200 | 1–2 | Detection only, no action recognition at full quality |
| Jetson Orin NX 16GB | ~$500 | 2–3 | RT-DETR + RTMPose + X3D at 15fps per stream |
| Jetson AGX Orin | ~$899 | 4–6 | Full pipeline including VideoMAEv2-lite |

Nigeria-specific edge concerns:
- Power cuts: use a UPS + write-ahead event logging so a sudden power loss does not corrupt state
- Ambient heat: active cooling or ventilated enclosure required for sustained inference
- Offline-first: all alerting logic must work with zero internet, sync in batches when connectivity returns

---

## 2-Day Demo Plan

The goal is a convincing, stable, repeatable demo. Not a production system.

### Day 1 — Stabilize and prepare the demo video path

**Morning (3–4 hours): Get a better weapon video**

1. Find or record a clear knife or gun scenario on video. Options:
   - film yourself holding a kitchen knife clearly in front of the webcam at different angles
   - download a free stock video with a visible weapon (Pexels, Pixabay — search "knife threat" or "gun prop")
   - use the webcam live if you have a prop knife with high contrast

   The goal is to have a 30–60 second video clip where the weapon is clearly visible, not occluded, and not too small in frame.

2. Run the detector on that video with debug flags on:

```powershell
python detector.py \
  --source "your_test_video.mp4" \
  --weights yolov8n.pt \
  --person-weights yolov8n.pt \
  --weapon-weights "models\weapon_best.pt" \
  --weapon-loader yolov5 \
  --pose-weights yolov8n-pose.pt \
  --person-classes person \
  --weapon-classes knife,gun \
  --threat-classes knife,gun \
  --weapon-conf 0.55 \
  --min-threat-frames 2 \
  --violence-min-frames 3 \
  --debug-weapon \
  --show
```

3. Watch what the model actually fires on. Note the confidence scores printed by `--debug-weapon`.

**Afternoon (3–4 hours): Tune the demo to be clean**

4. If weapon detections are firing but noisy: raise `--weapon-conf` incrementally (0.55 → 0.65 → 0.75) until only real weapon frames trigger.

5. If weapon is not detected at all: lower `--weapon-conf` to 0.35, and make sure the prop/video has the weapon clearly visible and large in frame. Size matters — if the weapon is smaller than ~8% of the frame width, the current checkpoint will miss it consistently.

6. Once you have a video clip that produces clean `ARMED PERSON` detections: that is your demo asset. Lock the command that produces it.

7. Test the same command on webcam live:
   - hold a clearly visible prop knife in front of the camera
   - the weapon should fire within 1–2 seconds
   - if it does, the live webcam demo is ready

**End of Day 1 target:** one stable video that produces `ARMED PERSON` or `POSSIBLE ASSAULT` on screen with minimal false positives. That video + that command = your demo.

---

### Day 2 — Polish the demo and prepare the pitch

**Morning (2–3 hours): Add a demo video file fallback**

If the webcam live demo is inconsistent (lighting, prop size, background noise), build a pre-recorded demo file that always works.

Record 3 short clips covering these scenarios:
- Clip A: person walking normally — no threat state should show
- Clip B: person holding a visible knife — `ARMED PERSON` fires
- Clip C: two people, one holding a knife, moving close to each other — `POSSIBLE ASSAULT` fires

These three clips together tell the full story: the system knows the difference between normal and threat.

Save them in a `demo/` folder. Run each with the same locked command from Day 1.

**Midday (1–2 hours): Clean the on-screen output**

Make sure what appears on screen is readable and looks intentional:
- bounding boxes appear around person and weapon
- the threat state banner at the top is visible and clearly labeled
- evidence clips are saving to `runs/detect/`

If the overlay is too cluttered, use `--show-all-detections` off (default) to hide irrelevant COCO classes.

**Afternoon (2 hours): Prepare the demo narrative**

A working detector is only half the demo. The other half is what you say.

Frame it this way:

> "Traditional CCTV is passive — it records but doesn't think. We built an active threat detection layer. Watch what happens when someone enters the scene with a weapon..."

Then run Clip A (normal), then Clip B (knife), then Clip C (assault proximity). Let the system do the talking.

Be honest about what it does and does not do:
- it detects visible weapons and armed proximity — that is real
- fighting detection without a weapon is still heuristic — be upfront about that
- the current checkpoint is a placeholder — you know exactly what to upgrade next

Being honest about limitations in a demo actually builds more trust than overselling.

**End of Day 2 target:** three demo clips that run cleanly, a stable live webcam path, and a 3-minute verbal narrative.

---

## Phased Roadmap

### Phase 1 — Current (POC demo)
- Current pipeline as-is
- Raise `--weapon-conf` to 0.80–0.85 for demo
- Add ByteTrack tracking (one `pip install` away via Ultralytics)
- Use a pre-recorded demo video as the safe fallback

### Phase 2 — POC hardening (weeks 2–4)
- Replace weapon checkpoint: fine-tune RT-DETR on Open Images v7 knife/gun classes
- Add X3D-S or SlowFast fine-tuned on RWF-2000 for real violence detection
- Switch pose model to RTMPose
- Add ByteTrack person tracking for temporal stability

### Phase 3 — Nigeria-specific data (month 2–3)
- Stage and record local threat scenarios: bag snatch, warehouse theft, street confrontation
- Fine-tune detection and action models on locally collected data
- Add Grounding DINO as the VLM verification gate for high-confidence alerts
- Measure false positive rate on real deployment scenes as a benchmark

### Phase 4 — Deployment (month 3+)
- Edge: Jetson Orin NX, TensorRT INT8 export, offline-first architecture
- Cloud: GPU inference server, Kafka/Redis stream ingestion, operator dashboard
- Multi-camera support, WebSocket alert push, evidence review UI

---

## Evidence Output

All threat events are saved under `runs/detect/`.

Each event folder contains:
- `frame.jpg` — annotated still from the moment of detection
- `clip.mp4` — short annotated video clip (default 5 seconds)
- `metadata.json` — structured event data including threat assessment, detections, confidence scores, and timestamp

---

## Open Questions

These are unresolved and should drive the next planning session:

- Which exact video action model should be fine-tuned first: X3D-S (edge-first) or VideoMAEv2 (accuracy-first)?
- Will the first real deployment be edge (Jetson) or cloud (GPU server)?
- Who records the Nigeria-specific training scenarios, and when?
- Should the alert layer integrate with WhatsApp Business API, SMS, or a custom dashboard first?
- What is the minimum viable camera hardware for the first real-world pilot location?
