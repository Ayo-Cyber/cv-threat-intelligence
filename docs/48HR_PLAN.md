# 48-Hour Demo Plan
**Tuesday Evening → Thursday Evening**

The goal is one clean, repeatable, honest demo — not a production system.  
A broken "impressive" demo is worse than a simple one that works every time.

---

## What Is Already Done

- ByteTrack tracking enabled on person model (2-line change, already committed)
- `predict_with_model` accepts `use_tracking=True` — weapon model still uses `predict` to avoid tracking noise on objects

---

## Tuesday Evening (tonight, ~2 hrs)

### Step 1 — Test RT-DETR swap (5 mins, zero risk)

RT-DETR is already inside Ultralytics. Just change the weights flag:

```powershell
python detector.py `
  --source 0 `
  --weights rtdetr-l.pt `
  --person-weights rtdetr-l.pt `
  --weapon-weights "models\weapon_best.pt" `
  --weapon-loader yolov5 `
  --person-classes person `
  --weapon-classes knife,gun `
  --threat-classes knife,gun `
  --weapon-conf 0.80 `
  --min-threat-frames 3 `
  --show
```

**Decision rule:**
- Runs at >15fps on your machine → keep `rtdetr-l.pt`
- Feels slow or stuttery → revert to `yolov8n.pt`, no one in the audience will know the difference

### Step 2 — Find or film your demo material (rest of the evening)

You need a 30–60 second video where a person is clearly holding a knife or prop.

**Options:**
- Film yourself holding a kitchen knife in good lighting (best — you control it)
- Free stock video: [Pexels](https://www.pexels.com) or [Pixabay](https://pixabay.com) — search `knife threat` or `prop gun`

**What makes a good weapon video:**
- Weapon fills at least 10% of the frame width — if it's too small the checkpoint misses it
- Good contrast between weapon and background — avoid dark knife on dark clothing
- Person's hand is visible and clearly gripping it
- 2–3 seconds of clear hold, not just a flash

**Stop coding after Step 1.** The rest of tonight is sourcing and filming.

---

## Wednesday (full day — your most important day)

### Morning (3 hrs) — Lock the demo command

Run your video through the detector with debug flags on:

```powershell
python detector.py `
  --source "your_test_video.mp4" `
  --weights yolov8n.pt `
  --person-weights yolov8n.pt `
  --weapon-weights "models\weapon_best.pt" `
  --weapon-loader yolov5 `
  --pose-weights yolov8n-pose.pt `
  --person-classes person `
  --weapon-classes knife,gun `
  --threat-classes knife,gun `
  --weapon-conf 0.55 `
  --min-threat-frames 2 `
  --violence-min-frames 3 `
  --debug-weapon `
  --show
```

Watch the terminal. `--debug-weapon` prints the exact confidence score every time a weapon fires.

**Tuning rules:**
- Weapon detected but noisy → raise `--weapon-conf` in steps: `0.55 → 0.65 → 0.75 → 0.80`
- Weapon not detected at all → lower `--weapon-conf` to `0.35`, check that weapon is large and clear in frame
- One-frame blips causing false alerts → raise `--min-threat-frames` to `3` or `4`

Once `ARMED PERSON` fires cleanly on your video with minimal false positives:  
**Write down the exact command. That is your locked demo command. Do not change it again.**

### Midday (2 hrs) — Record the 3 demo clips

You need exactly three short clips. These are your demo assets.

| Clip | Scene | Expected state |
|---|---|---|
| `demo/clip_a_clear.mp4` | Person walks in, no weapon | `CLEAR` — nothing fires |
| `demo/clip_b_armed.mp4` | Person holds weapon clearly toward camera | `ARMED PERSON` fires |
| `demo/clip_c_assault.mp4` | Two people visible, one armed, they move close | `POSSIBLE ASSAULT` fires |

Film these yourself in front of your webcam. Each clip should be 20–40 seconds.  
Run each through the locked demo command and confirm the expected state appears on screen.

Save the confirmed clips into a `demo/` folder. These are now untouchable.

### Afternoon (2 hrs) — Clean the on-screen output

Run the locked command and check:
- Bounding boxes appear cleanly around person and weapon
- The threat state banner at the top is readable and clearly labeled
- No irrelevant COCO classes cluttering the screen (`bird`, `cup`, `bear`) — if they appear, `--show-all-detections` is off by default, so check your `--weights` isn't the only model running
- `runs/detect/` has event folders with `frame.jpg`, `clip.mp4`, and `metadata.json`

Open one of the saved event folders during the demo — showing the evidence output is part of the story.

### Evening — Practice the narrative out loud

Time yourself. The demo narrative should be under 3 minutes.

```
[Clip A playing — CLEAR state]
"Traditional CCTV records everything and flags nothing.
 A security guard has to watch hours of footage to find one incident.
 We built an active intelligence layer on top of existing cameras."

[Switch to Clip B — ARMED PERSON fires]
"The moment a weapon enters the frame and is spatially attached to a person,
 the system raises a threat state and begins saving evidence automatically."

[Open runs/detect/ — show frame.jpg and metadata.json]
"Every event is timestamped, annotated, and saved.
 There is always a reviewable record — not just an alert."

[Switch to Clip C — POSSIBLE ASSAULT fires]
"When an armed person moves into proximity of another person,
 the threat level escalates."

[Pause]
"This is real inference on a live feed — not a simulation.
 The current model is a proof of concept. We know exactly what
 to upgrade next, and that is the conversation we want to have."
```

---

## Thursday

### Morning — One full dry run

Run the demo start to finish exactly as you will on Thursday evening.

- If something breaks and you fix it in under 30 minutes → fix it
- If it takes longer → cut the feature, use the pre-recorded clips only
- Do not add anything new on Thursday morning

### Afternoon — Do not touch the code

Rest. Charge your laptop. Check the demo folder is intact.

### Evening — Demo

Run `clip_a_clear.mp4` → `clip_b_armed.mp4` → `clip_c_assault.mp4`.  
If the audience asks for live webcam, run it only if you tested it for 30+ mins in the demo environment (same room, same lighting, same background).  
If you haven't — stay on the pre-recorded clips. That is not a weakness. It is a professional choice.

---

## Fallback Rules

| Problem | Fix |
|---|---|
| RT-DETR is slow | Revert to `yolov8n.pt` — same command, different weights |
| ByteTrack causes crash | Change `use_tracking=True` → `use_tracking=False` in `detector.py:1261` and `1271` |
| Weapon model too noisy live | Use pre-recorded clips only, do not demo live webcam |
| Evidence folder not saving | Add `--save-dir demo_evidence` and verify the folder exists after a test run |

---

## Useful Reference Links

- Ultralytics RT-DETR docs: https://docs.ultralytics.com/models/rtdetr/
- Ultralytics ByteTrack tracker docs: https://docs.ultralytics.com/modes/track/
- RWF-2000 dataset (real-world fighting, for Phase 2 training): https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
- Open Images v7 knife/gun classes (for Phase 2 weapon fine-tuning): https://storage.googleapis.com/openimages/web/index.html
- Grounding DINO repo (Phase 2 verification gate): https://github.com/IDEA-Research/GroundingDINO
- X3D model via PyTorchVideo (Phase 2 action recognition): https://github.com/facebookresearch/pytorchvideo
- RTMPose via rtmlib (Phase 2 pose upgrade): https://github.com/Tau-J/rtmlib
- Free stock video for demo material: https://www.pexels.com and https://pixabay.com

---

## What Success Looks Like Thursday Evening

- Three clips run cleanly without crashing
- `ARMED PERSON` fires on Clip B — visible on screen, no explanation needed
- `POSSIBLE ASSAULT` fires on Clip C
- `runs/detect/` has a saved event folder you can open and show
- You can answer "what would you improve next?" — the Phase 2 roadmap is the answer
