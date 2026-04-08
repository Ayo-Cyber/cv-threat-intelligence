# CV Threat Intelligence POC

This repo now contains the first starter build for the 36-hour proof of concept.

The goal of this POC is simple:
- connect a live camera feed
- run computer vision inference on each frame
- highlight detections on screen
- trigger a threat alert for configured classes
- save evidence frames and short clips when a threat is detected

## What This Starter Supports
- webcam input
- RTSP stream input
- video file input as a safe demo fallback
- configurable YOLO weights
- configurable threat classes
- evidence saving for detections

## Important Reality Check
If you use standard pretrained YOLO weights such as `yolov8n.pt`, you will usually only get common object classes from public datasets.

That means:
- the pipeline itself can be proven immediately
- true `knife`, `gun`, `fight`, or `stealing` detection will likely require custom weights or a more specialized model

So the fastest path is:
1. prove the live pipeline works
2. test with `person` or other available classes first
3. swap in custom weights as soon as you have them

## Recommended 36-Hour Plan
### Track 1: POC demo
- run the detector on a webcam
- verify overlays and alerts work
- verify evidence files are saved
- test the same app with a video file
- test the same app with an RTSP stream when available

### Track 2: model experimentation
- use Colab only for quick model experimentation or fine-tuning
- keep live inference local for the demo
- do not block the POC on Jetson or Jetson-like deployment work

## Quick Start

### Windows

```powershell
# 1. Clone the repo
git clone https://github.com/DEMILADE07/cv-threat-intelligence.git
cd cv-threat-intelligence

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r external/yolov5/requirements.txt

# 4. Run the demo
python detector.py `
  --source 0 `
  --weights rtdetr-l.pt `
  --person-weights rtdetr-l.pt `
  --pose-weights yolov8n-pose.pt `
  --person-classes person `
  --weapon-classes knife,gun `
  --threat-classes knife,gun `
  --weapon-conf 0.80 `
  --min-threat-frames 2 `
  --violence-min-frames 3 `
  --debug-weapon `
  --debug-violence `
  --show
```

> **Note:** If ByteTrack causes a crash, add `--no-track` to the command.

### Mac

```bash
# 1. Clone the repo
git clone https://github.com/DEMILADE07/cv-threat-intelligence.git
cd cv-threat-intelligence

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r external/yolov5/requirements.txt

# 4. Grant camera permission
# System Settings → Privacy & Security → Camera → enable Terminal
# Restart Terminal after granting, then re-run: source .venv/bin/activate

# 5. Run the demo
python3 detector.py \
  --source 0 \
  --weights rtdetr-l.pt \
  --person-weights rtdetr-l.pt \
  --pose-weights yolov8n-pose.pt \
  --person-classes person \
  --weapon-classes knife,gun \
  --threat-classes knife,gun \
  --weapon-conf 0.80 \
  --min-threat-frames 2 \
  --violence-min-frames 3 \
  --debug-weapon \
  --debug-violence \
  --show
```

> **Note:** `rtdetr-l.pt` downloads automatically (~63MB) on first run.  
> If it runs slowly on your machine, swap to `--weights yolov8n.pt --person-weights yolov8n.pt` instead.

### Run on a video file instead of webcam

```bash
# Windows
python detector.py --source "demo\clip_b_armed.mp4" ...

# Mac
python3 detector.py --source "demo/clip_b_armed.mp4" ...
```

### What you should see on startup

```
Loading model...
Loaded default model from: rtdetr-l.pt (ultralytics)
Loaded dedicated person model: rtdetr-l.pt (ultralytics)
Loaded pose model: yolov8n-pose.pt (ultralytics)
ByteTrack person tracking: ON
Starting inference loop. Press 'q' to quit.
```

### Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'torch'` | Virtual environment not active — run `source .venv/bin/activate` (Mac) or `.\.venv\Scripts\Activate.ps1` (Windows) |
| `ModuleNotFoundError: pandas` | `pip install -r external/yolov5/requirements.txt` |
| `Unable to open source: 0` on Mac | Grant camera access: System Settings → Privacy & Security → Camera → enable Terminal, then restart Terminal |
| `Unable to open source: 0` on Windows | Try `--source 1` — different camera index |
| ByteTrack crash | Add `--no-track` to the command |
| Weapon model not found | Confirm `models/weapon_best.pt` exists in the repo root |

### 2. Run a smoke test with your webcam
This proves the live pipeline works.

```powershell
python detector.py --source 0 --weights yolov8n.pt --threat-classes person --show
```

This is not your final threat logic. It is just the fastest way to validate:
- camera capture
- frame inference
- bounding box rendering
- alerting
- evidence saving

### 3. Run with an RTSP stream
```powershell
python detector.py --source "rtsp://username:password@camera-ip:554/stream" --weights yolov8n.pt --threat-classes person --show
```

### 4. Run with a prerecorded video
```powershell
python detector.py --source "demo.mp4" --weights yolov8n.pt --threat-classes person --show
```

## Running With Custom Threat Weights
When you have custom weights for classes like `knife`, `gun`, or `fight`, run:

```powershell
python detector.py --source 0 --weights "models\best.pt" --threat-classes knife,gun,fight --show
```

## Threat Logic Layer
The detector now supports a second layer of rule-based threat assessment on top of raw detections.

Useful arguments:
- `--person-classes`: labels treated as people by the rule engine
- `--weapon-classes`: labels treated as dangerous objects
- `--threat-classes`: explicit classes that should still trigger an alert directly
- `--assault-distance-ratio`: controls how close an armed person must be to another person before the app flags `POSSIBLE ASSAULT`

Example with custom weapon weights:

```powershell
python detector.py --source 0 --weights "models\best.pt" --person-classes person --weapon-classes knife,gun --threat-classes knife,gun --show
```

Example with separate person and weapon models:

```powershell
python detector.py --source 0 --weights yolov8n.pt --person-weights yolov8n.pt --weapon-weights "models\weapon_best.pt" --weapon-loader yolov5 --person-classes person --weapon-classes knife,gun --threat-classes knife,gun --show
```

This is the best same-day setup when your custom checkpoint only knows weapon classes.

Useful live-tuning flags:
- `--weapon-conf 0.65` or higher to reduce false positives
- `--debug-weapon` to print exact weapon detections and confidences
- `--min-threat-frames 3` to ignore one-frame blips before raising a threat

Example with stricter live tuning:

```powershell
python detector.py --source 0 --weights yolov8n.pt --person-weights yolov8n.pt --weapon-weights "models\weapon_best.pt" --weapon-loader yolov5 --person-classes person --weapon-classes knife,gun --threat-classes knife,gun --weapon-conf 0.80 --min-threat-frames 3 --debug-weapon --show
```

## Violence Heuristics
The detector now also supports a pose-based heuristic violence layer.

This is not a trained action-recognition model. It uses:
- person proximity
- wrist motion speed
- arm extension
- weapon-to-hand attachment heuristics

New high-level states:
- `VIOLENCE SUSPECTED`
- `POSSIBLE STABBING`
- `POSSIBLE ARMED ASSAULT`

Recommended violence test command:

```powershell
python detector.py --source 0 --weights yolov8n.pt --person-weights yolov8n.pt --weapon-weights "models\weapon_best.pt" --weapon-loader yolov5 --pose-weights yolov8n-pose.pt --person-classes person --weapon-classes knife,gun --threat-classes knife,gun --weapon-conf 0.80 --min-threat-frames 3 --violence-min-frames 4 --debug-weapon --debug-violence --show
```

If you want to temporarily disable pose-based violence logic:

```powershell
python detector.py --source 0 --weights yolov8n.pt --person-weights yolov8n.pt --weapon-weights "models\weapon_best.pt" --weapon-loader yolov5 --pose-weights "" --person-classes person --weapon-classes knife,gun --threat-classes knife,gun --show
```

Expected on-screen states:
- `DANGEROUS OBJECT`: dangerous item visible
- `ARMED PERSON`: a weapon appears spatially attached to a detected person
- `POSSIBLE ASSAULT`: an armed person is close to another detected person

Important:
- these higher-level states are currently heuristic
- they are meant for the same-day POC demo layer, not as final action-recognition claims

## Evidence Output
Detections are saved under `runs\detect\`.

Each event can produce:
- an annotated image
- a short annotated clip

## Suggested Immediate Next Steps
1. Run the webcam smoke test first.
2. Confirm the pipeline works locally on your machine.
3. Add your rented RTSP camera as the second test source.
4. Obtain or train weapon-aware weights for `knife` and `gun`.
5. Use the new threat-rule layer to demo `ARMED PERSON` and `POSSIBLE ASSAULT`.
6. Use Colab only if you need quick training or fine-tuning.

## Honest Recommendation
For this first deadline, do not try to solve all threat categories at once.

The best milestone is:
- one working detector app
- one live input
- one or two detectable threat classes
- one clean demo for your co-founder
