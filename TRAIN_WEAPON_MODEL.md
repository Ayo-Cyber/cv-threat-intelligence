# Train Custom Weapon Weights

This is the exact next step after the pipeline proof.

The goal is not to solve full threat intelligence yet. The goal is to produce one custom YOLO detector that is better aligned with this project than stock `yolov8n.pt`.

## First model scope

Start with object detection only:

- `knife`
- `gun`

Do not start with:

- `fight`
- `steal`
- bag snatching
- general suspicious behavior

Those are temporal behavior problems and will slow the POC down.

## Recommended training route

Use Ultralytics YOLO fine-tuning:

- base model: `yolov8n.pt`
- task: detection
- classes: `knife`, `gun`
- output: `best.pt`

That `best.pt` is the custom weights file you will plug into `detector.py`.

## Dataset layout

Create the dataset under `datasets/weapon-detection/` using this structure:

```text
datasets/
  weapon-detection/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
```

YOLO expects:

- one image file per sample under `images/...`
- one `.txt` label file with the same base filename under `labels/...`

Example:

```text
datasets/weapon-detection/images/train/frame_001.jpg
datasets/weapon-detection/labels/train/frame_001.txt
```

## Label format

Each label file uses YOLO detection format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `0-1`.

Class IDs for this project:

- `0` = `knife`
- `1` = `gun`

Example label file:

```text
0 0.512500 0.460000 0.125000 0.310000
1 0.742188 0.521875 0.180469 0.221875
```

## Data file

Use `training/weapon_data.example.yaml` as the starting template.

## What data to collect

Prioritize samples that look like your target deployment conditions:

- webcam scenes
- room and corridor scenes
- surveillance-like viewpoints
- people holding knives
- people holding guns
- partial occlusion
- low-to-medium lighting
- different distances from camera

Also include hard negatives:

- phones
- remotes
- wallets
- kitchen utensils that are not threats
- hands with no object
- people standing, sitting, moving normally

The negative set matters because otherwise the model will overfire on handheld objects.

## Minimum useful first pass

Aim for:

- `150-500` knife images
- `150-500` gun images
- a strong set of negatives mixed into scenes

If you can stage your own images quickly, do it. For this project, staged camera-like data is often more useful than random polished internet images.

## Local training command

From the repo root:

```powershell
.\.venv\Scripts\Activate.ps1
yolo detect train data=training/weapon_data.example.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=8 device=cpu workers=0
```

Use `batch=8` as a conservative starting point on this machine. If memory is tight, drop to `batch=4`.

## Colab training command

If you train in Colab, upload the dataset and run:

```python
!pip install ultralytics
!yolo detect train data=training/weapon_data.example.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=16 device=0
```

Use Colab for training only. Keep live inference local in this repo.

## Expected output

Ultralytics will write weights under a run directory similar to:

```text
runs/detect/train/weights/best.pt
```

Copy that file into a stable location for inference, for example:

```text
models/best.pt
```

## Run the live app with custom weights

Once you have a trained checkpoint:

```powershell
python detector.py --source 0 --weights "models\best.pt" --threat-classes knife,gun --show
```

Or directly from the training run output:

```powershell
python detector.py --source 0 --weights "runs\detect\train\weights\best.pt" --threat-classes knife,gun --show
```

## Evaluation rule for this phase

Do not ask whether the model is production ready.

Only ask:

- does it detect `knife` and `gun` better than generic weights?
- does it work on your actual camera setup?
- are false positives manageable in a staged demo?

If yes, that is enough for the next milestone.
