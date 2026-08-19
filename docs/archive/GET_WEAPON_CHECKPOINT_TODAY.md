# Get Or Create A Weapon Checkpoint Today

This is the exact path that is now working in this repo.

## Fastest path used here
Use an existing public YOLOv5 weapon checkpoint and load it through the local YOLOv5 runtime.

What was done:
- cloned `zaizou1003/knife_Gun_Detection` into `external/knife_Gun_Detection`
- copied `external/knife_Gun_Detection/exp6/weights/best.pt`
- saved it as `models/weapon_best.pt`
- cloned the official YOLOv5 repo into `external/yolov5`
- installed `external/yolov5/requirements.txt`
- validated that the checkpoint loads with classes `{0: 'gun', 1: 'knife'}`

## Exact commands

From the repo root:

```powershell
cd "C:\Users\Demilade\Desktop\CV Threat Intelligence"
git clone https://github.com/zaizou1003/knife_Gun_Detection.git external\knife_Gun_Detection
git clone https://github.com/ultralytics/yolov5.git external\yolov5
python -m pip install -r external\yolov5\requirements.txt
Copy-Item "external\knife_Gun_Detection\exp6\weights\best.pt" "models\weapon_best.pt" -Force
```

## Exact validation command

```powershell
cd "C:\Users\Demilade\Desktop\CV Threat Intelligence"
python -c "from detector import load_detection_model; m = load_detection_model('models/weapon_best.pt', 'external/yolov5', preferred_kind='yolov5'); print(m.kind, m.names)"
```

Expected output shape:

```text
yolov5 {0: 'gun', 1: 'knife'}
```

## Exact detector command for the same-day demo

```powershell
cd "C:\Users\Demilade\Desktop\CV Threat Intelligence"
python detector.py --source 0 --weights yolov8n.pt --person-weights yolov8n.pt --weapon-weights "models\weapon_best.pt" --weapon-loader yolov5 --person-classes person --weapon-classes knife,gun --threat-classes knife,gun --show
```

This setup means:
- `yolov8n.pt` handles `person`
- `models\weapon_best.pt` handles `gun` and `knife`
- `detector.py` merges both outputs
- the threat layer can then raise:
  - `DANGEROUS OBJECT`
  - `ARMED PERSON`
  - `POSSIBLE ASSAULT`

## If you want to create a new checkpoint instead of reusing one

Use `TRAIN_WEAPON_MODEL.md`.

The shortest training route is:

```powershell
cd "C:\Users\Demilade\Desktop\CV Threat Intelligence"
.\.venv\Scripts\Activate.ps1
yolo detect train data=training/weapon_data.example.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=8 device=cpu workers=0
```

That route is much slower for same-day delivery.

For today, the fastest practical checkpoint path is the one already completed in this repo:
- `models/weapon_best.pt`
