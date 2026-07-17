# Training video models (VideoMAE / X3D) — the way forward

Fine-tune a temporal model on the CamNuvem robbery-vs-normal clips. This is the
repeatable workflow; the earlier pain was process/command mistakes, not the model.

## TL;DR — one command

```bash
scripts/train_video.sh                        # good default (frozen head, lr 1e-3)
scripts/train_video.sh --unfreeze-last-block  # try to beat the frozen-head ceiling
```

The launcher runs the **current** code with the recipe that works, survives the
terminal closing (`nohup`) and the Mac sleeping (`caffeinate`), and logs to a
timestamped file. It prints the log path + how to watch/stop.

**Do not** re-run raw `python -m ...` commands from shell history — that's how we
kept launching a stale recipe. Always use the script.

## Watch it

```bash
tail -f runs/video_finetune/train-*.log          # live (Ctrl-C only stops watching)
cat runs/video_finetune/videomae/metrics.json     # per-epoch metrics
```

Watch **`balanced_acc`** (mean of recall + specificity), NOT raw recall. On this
dataset the train/test splits are class-inverted (train 69% theft, test 75%
normal), so a lazy "always theft" head shows recall 1.0 but balanced_acc ~0.5.
`balanced_acc > 0.5` = actually learning; best model + early-stop track it.

## Stop

```bash
pkill -f video_finetune
```

## Recipes (which `--freeze` mode)

| Flag | Trains | Use when |
|---|---|---|
| *(none)* | full model | large dataset; overfits fast on ~256 clips |
| `--freeze-backbone` | classifier head only | small dataset, fastest, safest baseline |
| `--unfreeze-last-block` | top encoder block + head | push accuracy past the frozen-head plateau |

Always-on regardless of recipe: **class-weighted loss** (stops the majority-class
collapse) and **balanced-accuracy** best/early-stop selection.

Useful flags: `--backend {videomae,x3d}` · `--epochs` · `--patience` (early stop)
· `--lr` (use ~1e-3 for a from-scratch head, lower for partial/full) · `--batch`
· `--per-class-limit N` (tiny smoke test) · `--device {mps,cuda,cpu}`.

## Current best (baseline to beat)

VideoMAE, frozen head, lr 1e-3, MPS, 220 train / 36 test:
**balanced_acc 0.852 @ epoch 7** — recall 0.889, FPR 0.185 (8/9 theft, 22/27 normal).
Checkpoint: `runs/video_finetune/videomae/` (best-by-balanced-acc, auto-saved).

## Retraining / next runs

- To try to beat 0.85: `scripts/train_video.sh --unfreeze-last-block --lr 5e-4 --epochs 40 --patience 8 --batch 2`
- X3D (teammate): `scripts/train_video.sh --backend x3d --unfreeze-last-block ...`
  (needs `pip install -r requirements-video.txt` for pytorchvideo).
- Real GPU: add `--device cuda`, bump `--batch`.

## Evaluate a checkpoint (Phase 5 metrics)

Training-loop numbers use CamNuvem's native split, which is **class-inverted**
(train 69% theft, test 75% normal) — misleading on its own. For honest metrics:

```bash
# full metrics on a checkpoint (native test — valid only if it was NOT trained stratified)
python -m cvti.training.video_eval --checkpoint runs/video_finetune/videomae

# trustworthy: train AND eval on a balanced (stratified) split, no leakage
scripts/train_video.sh --unfreeze-last-block --stratified --lr 5e-4 --epochs 40 --patience 8 --batch 2 \
  --out runs/video_finetune_stratified
python -m cvti.training.video_eval --checkpoint runs/video_finetune_stratified/videomae --stratified
```
`video_eval` reports accuracy, **balanced_acc**, precision/recall/F1 (theft),
**FPR** (normal), and a confusion matrix, and warns when the test set is too
small to call an FPR "validated". Always eval a checkpoint on data it did NOT
train on — `--stratified` uses the same seed as training, so it's leakage-free
only for a checkpoint that was itself trained `--stratified`.

Baseline (existing checkpoint, native test): balanced_acc 0.889, recall 0.889,
precision 0.727, F1 0.80, FPR 0.111.

## Known limits / honest caveats

- **36-clip test set is tiny** — 0.85 is encouraging, not a validated FPR. Needs
  more normal / hard-negative clips (plan.md Phase 5) before trusting it.
- The trained head is **2-class (normal/theft)**. The live hybrid detector maps
  *Kinetics* action labels, so using this checkpoint via
  `--video-action-backend videomae --video-action-model runs/video_finetune/videomae`
  needs a small label adapter first (not a drop-in yet).
- `runs/` is gitignored — checkpoints/logs stay local. Copy a good checkpoint
  somewhere durable if you want to keep it.
