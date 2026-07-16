# Task Brief — Demi: finish the multi-stream detector (+ X3D)

> **Written:** 2026-07-16 19:22 WAT (Thursday)
> **Author:** Ayo (Atunrase)
> **For:** Demi
> **Branch:** `main` (all referenced code is committed + pushed)

## Context (where things are)
The multi-stream serving pipeline (`cvti/serving/`) runs many cameras through ONE
shared batched object-detector + ONE shared pose model, with per-camera tracking,
rules, an alert queue, and an async VLM gate. It currently produces, per camera:
**object-detection + zones + pose-based concealment (theft) + rules + gate.**

The single-stream detector (`cvti/detector/core.py`) runs the *full* set:
weapons, violence, theft state machine, concealment, zones, and the fine-tuned
video-action model. Your job is to bring the rest of that into the per-camera path.

Ayo owns the video-model track (training + data + eval) — don't duplicate that.

---

## Task 1 — wire the rest of the detector into `PerCameraState`

File: [`cvti/serving/camera.py`](../../cvti/serving/camera.py). Reference
orchestration: `cvti/detector/core.py` ~lines 1900–2010.

**The pattern is already set** — copy how concealment was done. `_concealment_events()`
is the template:
1. use a **shared** stateless model (loaded once in `run_site`, passed into
   `PerCameraState`),
2. keep any **stateful** bits **per camera** (pose-track state, detectors, gates),
3. reuse the existing `core.py` function (don't re-implement),
4. emit `RawEvent`s into the same `raw_events` list the rules engine evaluates.

Add these, in order (each is a new `_xxx_events()` method + wiring, same shape):

1. **Violence** — `assess_violence(pose_people, validated_weapon_detections, ...)`
   + `validate_weapon_detections(...)`; gate it with a per-camera
   `ViolenceTemporalGate` (stateful). Then `assessments_to_events(...)`.
   Note the ~6 tuned params (`violence_distance_ratio`, `violence_wrist_speed`,
   `violence_arm_extension_ratio`, `weapon_hand_distance_ratio`,
   `violence_wrist_accel`, `assault_distance_ratio`) — copy the argparse defaults
   from `core.py` so behaviour matches single-stream.
2. **Weapons** — `assess_threat(...)` + `validate_weapon_detections(...)` +
   `gate_assessment(..., min_threat_frames)` (per-camera frame counter).
3. **Theft state machine** — `TheftDetector` (stateful per camera) `.update(...)`.
4. **Video-action** — `build_video_action_runtime(...)` per camera (gated on a
   detector-triggered moment), running Ayo's fine-tuned model
   (`--video-action-model runs/video_finetune/videomae`). This is the heaviest;
   do it last and keep the per-camera cooldown.

**Shared vs per-camera:** pose model is already shared. Weapons may need a
separate weapon model — load it once in `run_site` and pass it in like the pose
model. Everything stateful (trackers, gates, theft/concealment detectors, pose
history, video-action runtime buffer) stays **per camera**.

**Test:** extend `configs/site_theft_demo.json` with the relevant clips/configs;
run `python -m cvti.serving.pipeline --site-config ... --gate-provider mock`;
add cases to `tests/test_serving.py`. Keep changes additive — don't touch
`core.py` (the single-stream path must keep working).

## Task 2 — X3D training (parallel, non-blocking)

The unified trainer already supports X3D:
```bash
pip install -r requirements-video.txt          # pytorchvideo for X3D
python -m cvti.training.video_finetune --backend x3d --unfreeze-last-block \
  --lr 5e-4 --epochs 40 --patience 8 --batch 2
```
Same CamNuvem dataset as VideoMAE, so it's a direct comparison. Baseline to beat:
**VideoMAE balanced_acc 0.852 (frozen) / 0.889 (unfreeze).** See `docs/TRAINING.md`
for the workflow, what to watch (`balanced_acc`, not recall), and caveats. Use
`scripts/train_video.sh --backend x3d ...` for a durable run.

## Gotchas we already hit (so you don't)
- Run long trainings via `scripts/train_video.sh` (durable), never raw commands
  from shell history — the job dies on session end otherwise.
- Watch **`balanced_acc`**, not recall (recall pins at 1.0 during class collapse).
- Class weighting is on by default (train split is theft-heavy).
- `sv.ByteTrack` is deprecated (supervision 0.28, removed 0.30) — heads up.

## Definition of done
Multi-stream pipeline raises weapon / violence / theft / video-action alerts per
camera (not just concealment/zones), verified on a multi-camera site config, with
`tests/test_serving.py` green and `core.py` untouched.
