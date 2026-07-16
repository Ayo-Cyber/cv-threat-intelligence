# Demo Runbook

Everything to show the CVTI system working, with exact commands and what each
part proves. All of this is verified and runs on CPU/MPS — no live camera, no
GPU needed. Recorded clips act as the cameras.

## The story (product thesis in one line)
> Fast local detectors find suspicious moments → a fine-tuned video model adds a
> temporal opinion → the customer's rules decide what counts as a threat → a VLM
> gate confirms it before alerting. Threat meaning is customer-configurable.

## Prereqs (once)
```bash
cd cv-threat-intelligence
source .venv/bin/activate
# offline flags avoid any network hang on the VideoMAE weights (already cached)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp
```
Model weights auto-download once for YOLO; the fine-tuned VideoMAE is at
`runs/video_finetune/videomae/` (backed up at `videomae-best-bal0.889-*`).

---

## Demo 1 — Our fine-tuned robbery model (the headline)
**Shows:** we trained a VideoMAE model on CamNuvem that classifies a clip as
theft, and that signal flows into the rule engine as an alert.
```bash
python tools/video_action_probe.py data/test_clips/theft_shop_01.mp4 \
  --model runs/video_finetune/videomae \
  --hybrid-events --config configs/video_robbery_demo.json
```
**Expected:**
```
1. theft   0.878
2. normal  0.122
event: theft_candidate raw=0.878 ...
config alert: video_theft_candidate (high)
```
**Talking point:** "This is *our* model, fine-tuned on robbery CCTV — balanced
accuracy 0.89 on the held-out test split (recall 0.89, false-positive rate 0.11).
The label becomes a weak signal the customer's rules act on."

## Demo 1b — The fine-tuned model running INSIDE the live detector (strongest)
**Shows:** the same model, but running in the real pipeline — a per-frame
detector (concealment) flags the moment, the fine-tuned VideoMAE classifies the
event window as theft, and it becomes a gate-confirmed alert alongside the
always-on baseline.
```bash
python -m cvti.cli.detect --source data/test_clips/theft_shop_01.mp4 \
  --config configs/video_robbery_demo.json --concealment \
  --video-action-backend videomae --video-action-model runs/video_finetune/videomae \
  --video-action-cooldown 0 --gate-provider mock
```
**Expected:**
```
[VideoAction] videomae:runs/video_finetune/videomae 31..91 top=theft (0.986)
[CONFIRMED] video_theft_candidate (HIGH) — VIDEO ACTION: theft ...
[CONFIRMED] baseline_violence (CRITICAL) ...
```
**Talking point:** "This is the whole chain live: cheap detector finds the
moment → our fine-tuned model reads the motion (theft 0.98) → rule + gate →
alert. And the always-on safety baseline fires in parallel."

## Demo 2 — Zoning + loitering (customer-defined rules)
**Shows:** draw a zone → track dwell → a customer rule fires → gate confirms.
```bash
python -m cvti.retail.zones --source data/test_clips/theft_shop_01.mp4 \
  --zones configs/retail_zones.theft_shop_01.json \
  --rules configs/shelf_zones_demo.json
```
**Expected:** `[RULE FIRED] loitering_at_shelf ... person #6 ...`
**Talking point:** "Zones + dwell are customer-configured — the same engine, a
different JSON per site."

## Demo 3 — Multi-stream on one model (scale story)
**Shows:** many cameras through ONE batched model, per-camera rules, one alert
queue, async gate.
```bash
python -m cvti.serving.pipeline --site-config configs/site_demo.json --gate-provider mock
```
**Expected:** `batch_sizes={2}`, both cameras fire loitering, gate confirms,
`deduped=...`.
**Talking point:** "One box, one shared model, N cameras batched together — the
VLM gate (not detection) is the only real ceiling, and the alert queue keeps it
from being flooded."

## Demo 4 — Full detector + always-on safety baseline
**Shows:** violence detected → confirmed alert; the critical baseline fires even
if the customer config didn't ask for it.
```bash
python -m cvti.cli.detect --source data/test_clips/violence_suspected.mp4 \
  --config configs/shelf_zones_demo.json --gate-provider mock --max-frames 40
```
**Expected:** `[CONFIRMED] baseline_violence (CRITICAL) ... frames=4(motion_peak_span)`
**Talking point:** "A narrow customer config can't hide a critical threat — the
always-on baseline still catches violence/weapons, and the gate is sent
motion-peak frames, not one blurry still."

---

## If something misbehaves
- **VideoMAE slow / first call laggy:** it's MPS; the probe takes ~7s per clip. Fine for a demo.
- **A clip fires nothing:** most clips are single-signal; use the clip named for
  the threat you're showing (theft_shop_01 for theft, violence_suspected for violence).
- **`--gate-provider mock`** auto-confirms — that's intentional for offline demo.
  For a "real" gate, use `--gate-provider ollama` with a local model running.
- **Fine-tuned model path:** if `runs/video_finetune/videomae/` is missing, use the
  backup `runs/video_finetune/videomae-best-bal0.889-20260716/`.

## Honest caveats (say these, don't hide them)
- The robbery model's 0.89 is on a **36-clip** test set — a strong first result,
  not a validated production number. More normal/hard-negative data is the next step.
- The multi-stream path currently runs **object-detection + zones**; pose/violence
  in the multi-cam path is in progress.
