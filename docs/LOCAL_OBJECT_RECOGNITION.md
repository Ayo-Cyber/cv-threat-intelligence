# Local object recognition experiment

This lane answers **“is this enrolled object in this crop?”** with the same
canonical crop, embedding index, and matcher used by object watch. It is not a
generic tracker, an alert sender, or a live-performance benchmark. Nothing in
these commands downloads a model.

## Environment and local assets

The approved offline assets are provisioned in this checkout. Use the existing
shared CVTI virtual environment, but expose the isolated OpenAI CLIP runtime to
the launched process with `PYTHONPATH`. The shared environment itself was not
changed and `clip` remains absent when `PYTHONPATH` is unset.

```bash
ROOT="/Users/macbook/Desktop/Career/CV Threat Intelligence/argus-object-watchlists"
PY="/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python"
cd "$ROOT"
export PYTHONPATH="$PWD/models/watchlist/python"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="$ROOT/models/watchlist/siglip-base-patch16-224"
WORLD="$ROOT/models/watchlist/yolov8s-worldv2.pt"
CLIP="$ROOT/models/watchlist/ViT-B-32.pt"
"$PY" -m cvti.cli.watch_objects --help
```

The model weights total 1,192,572,674 bytes. The SigLIP directory contains
`config.json`, `preprocessor_config.json`, and `model.safetensors`. This is an
image-only embedding path, so tokenizer files are not required. YOLO-World uses
the local OpenAI CLIP weights for text features. Official pinned source URLs,
revisions, SHA-256 values, package versions, and sizes are retained in the
ignored `runs/model-provisioning/manifest.json`; inference does not download
assets.

OpenAI CLIP and its small dependencies live only in
`models/watchlist/python`. Preserve the same `PYTHONPATH` in any shell, service,
or Electron/backend launch environment that needs the proposal provider. UI
presets and runtime configuration are local to the site; provisioning does not
globally enable a model, camera, rule, TrueSight gate, or alert route.

The retained CPU smoke site can be checked without modifying it:

```bash
SITE="$ROOT/runs/pretrained-watchlist-smoke/site"
"$PY" -m cvti.cli.watch_objects doctor --site-dir "$SITE"
```

At the retained run, `doctor` reported the SigLIP backend, YOLO-World provider,
and one active target (`demo-green-atm-kiosk`) ready, with model fingerprint
`siglip-94e242717dd26c6705b75aae`. To configure a fresh local site instead:

```bash
SITE="$ROOT/runs/object-watch-onboarding"
"$PY" -m cvti.cli.watch_objects configure \
  --site-dir "$SITE" \
  --model-path "$MODEL" --device cpu \
  --world-weights "$WORLD" \
  --clip-weights "$CLIP"
"$PY" -m cvti.cli.watch_objects doctor --site-dir "$SITE"
```

Start with CPU. Use `--device mps` only after confirming this installed Torch,
Transformers, and model combination supports every required MPS operation.
`doctor` prints JSON and returns nonzero when required artifacts are missing; a
missing model never becomes a hash-based substitute. The production proposal
provider and this experiment CLI consume the configured local YOLO-World and
OpenAI CLIP weights directly. `doctor` reports proposal readiness separately
from the required SigLIP backend and does not download or load missing assets.

## Exact command surface

- `configure`: required `--site-dir`, `--model-path`; optional
  `--device {cpu,mps}` (default `cpu`), `--world-weights`, `--clip-weights`.
- `doctor`: required `--site-dir`.
- `enroll`: required `--site-dir`, `--object-id`, `--label`, and repeatable
  `--image`; optional `--description`, `--category
  {product,vehicle,pallet,ppe,custom}` (default `custom`), repeatable
  `--negative-image`, `--bbox X1 Y1 X2 Y2`, and `--review`.
- `embed`: required `--site-dir`.
- `activate`: required `--site-dir`, `--object-id`.
- `run`: required `--site-dir`, `--source`, `--output-dir`; optional `--weights`,
  `--proposals {none,yolo_world}` (default `none`), `--max-frames`, `--save-video`,
  and `--show`. Source is an existing local video path or webcam index `0`.
- `eval-crops` (alias `eval`): required `--site-dir`, `--candidates`, and
  `--output-dir`.

## Enrol, embed, and activate

Use multiple views. Keep held-out evaluation images separate from enrollment,
and include separately captured lookalike negatives rather than reusing test
images. Prefer preparing one tight object crop per image first; then omit
`--bbox` so each input is independently cropped rather than applying one shared
box to differently framed images.

```bash
"$PY" -m cvti.cli.watch_objects enroll --site-dir "$SITE" \
  --object-id chi-carton --label "Chi carton" \
  --description "branded brown shipping carton" \
  --image /absolute/path/target-front-crop.jpg \
  --image /absolute/path/target-side-crop.jpg \
  --negative-image /absolute/path/lookalike-negative-crop.jpg \
  --review
```

`--review` is explicit and applies to all supplied positive and negative
crops. Without it, examples remain draft and cannot silently become active.
By default the whole oriented image is canonicalized. To crop all supplied
images with pixel XYXY coordinates, add `--bbox X1 Y1 X2 Y2`.

```bash
"$PY" -m cvti.cli.watch_objects embed --site-dir "$SITE"
"$PY" -m cvti.cli.watch_objects activate --site-dir "$SITE" --object-id chi-carton
"$PY" -m cvti.cli.watch_objects doctor --site-dir "$SITE"
```

Use a dedicated `SITE`: this experiment writes an object library and is not a
production database. Activation makes the compatible target available to object
matching; it does not enable a camera or rule and does not automatically produce,
persist, verify, or notify an alert.

## Recognition-first video run

An existing generic YOLO weights file can supply geometric candidates:

```bash
"$PY" -m cvti.cli.watch_objects run --site-dir "$SITE" \
  --source /data/held-out/aisle.mp4 --weights /models/local-yolo.pt \
  --proposals none --output-dir /results/chi-run-001 \
  --max-frames 300 --save-video
```

To ground proposals with the active targets' descriptions, configure both local
proposal assets and select the production provider:

```bash
"$PY" -m cvti.cli.watch_objects configure --site-dir "$SITE" \
  --model-path "$MODEL" --device cpu \
  --world-weights "$WORLD" \
  --clip-weights "$CLIP"
"$PY" -m cvti.cli.watch_objects run --site-dir "$SITE" \
  --source /data/held-out/aisle.mp4 --proposals yolo_world \
  --output-dir /results/chi-world-001 --max-frames 300
```

`--weights` may also be supplied in this mode. Generic and grounded boxes are
merged under the configured candidate budget. Proposal labels only guide crop
discovery; final object labels come from SigLIP matching against the enrolled
library. This command establishes local execution support, not recognition or
localization accuracy.

The output directory must not exist. The command preflights the semantic
backend before creating it and preserves partial files after processing
failures. `decisions.jsonl` records matched, rejected, and ambiguous crop
decisions, similarities, proposal counts via the summary, and measured
recognition timing. `summary.json` reports true processed counts. Diagnostic
presence candidates are debounced once per geometric encounter; they are not
verified alerts and no operator HTTP notification is sent. `annotated.mp4` is
a constant-rate visualization and does not preserve all source timing.

This `watch_objects run` command is an offline diagnostic. The operator product
entry point is the main Argus GUI, which requires the site's existing TrueSight
configuration for verification and normal alert handling. CLI samples,
`decisions.jsonl`, evidence pairs, and `presence_candidate` rows are diagnostic
proposals—not user alerts.

With neither `--weights` nor a supported proposal provider, `--proposals none`
intentionally produces no candidates; it does not treat the whole frame as an
object. Webcam source `0` is supported. All paths are local and automatic
weight acquisition is disabled.

## Retained real-model smoke

The final CPU/offline smoke reused the one-target site above. Reproduce it only
with new, nonexistent output directories:

```bash
"$PY" -m cvti.cli.watch_objects run \
  --site-dir "$ROOT/runs/pretrained-watchlist-smoke/site" \
  --source "$ROOT/data/test_clips/theft_yt_01.mp4" \
  --output-dir "$ROOT/runs/pretrained-watchlist-smoke/positive-recheck100" \
  --proposals yolo_world --max-frames 100 --save-video

"$PY" -m cvti.cli.watch_objects run \
  --site-dir "$ROOT/runs/pretrained-watchlist-smoke/site" \
  --source "$ROOT/data/test_clips/empty_warehouse.mp4" \
  --output-dir "$ROOT/runs/pretrained-watchlist-smoke/negative-recheck60" \
  --proposals yolo_world --max-frames 60 --save-video
```

The retained `positive-final100` output processed 100 frames, produced 75
YOLO-World candidates and 75 SigLIP matches, had 25 frames with no proposal, and
emitted one continuity-debounced diagnostic presence candidate. The retained
`negative-final60` output processed 60 frames and produced no proposals, matches,
or presence candidates. `final-report.json` and `FINAL_PROVENANCE.md` contain the
exact retained facts and commands.

The crop-only controls at unchanged target threshold `0.72` accepted three
same-source positives and rejected three easy controls (vehicle region, yellow
bollard, and bus). See `crop-controls/report.json` and visually inspect
`crop-controls/contact-sheet.png`. These are tiny same-source/control checks, not
independent accuracy or lookalike evidence. Neither run measured VLM/TrueSight
verification, a persisted user alert, a notification, live throughput, removal,
or loading behavior.

## Crop-only recognition evaluation

To isolate semantic recognition from discovery, create JSONL manually with one
held-out image and ground-truth pixel bbox per line:

```json
{"image":"/data/held-out/positive-01.jpg","bbox":[120,80,420,350],"expected_object_id":"chi-carton"}
{"image":"/data/held-out/lookalike-01.jpg","bbox":[40,30,250,220],"expected_object_id":null}
```

Run:

```bash
"$PY" -m cvti.cli.watch_objects eval-crops --site-dir "$SITE" \
  --candidates /data/held-out/candidates.jsonl \
  --output-dir /results/chi-crops-001
```

The report explicitly sets `detector_bypass: true` and
`localization_accuracy_measured: false`. It exercises the canonical runtime
crop, configured backend, immutable recognition index, and matcher—not a
replica. `evidence/` contains a bounded sample of candidate/reference crop
pairs. Accuracy therefore describes recognition on manually supplied boxes
only and must not be presented as proposal, tracking, localization, source
duration, or real-time performance.
