# Session Handoff — Merge + Multi-Stream Serving + plan.md Phases 1–4

**Audience:** engineering colleague picking this up.
**Scope:** everything done this session. Two thrusts — (A) merge the ML branch
into the installable package and make it actually run, (B) build a multi-stream
serving layer and knock out plan.md Phases 1–4.
**Test status:** 33 → **57 passing** (`python -m pytest tests/ -q`).

---

## 0. TL;DR / where the code is

- `unify-detector` was merged into **`main`** and the whole thing was made
  runnable again (stale imports fixed, verified end-to-end, `SYSTEM_GUIDE.md`
  written).
- All the new backend work (multi-stream serving + plan.md Phases 1–4) is on
  branch **`phase8-throughput`** — **not merged to `main`, not pushed.**
- Nothing here needs a GPU to run; it all runs on CPU/MPS. A live camera is not
  required — recorded clips in `data/test_clips/` act as cameras.

**On `main` (from the merge):**
```
8a0ecaf docs: add full System Guide
3b1efee fix: repoint stale flat imports to cvti package paths
ffc5b93 Merge branch 'unify-detector'
```

**On `phase8-throughput` (this session's backend work, newest first):**
```
09b6ca7 feat(phase3): compound threat recipes (signals + logic + gate_question)
588e200 feat(phase4): per-rule evidence-frame selection for the gate
aa38453 feat(phase2): always-on critical safety baseline
74b95b6 feat(phase1): single-stream detector uses the alert queue, not just [0]
9f057da fix(phase8): frame-skip sampling so target-fps doesn't slow file playback
e40231e feat(phase8.1/8.3): wire multi-camera detect->track->rules->async gate
68695ad fix(phase8.0): report VideoMAE as per-event, not per-frame, in projection
6c39829 feat(phase8.1): multi-stream serving skeleton + gate-latency bench
dbc5f27 feat(phase8.0): multi-stream throughput benchmark + Phase 8 plan
fa2a2ad fix: repoint --zones import + add shelf-zone demo config
```

---

## 1. The merge (unify-detector → cvti package) — on `main`

`main` had restructured everything into the installable `cvti/` package;
`unify-detector` was still the old flat layout. So it was a rename-vs-modify
conflict across the board, plus a pile of new flat-import files.

What was done:
- **Gate providers combined, not chosen** — `cvti/verification/gate.py` now
  supports `mock / anthropic / local / openai_compatible / openrouter / ollama`,
  `base_url`, multi-frame verify, and CoT (both sides' features kept).
- **VideoMAE moved into the package** — `video_action_{model,hybrid,runtime}.py`
  now live under `cvti/`; imports repointed in the modules, tests, and tools.
- **Stale runtime imports fixed** — several `from concealment import…`,
  `from retail_zones import…`, `from agent_mapper import…`, `from detector import…`
  were lazy imports inside functions that byte-compile/tests didn't catch. Found
  by *running* the detector. All repointed to `cvti.*`.
- **Verified:** `cvti-detect` on a clip fires `[CONFIRMED] violence` and saves a
  clip/frame/metadata event bundle.
- **`docs/SYSTEM_GUIDE.md`** documents the pipeline, install, run, customize.

> Note: the rich per-frame detectors (weapons/violence/theft/concealment/VideoMAE)
> run in the **single-stream** `cvti/detector/core.py`. See §3 for what the
> multi-stream path does and does not yet run.

---

## 2. Zoning verified (no live camera)

- Ran zones on `data/test_clips/theft_shop_01.mp4` with
  `configs/retail_zones.theft_shop_01.json`; proved the full
  **zone → dwell → rule → alert** path (loitering on tracks #6/#8).
- Produced a zone-overlay image and an annotated `runs/zone_demo.mp4` (generated
  artifacts, not committed).

Run it:
```bash
python -m cvti.retail.zones \
  --source data/test_clips/theft_shop_01.mp4 \
  --zones  configs/retail_zones.theft_shop_01.json \
  --rules  configs/shelf_zones_demo.json
```

---

## 3. Multi-stream serving layer — plan.md **Phase 8** (new)

Target (confirmed with product): **edge box per site, NVIDIA GPU, 5–16 cameras,
1–2 s glass-to-alert.** plan.md never covered throughput/serving, so a new
"Phase 8" section was added to it.

### New module: `cvti/serving/`
| File | Role |
|---|---|
| `streams.py` | Per-camera decode thread. Latest-frame-drop (stay real-time), **frame-skip to target FPS** (don't slow playback), video-time timestamps, RTSP reconnect hook. |
| `batcher.py` | `collect_batch()` — one fresh frame per camera into a batch. |
| `alert_queue.py` | **`AlertQueue`** — dedup by `(camera, rule, track, zone, object)` within a cooldown, priority-ordered, throttled `drain()`. Thread-safe. This is plan.md **Phase 1**. |
| `camera.py` | **`PerCameraState`** — per-camera `sv.ByteTrack` + zone monitor + `CustomizationEngine` + scene context. Converts shared detections → tracks → RawEvents → QueuedAlerts. `build_camera_states()` parses a site config. |
| `gate_pool.py` | **`GatePool`** — async worker threads drain the queue and verify out-of-band, rate-limited, so the slow VLM never stalls detection. |
| `pipeline.py` | **`MultiStreamPipeline`** — shared model, ONE batched detector pass across cameras, per-camera routing. `run_site()` + `--site-config` for a full deployment. |

### Key design principle
Batch the **stateless** detector across cameras; keep the **stateful** tracker +
rules **per camera**. Models load once for all cameras (vs. process-per-camera).

### Benchmark harness: `tools/throughput_bench.py`
Measured on the MPS dev box (NVIDIA will be faster):
- detection (yolov8n@640, batch 16): **~200 fps** → ~40 cams @5fps
- pose: **~180 fps** → ~36 cams
- VideoMAE: **162 ms/clip** — per *event* (gated+cooldown), not per frame
- **Conclusion: detection/pose/VideoMAE are all cheap; the local VLM gate is the
  only real ceiling.** That's why the alert queue (dedup/throttle) matters.

Run:
```bash
python tools/throughput_bench.py --half --batch-sizes 1 8 16
python tools/throughput_bench.py --time-gate --gate-provider ollama --gate-model gemma3:4b  # measures the VLM ceiling
```

### Multi-stream demo (verified)
```bash
python -m cvti.serving.pipeline --site-config configs/site_demo.json --gate-provider mock
```
Result: 2 cameras through one shared model, `batch_sizes={2}`, both cameras fire
`loitering_at_shelf`, async gate confirms 4, `deduped=8`, clean EOF.

### ⚠️ Important limitation of the serving path (read this)
`PerCameraState` currently runs **object detection + ByteTrack + zones + rules +
gate**. It does **NOT** yet run the pose/violence/theft/concealment/VideoMAE
assessments — those still live only in `cvti/detector/core.py`. So today the
multi-stream pipeline handles **zone/presence-based rules**; weapons/violence/etc.
in the multi-cam path is the next wiring job (the seam is marked in
`PerCameraState` — add a batched pose pass and feed `pose_people` in the same way
zones are fed).

### Phase 8 sub-status
- 8.0 benchmark ✅ · 8.1 pipeline ✅ · 8.3 async gate + dedup ✅
- 8.2 frame governor (exists, not tuned) · 8.4 TensorRT/fp16/NVDEC (needs the box)
  · 8.5 per-camera reconnect hardening (hook only)

---

## 4. Backend intelligence — plan.md **Phases 1–4** (all code-only)

### Phase 1 — top-alert → alert queue  (`74b95b6`)
`detector/core.py` verified only `candidate_alerts[0]` (a second concurrent
threat was dropped). Now it enqueues **every** matching candidate into the shared
`AlertQueue` and verifies a bounded burst per frame (`--max-alerts-per-frame`,
default 2). Rest wait in the queue → gate never flooded.

### Phase 2 — always-on critical baseline  (`aa38453`)
A narrow customer config (e.g. loitering-only) could hide a weapon.
`CustomizationEngine(config, baseline_path=...)` merges an **undisableable**
baseline into every evaluation; `has_rules()` makes the gate run whenever any
rule exists (even with no `--config`).
- `configs/baseline_critical_v1.json` — weapon/violence fire today;
  person_down/camera_tampering/fire_smoke activate when those detectors land.
- Flags: `--baseline-config` (default on), `--no-baseline` (dev only).
- **Verified:** a presence-only config still fires `[CONFIRMED] baseline_violence`
  on the violence clip.

### Phase 4 — per-rule evidence-frame selection  (`588e200`)
`cvti/verification/frame_select.py` + a rolling frame buffer in the detector.
Per-rule strategy: weapon = 1 sharpest frame; violence = 4 over the motion peak;
concealment = 3; robbery = 5; zone = 1. Returns selection metadata; the CONFIRMED
log shows `frames=N(strategy)`.
- **Verified:** violence alerts now send `frames=4(motion_peak_span)`.
- **Caveat:** wired in the single-stream detector only (multi-stream sends one
  evidence frame per alert for now).

### Phase 3 — compound threat recipes  (`09b6ca7`)
`_eval_compound` in `customization.py`: a recipe rule with `signals` + `logic` +
`gate_question`. Logic ops: `any` / `all` / `at_least_N` / `one_high_or_two_medium`.
Signal aliases map names→detectors; VideoMAE `signal_type` also matches.
`CandidateAlert.question` threads the recipe's question to the gate.
- `configs/compound_recipes_v1.json` — `armed_robbery`, `violent_theft`.
- **Verified by unit tests** (logic ops; armed_robbery fires on one high;
  violent_theft needs two; VideoMAE `violence_candidate` counts; question threaded).
- **Caveat:** correctly does NOT over-fire on a lone medium signal, so our
  single-signal test clips don't trigger it end-to-end — needs true multi-signal
  footage to exercise live.

---

## 5. How to run / verify (quick reference)

```bash
# unit tests
python -m pytest tests/ -q                       # 57 pass

# single-stream detector (baseline is on by default now)
python -m cvti.cli.detect --source data/test_clips/violence_suspected.mp4 \
  --config configs/all_threats_v1.json --gate-provider mock --max-frames 40

# multi-stream serving (2 virtual cameras)
python -m cvti.serving.pipeline --site-config configs/site_demo.json --gate-provider mock

# zones standalone (loitering rule)
python -m cvti.retail.zones --source data/test_clips/theft_shop_01.mp4 \
  --zones configs/retail_zones.theft_shop_01.json --rules configs/shelf_zones_demo.json

# throughput benchmark (run on the edge GPU for real numbers)
python tools/throughput_bench.py --half --batch-sizes 1 8 16
```

New configs added: `baseline_critical_v1.json`, `shelf_zones_demo.json`,
`site_demo.json`, `compound_recipes_v1.json`.

---

## 6. Known caveats / tech debt
- **Multi-stream path runs only object-detection + zones + rules** — pose/
  violence/theft/concealment/VideoMAE not wired there yet (see §3).
- **Per-rule frame selection is single-stream only.**
- **Compound recipes** proven by unit tests, not yet on multi-signal footage.
- **Local VLM latency unmeasured** — run `--time-gate` on a box with Ollama; it
  sizes how aggressive the alert-queue throttle must be.
- **`sv.ByteTrack` is deprecated** in supervision 0.28 (removed in 0.30) — will
  need migrating.
- Phase 1 "one artifact bundle per verified candidate" is partial (gate saves per
  call; the detector's event-clip recorder still saves one per event).
- A stray `yolov8n.pt` may sit at repo root from an ultralytics auto-download —
  harmless, untracked.

---

## 7. What's left in plan.md
- **Blocked on data/GPU:** Phase 5 (eval set / validated FPR), Phase 6 (video
  fine-tuning), Phase 7 (feedback loop + registry).
- **Code-only, not yet done:** new CV-heuristic rules (person-down/fall,
  camera-tampering, running, crowd, tailgating); wiring the full detector into the
  multi-stream path; per-rule frames in multi-stream; Phase 8.4 GPU optimize.

**Recommended next:** wire the batched pose pass into `PerCameraState` so the
multi-stream path runs the full detector set (not just zones), then measure the
local VLM latency to finalize the gate throttle.

---

## 8. Reviewing the work
```bash
git checkout phase8-throughput
git log --oneline main..phase8-throughput      # the 10 commits above
python -m pytest tests/ -q
```
`main` is untouched by the Phase 8 / Phases 1–4 work — it's all isolated on the
branch, ready for a PR.
