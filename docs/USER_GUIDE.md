# CV Threat Intelligence — User Guide

A desktop app that watches a camera or video feed, detects threat events, and uses a
local AI model to confirm real alerts (and reject false ones). Everything runs on your
machine — no cloud account or API key required for the local mode.

---

## 1. Install (macOS)

1. Open **`CVTI-0.1.0-mac.dmg`**.
2. Drag **CVTI** into your **Applications** folder.
3. First launch: right-click **CVTI → Open**, then confirm **Open** in the dialog.
   (The app isn't notarized, so a plain double-click may be blocked the first time.)
4. When prompted, allow **Camera** access if you plan to use a live webcam.

---

## 2. First run — one-time model download

The app confirms alerts with a local vision model (**Gemma 3**, `gemma3:4b-it-qat`).
The AI *engine* (Ollama) is already inside the app; the **model file (~3.3 GB) downloads
once** on first use.

1. In the top toolbar set **Gate** to **`local`**.
2. Leave **Model** on **`gemma3:4b-it-qat`** (the default).
3. Press **▶ Start**. The app will:
   - start its built-in AI engine, and
   - ask to download the model — click **Yes**.
   Progress shows in the status bar at the bottom. This needs internet **once**; after
   that it works fully offline.

> Low on memory? Pick **`moondream`** instead — it's smaller (~2 GB) but less accurate.
> See [OFFLINE_VLM.md](OFFLINE_VLM.md) for the trade-offs.

---

## 3. Running a detection

1. **Source** (top-left): enter one of
   - `0` — the built-in/USB webcam
   - `rtsp://…` — an IP camera stream
   - a path to a video file, e.g. `data/test_clips/theft_shop_01.mp4`
2. **Gate:** `local` (offline AI), `anthropic` (cloud, needs a key), or `mock` (testing —
   auto-confirms everything, no AI).
3. Open the **Rules** tab and pick a rule config (e.g. `retail_pipeline_v1.json`).
   Optionally choose a **Zones** file to enable shelf/area monitoring.
4. Press **▶ Start**. The live feed shows on the left with boxes over detected people;
   confirmed alerts appear in the **Alerts** tab on the right.
5. Press **■ Stop** to end.

---

## 4. The tabs

- **Alerts** — confirmed threat events, with time, rule, confidence, and the AI's reason.
- **Agent Map** — point it at a camera/clip and press *Run Mapper* to auto-describe the
  scene (environment type, zones). Choose provider `ollama` to run it locally. The result
  feeds context into the detector.
- **Rules** — load and edit the JSON rules that decide what counts as an alert. Edit, then
  **Apply Config**. Restart a running detection for changes to take effect.

---

## 5. How verification works (why it's accurate)

The camera pipeline (YOLO pose + tracking) flags *candidate* events cheaply on every
frame. Only when a rule matches does it send that single frame to the local vision model,
which answers "is this a genuine threat?" This keeps the AI cost low and cuts false
alarms — the model runs a few times per event, not on every frame.

---

## 6. Troubleshooting

- **"Ollama not running" / local gate fails to start** — the built-in engine didn't
  launch. Quit and reopen the app. If it persists, install Ollama from
  https://ollama.com; the app will use it automatically.
- **Model download stalls** — check internet; press Stop then Start to retry. The partial
  download resumes.
- **Slow / high memory** — Gemma 3 uses ~3–4 GB while verifying. Switch **Model** to
  `moondream` for a lighter footprint.
- **No camera image** — confirm the Source value, and that macOS granted Camera
  permission (System Settings → Privacy & Security → Camera).
- **First launch blocked by macOS** — right-click the app → **Open** (step 1.3).

---

## 7. For developers

Run from source instead of the app:

```bash
pip install -r requirements.txt
python -m cvti.app.main            # desktop app
# or the CLI pipeline:
python -m cvti.pipelines.retail_pipeline \
  --source data/test_clips/theft_shop_01.mp4 \
  --config configs/retail_pipeline_v1.json \
  --gate-provider local --gate-model gemma3:4b-it-qat
```

Build the DMG: `bash scripts/build_mac.sh` (fetches the Ollama runtime, runs PyInstaller,
produces `dist/CVTI-0.1.0-mac.dmg`). See [OFFLINE_VLM.md](OFFLINE_VLM.md) for packaging
details.
