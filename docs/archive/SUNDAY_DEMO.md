# Sunday full-product demo — runbook

The whole product in one app: open → cameras → local AI gate → live wall →
a threat happens → alert appears with evidence + reasoning → phone notification →
operator marks it true/false.

> Run **from source** on this Mac (not the .dmg). The detection engine needs the
> full Python env (torch/ultralytics/VideoMAE) and the demo clips in
> `data/test_clips/`, which the lean app bundle doesn't carry.

## Pre-flight (do this Saturday, and again ~30 min before)

```bash
cd /Users/atunraseayomide/Documents/GitHub/cv-threat-intelligence
source .venv/bin/activate

# 1) Local VLM up + model present
ollama serve &                 # if not already running
ollama list | grep gemma3:4b   # must show the model

# 2) sanity: engine + gate chain (headless, ~40s)
python -m cvti.serving.pipeline --site-config configs/site_demo_live.json \
  --gate-provider ollama --gate-model gemma3:4b --notify console \
  --output-dir runs/demo --seconds 45 --gate-drain 40
# expect: "[CONFIRMED] ... shoplifting / video_theft_candidate" then "persisted N event(s)"
```

## Launch the app (the demo itself)

```bash
# LEAN — recommended on 16-18GB Macs (3 cams; ~7-8GB, cooler machine)
python -m cvti.app.shell --site-config configs/site_demo_lite.json --db runs/demo/events.db

# FULL — 5 cams, heavier (needs headroom; close Brave/other apps first)
python -m cvti.app.shell --site-config configs/site_demo_live.json --db runs/demo/events.db
```

> **Memory:** Gemma (~5GB) + the detection stack + your other apps can hit ~13GB.
> On an 18GB Mac use the **lite** (3-cam) config and quit Brave/extra apps before
> the demo. The engine also runs lean now (4 fps, 512px). Same story, cooler box.

## Demo flow (5 beats)

1. **It's a product.** Window opens on the site "Demo Store". Walk the left nav:
   Cameras (6 configured), Live, Alerts, System.
2. **Live wall.** Click **Live** — 6 feeds streaming (looping demo footage), each
   with a LIVE indicator. "These are the cameras the customer already owns."
3. **Turn it on.** Click **▶ Start monitoring** (top right). Status flips to
   "Monitoring". The engine loads (~25–35s) — narrate the architecture meanwhile:
   batched detection → per-camera threat models (concealment + fine-tuned
   VideoMAE) → **local Gemma 3 4B gate double-checks every alert** → nothing
   leaves the box.
4. **A threat is caught.** Click **Alerts**. Within ~35s of starting, confirmed
   alerts appear **live** (auto-refresh): `shoplifting`, `video_theft_candidate`
   on the aisle cams. Open one — evidence frames, the gate's plain-English reason,
   confidence. Point out the normal cams did **not** fire (targeting + low false
   alarms). If WhatsApp is set, **your phone buzzes** at this moment.
5. **Human in the loop.** Mark an alert **True threat** / **False alarm** —
   "every label retrains this site's model."

To reset between runs: **Stop monitoring**, then `rm -rf runs/demo && mkdir runs/demo`, relaunch.

## WhatsApp (drop-in when your teammate sends the creds)

The demo site is set to `notify: "console,whatsapp"` — console always fires; WhatsApp
activates the moment these env vars exist (set them in the terminal *before* launching):

```bash
export TWILIO_ACCOUNT_SID=ACxxxx…
export TWILIO_AUTH_TOKEN=xxxx…
export TWILIO_WHATSAPP_FROM=whatsapp:+14155238886   # Twilio sandbox or your number
export WHATSAPP_TO=whatsapp:+234…                   # your phone (joined to the sandbox)
```
No creds = it silently falls back to console; the demo still works.

## If something misbehaves (fallbacks)

- **Alerts not appearing:** give it 40s (model load + first ~12s Gemma verdict).
  Check `runs/demo/monitor.log`.
- **"Start monitoring" does nothing:** run the engine in a separate terminal with
  the pre-flight pipeline command above (`--seconds 100000`); the app still reads
  `runs/demo/events.db` and auto-refreshes.
- **Ollama down:** `ollama serve`. Worst case set `--gate-provider mock` (alerts
  still flow; verification is stubbed — only as a last resort).
- **Gemma slow on this Mac:** ~12s/verdict is expected (MPS). It's the honest
  edge-box number; the production box has an NVIDIA GPU.
