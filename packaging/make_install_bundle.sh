#!/usr/bin/env bash
# Wrap the full Argus system (source + demo videos + models + configs + launchers)
# into ONE distributable zip: dist/argus-install.zip
#
# This is the LIVE system (real detection), not the lean viewer .app — so it needs
# Python + Ollama on the target machine (see the generated INSTALL.md). The demo
# videos are bundled so it runs offline out of the box; switch to live cams in-app.
#
#   ./packaging/make_install_bundle.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
STAGE="dist/argus-install"
rm -rf "$STAGE"; mkdir -p "$STAGE"

echo "[bundle] copying source + configs + launchers…"
# source, configs, launchers, deps — everything needed to run
for item in cvti configs run_demo.sh run_live.sh requirements.txt pyproject.toml; do
  [ -e "$item" ] && cp -R "$item" "$STAGE/"
done

echo "[bundle] copying models (YOLO + VideoMAE)…"
mkdir -p "$STAGE/models" "$STAGE/runs/video_finetune"
cp models/yolov8n.pt models/yolov8n-pose.pt "$STAGE/models/" 2>/dev/null || true
[ -d runs/video_finetune/videomae ] && cp -R runs/video_finetune/videomae "$STAGE/runs/video_finetune/"

echo "[bundle] copying demo videos (~58M)…"
mkdir -p "$STAGE/data"
cp -R data/test_clips "$STAGE/data/" 2>/dev/null || true
cp -R data/hse_demo   "$STAGE/data/" 2>/dev/null || true

cat > "$STAGE/INSTALL.md" <<'MD'
# Argus — Install & Run

The full live system (real on-device detection). The demo videos are bundled, so
it works offline out of the box; use the in-app **Feed** toggle to switch between
**Demo Videos** and **Live EarthCams**.

## Requirements
- **Python 3.9+** (3.10+ recommended)
- **Ollama** — the local AI verifier (TrueSight). https://ollama.com/download

## One-time setup
1. Install Ollama, then pull the model:

       ollama pull gemma3:4b

2. Create the Python environment (from this folder):

       python3 -m venv .venv
       source .venv/bin/activate           # Windows: .venv\Scripts\activate
       pip install -r requirements.txt
       pip install yt-dlp                   # only needed for Live EarthCams

## Run
- **Demo videos:**  `./run_demo.sh configs/deluxe_demo.json`
- **Live cameras:** `./run_live.sh`

The app window opens automatically. Inside it, the **Feed** toggle (top bar) flips
between Demo and Live at any time.

> Windows: the `.sh` launchers need **Git Bash** (or WSL). Otherwise run the two
> commands they contain directly — start `python -m cvti.serving.pipeline …` then
> `python -m cvti.app.shell …` (see the top of run_demo.sh for the exact flags).

## Your own cameras
Point a camera's `source` at its RTSP URL, e.g.
`"source": "rtsp://user:pass@192.168.0.50:554/stream"`, in a config under `configs/`.
MD

echo "[bundle] zipping…"
cd dist
zip -rq argus-install.zip argus-install
cd "$ROOT"
SZ=$(du -sh dist/argus-install.zip | cut -f1)
echo "[bundle] done -> dist/argus-install.zip ($SZ)"
