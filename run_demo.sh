#!/usr/bin/env bash
# Argus — one-command launcher. Starts Ollama + engine + app, waits for each,
# and cleans everything up on Ctrl+C.
#
#   ./run_demo.sh                         # 4-camera demo (default)
#   ./run_demo.sh configs/site_theft_demo.json    # 6-camera theft wall
#   ./run_demo.sh configs/site_safety_demo.json   # fire / fall / crowd
#
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-configs/site_demo_4cam.json}"
DB="runs/live/events.db"

# --- venv ---
if [ -d ".venv" ]; then source .venv/bin/activate; fi

# --- track children so we can kill them all on exit ---
PIDS=()
cleanup() {
  echo ""
  echo "[argus] shutting down…"
  pkill -f "cvti.serving.pipeline" 2>/dev/null || true
  pkill -f "cvti.app.shell" 2>/dev/null || true
  # leave ollama running (shared); uncomment to also stop it:
  # pkill -f "ollama" 2>/dev/null || true
  echo "[argus] done."
}
trap cleanup EXIT INT TERM

# --- 1. Ollama (TrueSight model host) ---
if ! curl -s --max-time 3 http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "[argus] starting Ollama…"
  nohup ollama serve >/tmp/argus_ollama.log 2>&1 &
  for i in $(seq 1 20); do
    curl -s --max-time 2 http://localhost:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
fi
echo "[argus] Ollama ready."

# --- fresh run ---
rm -rf runs/live && mkdir -p runs/live

# --- 2. Engine (detection + gate) ---
echo "[argus] starting engine on ${CONFIG} …"
PYTHONUNBUFFERED=1 python -u -m cvti.serving.pipeline \
  --site-config "$CONFIG" \
  --gate-provider ollama --gate-model gemma3:4b \
  --notify console --output-dir runs/live \
  --target-fps 4 --imgsz 512 --seconds 100000 --gate-drain 60 \
  > runs/live/run.log 2>&1 &
PIDS+=($!)

# wait until models are loaded (or 120s)
for i in $(seq 1 24); do
  grep -q "camera(s)" runs/live/run.log 2>/dev/null && { echo "[argus] engine live."; break; }
  sleep 5
done

# --- 3. App (Argus console) ---
echo "[argus] opening the console…"
python -m cvti.app.shell --site-config "$CONFIG" --db "$DB" &
PIDS+=($!)

echo ""
echo "[argus] running. Engine log: runs/live/run.log"
echo "[argus] press Ctrl+C here to stop everything."
wait
