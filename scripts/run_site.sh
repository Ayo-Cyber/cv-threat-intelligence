#!/usr/bin/env bash
# One-box CVTI launcher (plan.md Phase 9 / EPIC D).
#
# Starts the detection pipeline AND the operator inbox together, offline, from a
# single command — what you drop on a site PC and walk away from. Config is env-
# driven so no code edits per site.
#
#   scripts/run_site.sh [site-config.json]
#
# Env (all optional):
#   CVTI_OUT        output dir for events.db + evidence   (default runs/site)
#   CVTI_NOTIFY     console | whatsapp | telegram:.. | webhook:..  (default console)
#   CVTI_GATE       mock | ollama | anthropic | openrouter (default ollama)
#   CVTI_INBOX_PORT operator inbox port                   (default 8080)
#   plus Twilio vars if CVTI_NOTIFY=whatsapp (TWILIO_ACCOUNT_SID/…/WHATSAPP_TO)
#
# Run-on-boot: wrap this in a launchd plist (macOS) or systemd unit (Linux); for
# a quick always-on run:  caffeinate -i nohup scripts/run_site.sh <site>.json &
set -euo pipefail
cd "$(dirname "$0")/.."

SITE="${1:-configs/site_6cam_demo.json}"
OUT="${CVTI_OUT:-runs/site}"
NOTIFY="${CVTI_NOTIFY:-console}"
GATE="${CVTI_GATE:-ollama}"
PORT="${CVTI_INBOX_PORT:-8080}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp
mkdir -p "$OUT"

if pgrep -f 'cvti.serving.pipeline' >/dev/null 2>&1; then
  echo "[cvti] a pipeline is already running (pkill -f cvti.serving to stop it)." >&2
  exit 1
fi

# Operator inbox in the background; stop it when the pipeline exits.
python -m cvti.serving.inbox --db "$OUT/events.db" --port "$PORT" &
INBOX=$!
trap 'kill $INBOX 2>/dev/null || true' EXIT

echo "[cvti] site=$SITE | gate=$GATE | notify=$NOTIFY"
echo "[cvti] operator inbox → http://localhost:$PORT"
echo "[cvti] events + evidence → $OUT/  (Ctrl-C to stop)"

# Pipeline in the foreground; runs until stopped or all streams end.
python -m cvti.serving.pipeline --site-config "$SITE" --gate-provider "$GATE" \
  --notify "$NOTIFY" --output-dir "$OUT" --seconds "${CVTI_SECONDS:-100000}"
