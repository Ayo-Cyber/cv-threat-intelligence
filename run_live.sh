#!/usr/bin/env bash
# Argus on a REAL live YouTube camera.
#   ./run_live.sh                 # default: EarthCam Dublin street cam
#   ./run_live.sh <youtube_id>    # any currently-live cam
#
# YouTube live URLs are time-limited, so this resolves a fresh HLS URL each run.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

ID="${1:-3nyPER2kzqk}"   # EarthCam Dublin (pedestrians + vehicles)
echo "[argus] resolving live stream for youtube id $ID ..."
URL=$(yt-dlp -g --extractor-args "youtube:player_client=android" \
        -f "best[height<=720]/best" "https://www.youtube.com/watch?v=$ID" 2>/dev/null | head -1)
if [ -z "$URL" ]; then
  echo "[argus] could not resolve a stream. Try another id, or: yt-dlp -U (update)."; exit 1
fi
echo "[argus] got a live HLS URL (${#URL} chars)."

# Write a live-camera config. Street-relevant detectors: panic-running, crowd
# formation, fall — all still TrueSight-verified, so a calm street stays quiet.
python3 - "$URL" <<'PY'
import json, sys
url = sys.argv[1]
cfg = {
  "name": "Live Camera", "notify": "console", "configured": True,
  "cameras": [{
    "id": "Live Camera", "source": url, "config": "configs/manufacturing_hse_v1.json",
    "environment_type": "public street",
    "scene_description": "A public street/plaza with pedestrians and vehicles; a person running may signal panic, and a tight fast-forming crowd may signal an incident.",
    "running": True, "running_min_speed_ratio": 0.08, "running_min_frames": 3,
    "crowd_formation": True, "crowd_min_people": 5, "crowd_min_frames": 3,
    "fall": True
  }]
}
open("configs/live_camera.json", "w").write(json.dumps(cfg, indent=2))
print("[argus] wrote configs/live_camera.json")
PY

exec ./run_demo.sh configs/live_camera.json
