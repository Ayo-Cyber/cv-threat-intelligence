#!/usr/bin/env bash
# Argus on REAL live YouTube cameras — a multi-feed live dashboard.
#   ./run_live.sh                         # curated set of live cams
#   ./run_live.sh id1 id2 id3 ...         # your own list of youtube ids
#
# YouTube live URLs are time-limited, so this resolves a fresh HLS URL per feed.
# Street cams use running-only detection (panic) so they stay calm — crowd/fall
# spam on a busy street is a context mismatch (see chat), not useful here.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

# id|Friendly Name  — curated, verified-live public cams (pass your own ids to override)
DEFAULTS=(
  "3nyPER2kzqk|Dublin Street"
  "OUBslrCqREs|LaGuardia Airport"
  "mt7uE-n0YPI|Venice Canal"
  "201MyYKtXaQ|Heathrow Airport"
)
if [ "$#" -gt 0 ]; then
  DEFAULTS=(); for id in "$@"; do DEFAULTS+=("$id|$id"); done
fi

echo "[argus] resolving ${#DEFAULTS[@]} live feeds ..."
: > /tmp/argus_feeds.tsv
for entry in "${DEFAULTS[@]}"; do
  id="${entry%%|*}"; name="${entry#*|}"
  url=$(yt-dlp -g --extractor-args "youtube:player_client=android" \
          -f "best[height<=720]/best" "https://www.youtube.com/watch?v=$id" 2>/dev/null | head -1)
  if [ -n "$url" ]; then echo "  ✓ $name"; printf '%s\t%s\n' "$name" "$url" >> /tmp/argus_feeds.tsv
  else echo "  ✗ $name (not live / could not resolve — skipping)"; fi
done
[ -s /tmp/argus_feeds.tsv ] || { echo "[argus] no feeds resolved. Try 'yt-dlp -U' or other ids."; exit 1; }

python3 - <<'PY'
import json
cams=[]
for line in open("/tmp/argus_feeds.tsv"):
    name,url=line.rstrip("\n").split("\t",1)
    cams.append({"id":name,"source":url,"config":"configs/manufacturing_hse_v1.json",
        "environment_type":"public area",
        "scene_description":"A public street/plaza/terminal with pedestrians and vehicles; a person running may signal panic or an incident.",
        "running":True,"running_min_speed_ratio":0.08,"running_min_frames":3})
json.dump({"name":"Live Dashboard","notify":"console","configured":True,"cameras":cams},
          open("configs/live_camera.json","w"),indent=2)
print(f"[argus] wrote configs/live_camera.json with {len(cams)} feed(s)")
PY

exec ./run_demo.sh configs/live_camera.json
