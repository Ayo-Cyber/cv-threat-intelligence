#!/usr/bin/env bash
# Durable CamNuvem video-model training launcher (plan.md Phase 6).
#
# Always runs the CURRENT code with the good defaults, survives closing the
# terminal (nohup) and the Mac sleeping (caffeinate), and logs to a timestamped
# file. Fixes the two things that bit us: stale shell-history commands and the
# job dying on session end.
#
# Usage:
#   scripts/train_video.sh                       # good default recipe (frozen head)
#   scripts/train_video.sh --unfreeze-last-block # push past the frozen-head ceiling
#   scripts/train_video.sh --backend x3d ...     # any trainer flags pass through
#
# Watch:  tail -f <printed log>       Stop:  pkill -f video_finetune
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p runs/video_finetune
TS="$(date +%Y%m%d-%H%M)"
LOG="runs/video_finetune/train-${TS}.log"

# Default recipe = what actually worked (frozen head, higher LR, class weights on,
# balanced-acc early stop). Pass any args to override / extend.
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(--backend videomae --freeze-backbone --lr 1e-3 --epochs 40 --patience 8 --batch 2)
fi

echo "[train] recipe: ${ARGS[*]}"
echo "[train] log:    $LOG"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp \
  caffeinate -i nohup python -m cvti.training.video_finetune "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
echo "[train] started (pid $PID) — safe to close this terminal"
echo "[train] watch:   tail -f $LOG"
echo "[train] metrics: cat runs/video_finetune/videomae/metrics.json   (watch balanced_acc)"
echo "[train] stop:    pkill -f video_finetune"
