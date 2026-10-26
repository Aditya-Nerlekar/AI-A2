#!/usr/bin/env bash
# Reproduce every NotifyRL result end to end (CPU only).
#   bash scripts/run_all.sh          full pipeline (all figures + tables)
#   bash scripts/run_all.sh --quick  smoke run: fewer seeds/episodes (~1/4 of the time)
set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES=400; FT_EPISODES=100
MAIN_SEEDS="0 1 2 3 4"; SMALL_SEEDS="0 1 2"; RW_SEEDS="0 1"; EVAL_EPISODES=100
if [[ "${1:-}" == "--quick" ]]; then
  EPISODES=150; FT_EPISODES=50
  MAIN_SEEDS="0 1"; SMALL_SEEDS="0"; RW_SEEDS="0"; EVAL_EPISODES=50
fi
T0=$(date +%s)
log() { echo; echo "=== [$(( $(date +%s) - T0 ))s] $* ==="; }

log "1/7 unit tests"
python -m pytest -q tests

log "2/7 DQN variant ablation (vanilla / double / dueling+double)"
for v in vanilla double dueling_double; do
  python -m notifyrl.train --variant "$v" --seeds $MAIN_SEEDS --episodes "$EPISODES" --tag "$v"
done

log "3/7 partial observability: 4-frame observation stack"
python -m notifyrl.train --variant dueling_double --frame-stack 4 --seeds $SMALL_SEEDS --episodes "$EPISODES" --tag fs4

log "4/7 reward-weight sensitivity"
python -m notifyrl.train --reward w_interrupt=0.5 --seeds $RW_SEEDS --episodes "$EPISODES" --tag rw_interrupt_x0.5
python -m notifyrl.train --reward w_interrupt=2.0 --seeds $RW_SEEDS --episodes "$EPISODES" --tag rw_interrupt_x2
python -m notifyrl.train --reward w_miss=1.5     --seeds $RW_SEEDS --episodes "$EPISODES" --tag rw_miss_x0.5
python -m notifyrl.train --reward w_miss=6.0     --seeds $RW_SEEDS --episodes "$EPISODES" --tag rw_miss_x2

log "5/7 generalisation: student -> night-shift worker"
python -m notifyrl.train --profile night_shift --seeds $SMALL_SEEDS --episodes "$EPISODES" --tag night_native
python -m notifyrl.train --profile night_shift --seeds $SMALL_SEEDS --episodes "$FT_EPISODES" \
  --init-from "models/dueling_double_s{seed}.pt" --eps-start 0.3 --tag finetune_night

log "6/7 held-out evaluation (seeds 10000+)"
python -m notifyrl.evaluate --episodes "$EVAL_EPISODES" --models \
  "models/vanilla_s*.pt" "models/double_s*.pt" "models/dueling_double_s*.pt" "models/fs4_s*.pt" "models/rw_*_s*.pt"
python -m notifyrl.evaluate --episodes "$EVAL_EPISODES" --profile night_shift --tag night --models \
  "models/dueling_double_s*.pt" "models/finetune_night_s*.pt" "models/night_native_s*.pt"

log "7/7 figures and LaTeX tables"
python -m notifyrl.plots

log "done in $(( $(date +%s) - T0 ))s  -> results/, models/, report/generated/"
