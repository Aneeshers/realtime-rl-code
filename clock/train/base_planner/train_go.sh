#!/bin/bash
# Train an AlphaZero base planner for 9x9 Go (self-play).

set -euo pipefail
cd "$(dirname "$0")/../../.."

NUM_SIMULATIONS=${NUM_SIMULATIONS:-32}
SEED=${SEED:-0}
SAVE_DIR=${SAVE_DIR:-./checkpoints/clock/go/base}
mkdir -p "${SAVE_DIR}"

PYTHONPATH=clock/networks:. python clock/train/base_planner/train_go.py \
  --num_simulations "${NUM_SIMULATIONS}" \
  --seed            "${SEED}"
