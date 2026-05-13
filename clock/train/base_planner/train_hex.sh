#!/bin/bash
# Train an AlphaZero base planner for 11x11 Hex (self-play).
# Saves checkpoints under {save_dir}/nsim_{N}/{seed}/{iter}.ckpt.
#
# Override knobs:
#   NUM_SIMULATIONS  MCTS sims per move during self-play (default: 32)
#   SEED             random seed (default: 0)
#   SAVE_DIR         destination root (default: ./checkpoints/clock/hex/base)
#                    NOTE: train_hex.py reads this from its Pydantic default; pass
#                    via --save_dir override (added below) or edit the file.

set -euo pipefail
cd "$(dirname "$0")/../../.."

NUM_SIMULATIONS=${NUM_SIMULATIONS:-32}
SEED=${SEED:-0}
SAVE_DIR=${SAVE_DIR:-./checkpoints/clock/hex/base}
mkdir -p "${SAVE_DIR}"

PYTHONPATH=clock/networks:. python clock/train/base_planner/train_hex.py \
  --num_simulations "${NUM_SIMULATIONS}" \
  --seed            "${SEED}"
