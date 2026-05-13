#!/bin/bash
# Train an AlphaZero base planner for real-time Pac-Man (PacManKT-v1).
# Iterate K=1..4 to match the paper's set of action delays.
#
# Override knobs:
#   CKPT_ROOT  destination for training_state_*.pkl  (default: ./checkpoints/committed_action/pacman/base)
#   K          action delay (default: 1)
#   EPOCHS     number of training epochs (default: 25)
#   WANDB_PROJECT, WANDB_ENTITY  optional W&B logging

set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT_ROOT=${CKPT_ROOT:-./checkpoints/committed_action/pacman/base}
K=${K:-1}
EPOCHS=${EPOCHS:-25}
LOGGER_TYPE=${LOGGER_TYPE:-terminal}

mkdir -p "${CKPT_ROOT}/k${K}"

PYTHONPATH=committed_action python -m jumanji.training.train \
  env=pac_man_k_t \
  agent=gumbel_alphazero \
  logger.type=${LOGGER_TYPE} \
  logger.project=${WANDB_PROJECT:-pacman_az} \
  logger.checkpoint_dir="${CKPT_ROOT}/k${K}" \
  env.az.action_delay=${K} \
  env.training.num_epochs=${EPOCHS}
