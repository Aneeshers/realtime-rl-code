#!/bin/bash
# Cross-evaluate the Pac-Man (PacManKT) AlphaZero base planner across (K_eval, sims).
# Reproduces the always-K fixed-budget rows in Table 1.

set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT_ROOT=${CKPT_ROOT:-./checkpoints/committed_action/pacman}
K_MODEL=${K_MODEL:-1}
BASE_MODEL_DIR=${BASE_MODEL_DIR:-${CKPT_ROOT}/base/k}

PYTHONPATH=committed_action python -m jumanji.training.eval_pacman_cross \
  --k_model          ${K_MODEL}                       \
  --k_eval_list      ${K_EVAL_LIST:-1 2 3 4}          \
  --sims_list        ${SIMS_LIST:-32 64 96 128}       \
  --eval_batch_size  ${EVAL_BATCH_SIZE:-100}          \
  --wandb_project    ${WANDB_PROJECT:-pacman_cross_eval} \
  --seed             ${SEED:-42}
