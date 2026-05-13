#!/bin/bash
# Cross-evaluate the real-time Tetris (TetrisRTKT) AlphaZero base planner across (K_eval, sims).

set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT_ROOT=${CKPT_ROOT:-./checkpoints/committed_action/tetris_rt}
K_MODEL=${K_MODEL:-1}
BASE_MODEL_DIR=${BASE_MODEL_DIR:-${CKPT_ROOT}/base/k}

PYTHONPATH=committed_action python -m jumanji.training.eval_tetris_rt_kt_cross \
  --k_model          ${K_MODEL}                          \
  --base_model_dir   "${BASE_MODEL_DIR}"                 \
  --checkpoint_name  ${CHECKPOINT_NAME:-training_state_best.pkl} \
  --k_eval_list      ${K_EVAL_LIST:-1 2 3 4}             \
  --sims_list        ${SIMS_LIST:-32 64 96 128}          \
  --eval_batch_size  ${EVAL_BATCH_SIZE:-100}             \
  --wandb_project    ${WANDB_PROJECT:-tetris_rt_kt_cross_eval} \
  --seed             ${SEED:-42}
