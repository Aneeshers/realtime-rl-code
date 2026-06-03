#!/bin/bash
# Evaluate a trained real-time Tetris gating policy paired with its frozen AlphaZero base planner.

set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT_ROOT=${CKPT_ROOT:-./checkpoints/committed_action/tetris_rt}
GATING_CKPT=${GATING_CKPT:-${CKPT_ROOT}/gating/gating_state_best.pkl}
AZ_CKPT=${AZ_CKPT:-${CKPT_ROOT}/base/training_state_best.pkl}
OUTPUT_DIR=${OUTPUT_DIR:-./eval_outputs/tetris_rt_gating}

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=committed_action python -m jumanji.training.eval_tetris_rt_gating_policy \
  --env_name               TetrisRTKT-v0                \
  --gating_checkpoint_path "${GATING_CKPT}"             \
  --az_checkpoint_path     "${AZ_CKPT}"                 \
  --n_episodes             ${N_EPISODES:-100}           \
  --eval_num_envs          ${EVAL_NUM_ENVS:-100}        \
  --eval_meta_steps        ${EVAL_META_STEPS:-2000}     \
  --timing_batch           ${TIMING_BATCH:-1}           \
  --timing_reps            ${TIMING_REPS:-50}           \
  --seed                   ${SEED:-42}                  \
  --wandb_project          ${WANDB_PROJECT:-tetris_rt_kt_gating_eval} \
  --output_dir             "${OUTPUT_DIR}"
