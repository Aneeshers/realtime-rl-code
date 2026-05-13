#!/bin/bash
# Train the PPO gating policy on top of a frozen real-time Tetris (TetrisRTKT) AlphaZero planner.

set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT_ROOT=${CKPT_ROOT:-./checkpoints/committed_action/tetris_rt}
AZ_CHECKPOINT_PATH=${AZ_CHECKPOINT_PATH:-${CKPT_ROOT}/base/k1/training_state_best.pkl}
GATING_CHECKPOINT_DIR=${GATING_CHECKPOINT_DIR:-${CKPT_ROOT}/gating}
SEED=${SEED:-1}

PYTHONPATH=committed_action python -m jumanji.training.train_tetris_rt_gating_ppo \
  --env_name              TetrisRTKT-v0                   \
  --num_envs              ${NUM_ENVS:-32}                 \
  --eval_num_envs         ${EVAL_NUM_ENVS:-16}            \
  --meta_steps            ${META_STEPS:-384}              \
  --ppo_epochs            ${PPO_EPOCHS:-4}                \
  --num_minibatches       ${NUM_MINIBATCHES:-16}          \
  --gamma                 ${GAMMA:-0.99}                  \
  --gae_lambda            ${GAE_LAMBDA:-0.95}             \
  --lr                    ${LR:-3e-4}                     \
  --entropy_coef          ${ENTROPY_COEF:-0.05}           \
  --epsilon_clip          ${EPSILON_CLIP:-0.2}            \
  --value_loss_coef       ${VALUE_LOSS_COEF:-0.5}         \
  --num_epochs            ${NUM_EPOCHS:-3000}             \
  --eval_every            ${EVAL_EVERY:-5}                \
  --reward_mode           ${REWARD_MODE:-raw}             \
  --sim_options           ${SIM_OPTIONS:-32 64 96 128}    \
  --az_checkpoint_path    "${AZ_CHECKPOINT_PATH}"         \
  --gating_checkpoint_dir "${GATING_CHECKPOINT_DIR}"      \
  --wandb_project         ${WANDB_PROJECT:-tetris_rt_kt_gating_ppo} \
  --seed                  ${SEED}
