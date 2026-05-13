#!/bin/bash
# Train an AlphaZero base planner for Snake (SnakeKT-v1).
# K-step delay: MCTS uses argmax(masked_policy_logits) for the K-1 committed steps.

set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT_ROOT=${CKPT_ROOT:-./checkpoints/committed_action/snake/base}
K=${K:-1}
LOGGER_TYPE=${LOGGER_TYPE:-terminal}

mkdir -p "${CKPT_ROOT}/k${K}"

PYTHONPATH=committed_action python -m jumanji.training.train \
  env=snake_k_t \
  agent=gumbel_alphazero \
  logger.type=${LOGGER_TYPE} \
  logger.project=${WANDB_PROJECT:-snake_kt_az} \
  logger.checkpoint_dir="${CKPT_ROOT}/k${K}" \
  env.az.action_delay=${K}
