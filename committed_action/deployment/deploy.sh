#!/bin/bash
# Two-GPU real-time deployment for committed-action environments.
# GPU 0 runs the environment + reflex committed actions; GPU 1 runs the MCTS planner.
#
# Reproduces Section 6 of the paper across all three committed-action games.
#
# Required env: a CUDA install with two visible GPUs.
#
# Usage:
#   GAME=tetris FPS=9 ./deployment/deploy.sh
#   GAME=pacman FPS=9 ./deployment/deploy.sh
#   GAME=snake  FPS=9 ./deployment/deploy.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

GAME=${GAME:-tetris}                   # tetris | pacman | snake
FPS=${FPS:-9}                          # paper sweeps 8..12
GPU_TYPE=${GPU_TYPE:-h100}
N_EPISODES=${N_EPISODES:-100}
SEED=${SEED:-42}

CKPT_ROOT=${CKPT_ROOT:-./checkpoints/committed_action}

case "${GAME}" in
  tetris)
    AZ_CKPT=${AZ_CKPT:-${CKPT_ROOT}/tetris_rt/base/k1/training_state_best.pkl}
    GATING_CKPT=${GATING_CKPT:-${CKPT_ROOT}/tetris_rt/gating/gating_state_best.pkl} ;;
  pacman)
    AZ_CKPT=${AZ_CKPT:-${CKPT_ROOT}/pacman/base/k1/training_state_best.pkl}
    GATING_CKPT=${GATING_CKPT:-${CKPT_ROOT}/pacman/gating/gating_state_best.pkl} ;;
  snake)
    AZ_CKPT=${AZ_CKPT:-${CKPT_ROOT}/snake/base/k3/training_state_best.pkl}
    GATING_CKPT=${GATING_CKPT:-${CKPT_ROOT}/snake/gating/gating_state_best.pkl} ;;
  *)
    echo "Unknown GAME=${GAME}. Use tetris, pacman, or snake." >&2; exit 1 ;;
esac

PYTHONPATH=committed_action python -m jumanji.training.deploy_tetris_rt_realtime \
  --game                   "${GAME}"        \
  --gpu_type               "${GPU_TYPE}"    \
  --fps                    "${FPS}"         \
  --n_episodes             "${N_EPISODES}"  \
  --az_checkpoint_path     "${AZ_CKPT}"     \
  --gating_checkpoint_path "${GATING_CKPT}" \
  --wandb_project          ${WANDB_PROJECT:-rt_deploy_realtime} \
  --seed                   "${SEED}"
