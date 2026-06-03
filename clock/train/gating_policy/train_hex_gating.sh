#!/bin/bash
# Train the GRU-PPO gating policy for Speed Hex on top of a frozen AlphaZero base.
#
# Required:
#   CKPT_ROOT             root of the frozen base-planner checkpoints
#   PRETRAINED_NSIM       sims-per-move used for the chosen base ckpt (default 32)
#   ITER_FILE             checkpoint filename, e.g. base_planner.ckpt
#   GATE_CKPT_ROOT        where this run's gate ckpts will be written

set -euo pipefail
cd "$(dirname "$0")/../../.."

export SPEED_ENV=${SPEED_ENV:-speed_hex}
export ENV_KWARGS=${ENV_KWARGS:-}
export CKPT_ROOT=${CKPT_ROOT:-./checkpoints/clock/hex/base}
export PRETRAINED_NSIM=${PRETRAINED_NSIM:-32}
export ITER_FILE=${ITER_FILE:-base_planner.ckpt}
export GATE_CKPT_ROOT=${GATE_CKPT_ROOT:-./checkpoints/clock/hex/gating}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_MODE=${WANDB_MODE:-disabled}
SEED=${SEED:-1}

mkdir -p "${GATE_CKPT_ROOT}"

PYTHONPATH=clock/networks:clock/envs:. python clock/train/gating_policy/train_hex_gating.py \
  --env       "${SPEED_ENV}" \
  --env_kwargs "${ENV_KWARGS}" \
  --seed      "${SEED}"
