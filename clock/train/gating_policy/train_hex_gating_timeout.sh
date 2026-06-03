#!/bin/bash
# Train the GRU-PPO gating policy for Speed Hex with a strict timeout penalty
# (appendix experiment: running out of clock = immediate loss).

set -euo pipefail
cd "$(dirname "$0")/../../.."

export SPEED_ENV=${SPEED_ENV:-speed_hex_timeout}
export ENV_KWARGS=${ENV_KWARGS:-}
export CKPT_ROOT=${CKPT_ROOT:-./checkpoints/clock/hex/base}
export PRETRAINED_NSIM=${PRETRAINED_NSIM:-32}
export ITER_FILE=${ITER_FILE:-base_planner.ckpt}
export GATE_CKPT_ROOT=${GATE_CKPT_ROOT:-./checkpoints/clock/hex/gating_timeout}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_MODE=${WANDB_MODE:-disabled}
SEED=${SEED:-0}

mkdir -p "${GATE_CKPT_ROOT}"

PYTHONPATH=clock/networks:clock/envs:. python clock/train/gating_policy/train_hex_gating_timeout.py \
  --env       "${SPEED_ENV}" \
  --env_kwargs "${ENV_KWARGS}" \
  --seed      "${SEED}"
