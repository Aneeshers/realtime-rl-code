#!/bin/bash
# Evaluate a trained Speed Go gating policy against fixed-budget and heuristic baselines.

set -euo pipefail
cd "$(dirname "$0")/../../.."

ENV=${ENV:-speed_go}
TIMES=${TIMES:-300 1200 2300 3500 4100}
SEEDS=${SEEDS:-1}
SIM_OPTIONS=${SIM_OPTIONS:-16,32,64,96}
OPPONENTS=${OPPONENTS:-0,16,32,64,96,random}
NUM_GAMES=${NUM_GAMES:-100}
CKPT_ROOT=${CKPT_ROOT:-./checkpoints/clock/go/base}
GATE_ROOT=${GATE_ROOT:-./checkpoints/clock/go/gating}
ITER_FILE=${ITER_FILE:-000600.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-./eval_outputs/clock/go_gating}

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=clock/networks:clock/envs:. python clock/eval/gating_policy/eval_go_gating.py \
  --env             "${ENV}"            \
  --times           ${TIMES}            \
  --seeds           "${SEEDS}"          \
  --sim_options     "${SIM_OPTIONS}"    \
  --opponents       "${OPPONENTS}"      \
  --num_games       "${NUM_GAMES}"      \
  --ckpt_root       "${CKPT_ROOT}"      \
  --gate_root       "${GATE_ROOT}"      \
  --iter_file       "${ITER_FILE}"      \
  --output_dir      "${OUTPUT_DIR}"     \
  ${EXTRA_ARGS:-}
