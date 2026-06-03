#!/bin/bash
# Evaluate Speed Hex gating policy under the strict-timeout variant (appendix experiment).

set -euo pipefail
cd "$(dirname "$0")/../../.."

ENV=${ENV:-speed_hex_timeout}
HEX_SIZE=${HEX_SIZE:-11}
TIMES=${TIMES:-300 1200 2300 3500 4100}
SEEDS=${SEEDS:-0}
SIM_OPTIONS=${SIM_OPTIONS:-2,8,32,128}
OPPONENTS=${OPPONENTS:-0,2,8,32,128,random,midpeak,proportional}
NUM_GAMES=${NUM_GAMES:-100}
CKPT_ROOT=${CKPT_ROOT:-./checkpoints/clock/hex/base}
GATE_ROOT=${GATE_ROOT:-./checkpoints/clock/hex/gating_timeout}
GATE_CKPT=${GATE_CKPT:-${GATE_ROOT}/gate.pkl}
ITER_FILE=${ITER_FILE:-base_planner.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-./eval_outputs/clock/hex_gating_timeout}

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=clock/networks:clock/envs:. python clock/eval/gating_policy/eval_hex_gating_timeout.py \
  --env             "${ENV}"            \
  --hex_size        "${HEX_SIZE}"       \
  --times           ${TIMES}            \
  --seeds           "${SEEDS}"          \
  --sim_options     "${SIM_OPTIONS}"    \
  --opponents       "${OPPONENTS}"      \
  --num_games       "${NUM_GAMES}"      \
  --ckpt_root       "${CKPT_ROOT}"      \
  --gate_root       "${GATE_ROOT}"      \
  --gate_ckpt       "${GATE_CKPT}"      \
  --iter_file       "${ITER_FILE}"      \
  --output_dir      "${OUTPUT_DIR}"     \
  ${EXTRA_ARGS:-}
