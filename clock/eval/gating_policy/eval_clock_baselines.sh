#!/bin/bash
# Run the gating policy plus each fixed-budget baseline for one clock game.
# Each baseline gets its own --wandb_group and output dir.
#
# Usage:
#   ENV_KIND=hex bash clock/eval/gating_policy/eval_clock_baselines.sh
#   ENV_KIND=go  bash clock/eval/gating_policy/eval_clock_baselines.sh

set -euo pipefail
cd "$(dirname "$0")/../../.."

ENV_KIND=${ENV_KIND:-hex}   # hex | go
case "$ENV_KIND" in
  hex)
    LAUNCHER=clock/eval/gating_policy/eval_hex_gating.sh
    FIXED_KS=${FIXED_KS:-"2 8 32 128"}
    ;;
  go)
    LAUNCHER=clock/eval/gating_policy/eval_go_gating.sh
    FIXED_KS=${FIXED_KS:-"16 32 64 96"}
    ;;
  *)
    echo "ENV_KIND must be 'hex' or 'go' (got '$ENV_KIND')" >&2
    exit 1
    ;;
esac

OUT_BASE=${OUT_BASE:-./eval_outputs/clock/${ENV_KIND}_fig4}

echo "${ENV_KIND}: gate -> ${OUT_BASE}/gate"
OUTPUT_DIR="${OUT_BASE}/gate" \
EXTRA_ARGS="--wandb_group fig4_${ENV_KIND}_gate ${EXTRA_ARGS:-}" \
  bash "$LAUNCHER"

for K in ${FIXED_KS}; do
  echo "${ENV_KIND}: fixed k=${K} -> ${OUT_BASE}/fixed_k${K}"
  OUTPUT_DIR="${OUT_BASE}/fixed_k${K}" \
  EXTRA_ARGS="--force_gate_choice ${K} --wandb_group fig4_${ENV_KIND}_fixed_k${K} ${EXTRA_ARGS:-}" \
    bash "$LAUNCHER"
done

echo "done; per-bar expected_score is in wandb (group fig4_${ENV_KIND}_*)"
