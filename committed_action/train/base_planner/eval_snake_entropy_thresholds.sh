#!/bin/bash
#SBATCH --job-name=eval-snake-entropy
#SBATCH --account=engs-a2i
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-user=firas.darwish@worc.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=eval-snake-entropy-%j.out
#SBATCH --error=eval-snake-entropy-%j.err

set -euo pipefail

# =============================================================================
# Repository setup
# =============================================================================

REPO_ROOT=/data/engs-a2i/worc6991/realtime-rl-code

if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "ERROR: Repository root does not exist: ${REPO_ROOT}"
    exit 1
fi

cd "${REPO_ROOT}"

echo "============================================================"
echo "Repository setup"
echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "PWD=$(pwd)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
echo "SCRIPT_ARG_ZERO=$0"
echo

module purge
module load Anaconda3

ENV_PY=/data/engs-a2i/worc6991/conda/envs/realtime_rl/bin/python

if [[ ! -x "${ENV_PY}" ]]; then
    echo "ERROR: Python executable does not exist: ${ENV_PY}"
    exit 1
fi

"${ENV_PY}" -c "import sys; print('Python executable:', sys.executable)"
"${ENV_PY}" -c "import jax; print('JAX version:', jax.__version__)"

echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

"${ENV_PY}" - << 'PYEOF'
import jax
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())
if jax.default_backend() != "gpu":
    raise RuntimeError("JAX is not using the GPU.")
PYEOF

AZ_CHECKPOINT=${AZ_CHECKPOINT:-"${REPO_ROOT}/checkpoints/committed_action/snake/base/training_state_best.pkl"}
THRESHOLD_PATH=${THRESHOLD_PATH:-"${REPO_ROOT}/checkpoints/committed_action/snake/entropy_threshold/entropy_thresholds_best.json"}
OUTPUT_PATH=${OUTPUT_PATH:-"${REPO_ROOT}/checkpoints/committed_action/snake/entropy_threshold/evaluation_entropy_final.json"}

NUM_EPISODES=${NUM_EPISODES:-100}
EVAL_META_STEPS=${EVAL_META_STEPS:-1200}
EVALUATION_SEED_BASE=${EVALUATION_SEED_BASE:-20000}
MCTS_SEED_OFFSET=${MCTS_SEED_OFFSET:-1000000}

if [[ ! -f "${AZ_CHECKPOINT}" ]]; then
    echo "ERROR: AZNet checkpoint not found: ${AZ_CHECKPOINT}"
    exit 1
fi
if [[ ! -f "${THRESHOLD_PATH}" ]]; then
    echo "ERROR: Threshold JSON not found: ${THRESHOLD_PATH}"
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "============================================================"
echo "Snake entropy-only evaluation configuration"
echo "============================================================"
echo "AZ_CHECKPOINT=${AZ_CHECKPOINT}"
echo "THRESHOLD_PATH=${THRESHOLD_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "NUM_EPISODES=${NUM_EPISODES}"
echo "EVAL_META_STEPS=${EVAL_META_STEPS}"
echo "EVALUATION_SEED_BASE=${EVALUATION_SEED_BASE}"
echo "MCTS_SEED_OFFSET=${MCTS_SEED_OFFSET}"
echo
cat "${THRESHOLD_PATH}"
echo

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_ROOT}/committed_action${PYTHONPATH:+:${PYTHONPATH}}"

echo "Start time: $(date)"

"${ENV_PY}" -u -m jumanji.training.eval_snake_entropy_thresholds \
    --threshold_path "${THRESHOLD_PATH}" \
    --az_checkpoint_path "${AZ_CHECKPOINT}" \
    --num_episodes "${NUM_EPISODES}" \
    --eval_meta_steps "${EVAL_META_STEPS}" \
    --evaluation_seed_base "${EVALUATION_SEED_BASE}" \
    --mcts_seed_offset "${MCTS_SEED_OFFSET}" \
    --entropy_only \
    --output_path "${OUTPUT_PATH}"

if [[ ! -f "${OUTPUT_PATH}" ]]; then
    echo "ERROR: Expected evaluation output not created: ${OUTPUT_PATH}"
    exit 1
fi

echo "Evaluation result:"
cat "${OUTPUT_PATH}"
echo
ls -lh "${OUTPUT_PATH}"

echo "Snake entropy-policy evaluation completed successfully at $(date)."
