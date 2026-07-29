#!/bin/bash
#SBATCH --job-name=fit-tetris-entropy

#SBATCH --account=engs-a2i
#SBATCH --reservation=a2i2025

#SBATCH --partition=short
#SBATCH --nodelist=htc-g051

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-user=firas.darwish@worc.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=fit-tetris-entropy-%j.out
#SBATCH --error=fit-tetris-entropy-%j.err

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

# =============================================================================
# Python environment
# =============================================================================

module purge
module load Anaconda3

ENV_PY=/data/engs-a2i/worc6991/conda/envs/realtime_rl/bin/python

if [[ ! -x "${ENV_PY}" ]]; then
    echo "ERROR: Python executable does not exist: ${ENV_PY}"
    exit 1
fi

"${ENV_PY}" -c "import sys; print('Python executable:', sys.executable)"
"${ENV_PY}" -c "import jax; print('JAX version:', jax.__version__)"

# =============================================================================
# Job and GPU checks
# =============================================================================

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

# =============================================================================
# Experiment configuration
# =============================================================================

AZ_CHECKPOINT=${AZ_CHECKPOINT:-"${REPO_ROOT}/checkpoints/committed_action/tetris_rt/base/training_state_best.pkl"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/checkpoints/committed_action/tetris_rt-v1_tryfix/entropy_threshold"}

SEARCH_NUM_EPISODES=${SEARCH_NUM_EPISODES:-16}
VALIDATION_NUM_EPISODES=${VALIDATION_NUM_EPISODES:-32}
TOP_K_CANDIDATES=${TOP_K_CANDIDATES:-2}
EVAL_META_STEPS=${EVAL_META_STEPS:-1200}

SIM_K1=${SIM_K1:-32}
SIM_K2=${SIM_K2:-64}
SIM_K3=${SIM_K3:-96}
SIM_K4=${SIM_K4:-128}

SEARCH_SEED_BASE=${SEARCH_SEED_BASE:-0}
VALIDATION_SEED_BASE=${VALIDATION_SEED_BASE:-10000}
MCTS_SEED_OFFSET=${MCTS_SEED_OFFSET:-1000000}

if [[ ! -f "${AZ_CHECKPOINT}" ]]; then
    echo "ERROR: AZNet checkpoint not found: ${AZ_CHECKPOINT}"
    find "${REPO_ROOT}/checkpoints/committed_action/tetris_rt" \
        -maxdepth 4 -type f -name 'training_state_*.pkl' -print 2>/dev/null || true
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Tetris entropy-threshold fitting configuration"
echo "============================================================"
echo "AZ_CHECKPOINT=${AZ_CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SEARCH_NUM_EPISODES=${SEARCH_NUM_EPISODES}"
echo "VALIDATION_NUM_EPISODES=${VALIDATION_NUM_EPISODES}"
echo "TOP_K_CANDIDATES=${TOP_K_CANDIDATES}"
echo "EVAL_META_STEPS=${EVAL_META_STEPS}"
echo "SIM_OPTIONS=${SIM_K1} ${SIM_K2} ${SIM_K3} ${SIM_K4}"
echo "THRESHOLD_GRID=0.2 0.4 0.6 0.8"
echo "CANDIDATES=C(4,3)=4"
echo

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_ROOT}/committed_action${PYTHONPATH:+:${PYTHONPATH}}"

echo "Start time: $(date)"

"${ENV_PY}" -u -m jumanji.training.fit_tetris_entropy_thresholds \
    --az_checkpoint_path "${AZ_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name entropy_thresholds_best.json \
    --sim_options "${SIM_K1}" "${SIM_K2}" "${SIM_K3}" "${SIM_K4}" \
    --threshold_values 0.2 0.4 0.6 0.8 \
    --search_num_episodes "${SEARCH_NUM_EPISODES}" \
    --validation_num_episodes "${VALIDATION_NUM_EPISODES}" \
    --top_k_candidates "${TOP_K_CANDIDATES}" \
    --eval_meta_steps "${EVAL_META_STEPS}" \
    --search_seed_base "${SEARCH_SEED_BASE}" \
    --validation_seed_base "${VALIDATION_SEED_BASE}" \
    --mcts_seed_offset "${MCTS_SEED_OFFSET}" \
    --save_all_results

BEST_THRESHOLDS="${OUTPUT_DIR}/entropy_thresholds_best.json"
ALL_RESULTS="${OUTPUT_DIR}/entropy_threshold_search_results.json"

if [[ ! -f "${BEST_THRESHOLDS}" ]]; then
    echo "ERROR: Expected output not created: ${BEST_THRESHOLDS}"
    exit 1
fi

echo "Best threshold configuration:"
cat "${BEST_THRESHOLDS}"
echo
ls -lh "${BEST_THRESHOLDS}"
[[ -f "${ALL_RESULTS}" ]] && ls -lh "${ALL_RESULTS}"

echo "Tetris entropy-threshold fitting completed successfully at $(date)."
