#!/bin/bash
#SBATCH --job-name=fit-pacman-entropy

#SBATCH --account=engs-a2i

#SBATCH --partition=short

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=24:00:00

#SBATCH --mail-user=firas.darwish@worc.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output=fit-pacman-entropy-%j.out
#SBATCH --error=fit-pacman-entropy-%j.err

set -euo pipefail


# =============================================================================
# Repository setup
# =============================================================================

REPO_ROOT=/data/engs-a2i/worc6991/realtime-rl-code

if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "ERROR: Repository root does not exist:"
    echo "  ${REPO_ROOT}"
    exit 1
fi

cd "${REPO_ROOT}"

echo "============================================================"
echo "Repository setup"
echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "PWD=$(pwd)"
echo


# =============================================================================
# Python environment
# =============================================================================

module purge
module load Anaconda3

ENV_PY=/data/engs-a2i/worc6991/conda/envs/realtime_rl/bin/python

if [[ ! -x "${ENV_PY}" ]]; then
    echo "ERROR: Python executable does not exist:"
    echo "  ${ENV_PY}"
    exit 1
fi

echo "============================================================"
echo "Python environment"
echo "============================================================"
echo "ENV_PY=${ENV_PY}"
"${ENV_PY}" -c "import sys; print('Python executable:', sys.executable)"
"${ENV_PY}" -c "import jax; print('JAX version:', jax.__version__)"
echo


# =============================================================================
# Job information
# =============================================================================

echo "============================================================"
echo "Slurm job information"
echo "============================================================"
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo

nvidia-smi || true

echo
echo "============================================================"
echo "JAX device check"
echo "============================================================"

"${ENV_PY}" - << 'PYEOF'
import jax

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

if jax.default_backend() != "gpu":
    raise RuntimeError(
        "JAX is not using the GPU. Check the JAX/CUDA installation "
        "before running the threshold search."
    )
PYEOF


# =============================================================================
# Experiment configuration
# =============================================================================

# The pretrained frozen AZNet used by every K branch.
#
# Override when submitting with:
#   sbatch --export=ALL,AZ_CHECKPOINT=/some/path/file.pkl \
#     committed_action/train/base_planner/fit_pacman_entropy_thresholds.sh
#
# This default matches the checkpoint path that loaded successfully in your
# interactive command.
AZ_CHECKPOINT=${AZ_CHECKPOINT:-"${REPO_ROOT}/checkpoints/committed_action/pacman/base/training_state_best.pkl"}

# Output directory for:
#   entropy_thresholds_best.json
#   entropy_threshold_search_results.json
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/checkpoints/committed_action/pacman/entropy_threshold"}

# Official threshold-fitting configuration.
SEARCH_NUM_EPISODES=${SEARCH_NUM_EPISODES:-32}
VALIDATION_NUM_EPISODES=${VALIDATION_NUM_EPISODES:-64}
TOP_K_CANDIDATES=${TOP_K_CANDIDATES:-8}
EVAL_META_STEPS=${EVAL_META_STEPS:-1200}

# MCTS simulation budgets corresponding to K=1,2,3,4.
SIM_K1=${SIM_K1:-32}
SIM_K2=${SIM_K2:-64}
SIM_K3=${SIM_K3:-96}
SIM_K4=${SIM_K4:-128}

# Separate seed ranges for coarse search and validation.
SEARCH_SEED_BASE=${SEARCH_SEED_BASE:-0}
VALIDATION_SEED_BASE=${VALIDATION_SEED_BASE:-10000}
MCTS_SEED_OFFSET=${MCTS_SEED_OFFSET:-1000000}


# =============================================================================
# Validate paths
# =============================================================================

if [[ ! -f "${AZ_CHECKPOINT}" ]]; then
    echo
    echo "ERROR: AZNet checkpoint was not found:"
    echo "  ${AZ_CHECKPOINT}"
    echo
    echo "Available Pac-Man checkpoint files:"
    find "${REPO_ROOT}/checkpoints/committed_action/pacman" \
        -maxdepth 4 \
        -type f \
        -name 'training_state_*.pkl' \
        -print 2>/dev/null || true
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"


# =============================================================================
# Print resolved configuration
# =============================================================================

echo
echo "============================================================"
echo "Entropy-threshold fitting configuration"
echo "============================================================"
echo "AZ_CHECKPOINT=${AZ_CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo
echo "SEARCH_NUM_EPISODES=${SEARCH_NUM_EPISODES}"
echo "VALIDATION_NUM_EPISODES=${VALIDATION_NUM_EPISODES}"
echo "TOP_K_CANDIDATES=${TOP_K_CANDIDATES}"
echo "EVAL_META_STEPS=${EVAL_META_STEPS}"
echo
echo "SIM_OPTIONS=${SIM_K1} ${SIM_K2} ${SIM_K3} ${SIM_K4}"
echo "SEARCH_SEED_BASE=${SEARCH_SEED_BASE}"
echo "VALIDATION_SEED_BASE=${VALIDATION_SEED_BASE}"
echo "MCTS_SEED_OFFSET=${MCTS_SEED_OFFSET}"
echo
echo "Threshold grid:"
echo "  0.2 0.4 0.6 0.8"
echo "Number of threshold triples:"
echo "  C(4, 3) = 4"
echo "============================================================"
echo


# =============================================================================
# JAX runtime settings
# =============================================================================

# Avoid JAX preallocating nearly all GPU memory before the program needs it.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Uncomment when diagnosing compilation:
# export JAX_LOG_COMPILES=1

# The repository exposes the local jumanji package from committed_action.
export PYTHONPATH="${REPO_ROOT}/committed_action${PYTHONPATH:+:${PYTHONPATH}}"


# =============================================================================
# Run official entropy-threshold fit
# =============================================================================

echo "============================================================"
echo "Starting Pac-Man entropy-threshold fitting"
echo "Start time: $(date)"
echo "============================================================"
echo

"${ENV_PY}" -u -m jumanji.training.fit_pacman_entropy_thresholds \
    --az_checkpoint_path "${AZ_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name entropy_thresholds_best.json \
    --sim_options \
        "${SIM_K1}" \
        "${SIM_K2}" \
        "${SIM_K3}" \
        "${SIM_K4}" \
    --threshold_values \
        0.2 \
        0.4 \
        0.6 \
        0.8 \
    --search_num_episodes "${SEARCH_NUM_EPISODES}" \
    --validation_num_episodes "${VALIDATION_NUM_EPISODES}" \
    --top_k_candidates "${TOP_K_CANDIDATES}" \
    --eval_meta_steps "${EVAL_META_STEPS}" \
    --search_seed_base "${SEARCH_SEED_BASE}" \
    --validation_seed_base "${VALIDATION_SEED_BASE}" \
    --mcts_seed_offset "${MCTS_SEED_OFFSET}" \
    --save_all_results


# =============================================================================
# Verify outputs
# =============================================================================

BEST_THRESHOLDS="${OUTPUT_DIR}/entropy_thresholds_best.json"
ALL_RESULTS="${OUTPUT_DIR}/entropy_threshold_search_results.json"

echo
echo "============================================================"
echo "Threshold fitting command finished"
echo "End time: $(date)"
echo "============================================================"

if [[ ! -f "${BEST_THRESHOLDS}" ]]; then
    echo "ERROR: Expected output was not created:"
    echo "  ${BEST_THRESHOLDS}"
    exit 1
fi

echo
echo "Best threshold configuration:"
cat "${BEST_THRESHOLDS}"

echo
echo "Created output files:"
ls -lh "${BEST_THRESHOLDS}"

if [[ -f "${ALL_RESULTS}" ]]; then
    ls -lh "${ALL_RESULTS}"
fi

echo
echo "============================================================"
echo "Pac-Man entropy-threshold fitting completed successfully."
echo "============================================================"