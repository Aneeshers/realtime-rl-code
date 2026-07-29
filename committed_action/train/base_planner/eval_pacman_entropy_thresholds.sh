#!/bin/bash
#SBATCH --job-name=eval-pacman-entropy

#SBATCH --account=engs-a2i

#SBATCH --partition=short

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=24:00:00

#SBATCH --mail-user=firas.darwish@worc.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output=eval-pacman-entropy-%j.out
#SBATCH --error=eval-pacman-entropy-%j.err

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
        "before running the entropy-policy evaluation."
    )
PYEOF


# =============================================================================
# Experiment configuration
# =============================================================================

# Frozen pretrained AZNet checkpoint.
AZ_CHECKPOINT=${AZ_CHECKPOINT:-"${REPO_ROOT}/checkpoints/committed_action/pacman/base/training_state_best.pkl"}

# Threshold JSON produced by fit_pacman_entropy_thresholds.py.
THRESHOLD_PATH=${THRESHOLD_PATH:-"${REPO_ROOT}/checkpoints/committed_action/pacman/entropy_threshold/entropy_thresholds_best.json"}

# Output JSON for the held-out entropy-policy evaluation.
OUTPUT_PATH=${OUTPUT_PATH:-"${REPO_ROOT}/checkpoints/committed_action/pacman/entropy_threshold/evaluation_entropy_final.json"}

# Official held-out evaluation configuration.
NUM_EPISODES=${NUM_EPISODES:-100}
EVAL_META_STEPS=${EVAL_META_STEPS:-1200}
EVALUATION_SEED_BASE=${EVALUATION_SEED_BASE:-20000}
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

if [[ ! -f "${THRESHOLD_PATH}" ]]; then
    echo
    echo "ERROR: Entropy-threshold JSON was not found:"
    echo "  ${THRESHOLD_PATH}"
    echo
    echo "Run fit_pacman_entropy_thresholds.py first."
    exit 1
fi

OUTPUT_DIR="$(dirname "${OUTPUT_PATH}")"
mkdir -p "${OUTPUT_DIR}"


# =============================================================================
# Print resolved configuration
# =============================================================================

echo
echo "============================================================"
echo "Entropy-policy evaluation configuration"
echo "============================================================"
echo "AZ_CHECKPOINT=${AZ_CHECKPOINT}"
echo "THRESHOLD_PATH=${THRESHOLD_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo
echo "NUM_EPISODES=${NUM_EPISODES}"
echo "EVAL_META_STEPS=${EVAL_META_STEPS}"
echo "EVALUATION_SEED_BASE=${EVALUATION_SEED_BASE}"
echo "MCTS_SEED_OFFSET=${MCTS_SEED_OFFSET}"
echo
echo "Evaluation mode:"
echo "  entropy policy only"
echo "  fixed-K baselines skipped"
echo "  random-K baseline skipped"
echo "============================================================"
echo

echo "Threshold configuration:"
cat "${THRESHOLD_PATH}"
echo


# =============================================================================
# JAX runtime settings
# =============================================================================

# Avoid JAX preallocating nearly all GPU memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Uncomment when diagnosing compilation:
# export JAX_LOG_COMPILES=1

# Make committed_action/jumanji importable while running from the repository root.
export PYTHONPATH="${REPO_ROOT}/committed_action${PYTHONPATH:+:${PYTHONPATH}}"


# =============================================================================
# Run official entropy-only evaluation
# =============================================================================

echo "============================================================"
echo "Starting Pac-Man entropy-policy evaluation"
echo "Start time: $(date)"
echo "============================================================"
echo

"${ENV_PY}" -u -m jumanji.training.eval_pacman_entropy_thresholds \
    --threshold_path "${THRESHOLD_PATH}" \
    --az_checkpoint_path "${AZ_CHECKPOINT}" \
    --num_episodes "${NUM_EPISODES}" \
    --eval_meta_steps "${EVAL_META_STEPS}" \
    --evaluation_seed_base "${EVALUATION_SEED_BASE}" \
    --mcts_seed_offset "${MCTS_SEED_OFFSET}" \
    --entropy_only \
    --output_path "${OUTPUT_PATH}"


# =============================================================================
# Verify output
# =============================================================================

echo
echo "============================================================"
echo "Entropy-policy evaluation command finished"
echo "End time: $(date)"
echo "============================================================"

if [[ ! -f "${OUTPUT_PATH}" ]]; then
    echo "ERROR: Expected evaluation output was not created:"
    echo "  ${OUTPUT_PATH}"
    exit 1
fi

echo
echo "Evaluation result:"
cat "${OUTPUT_PATH}"

echo
echo "Created output file:"
ls -lh "${OUTPUT_PATH}"

echo
echo "============================================================"
echo "Pac-Man entropy-policy evaluation completed successfully."
echo "============================================================"