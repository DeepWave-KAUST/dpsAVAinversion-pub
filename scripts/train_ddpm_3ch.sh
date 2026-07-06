#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=ddpm_avo_3ch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --constraint=v100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =====================================================
#  Environment setup
# =====================================================
echo "=============================================="
echo "SLURM job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "=============================================="

START_TIME=$(date +%s)

# -----------------------------
# Environment setup
# -----------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffseis

# Threading / perf
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# =====================================================
#  Project paths (match your python script)
# =====================================================
WORKDIR="/home/brandof/diff/scripts_avo"   # <-- change to where the .py lives
PY_SCRIPT="train_ddpm_3ch.py"


echo "GPU info:"
nvidia-smi || true
echo "----------------------------------------------"
echo "Workdir    : ${WORKDIR}"
echo "Script     : ${PY_SCRIPT}"
echo "H5 path    : ${H5_PATH}"
echo "Batch size : ${BATCH_SIZE}  (hardcoded in python)"
echo "Epochs     : ${EPOCHS}      (hardcoded in python)"
echo "Workers    : ${NUM_WORKERS} (hardcoded in python)"
echo "LR         : ${LR}          (hardcoded in python)"
echo "AMP        : ${USE_AMP}     (hardcoded True in python)"
echo "Padding    : ${PAD}, pad_px=${PAD_PX} (hardcoded in python)"
echo "----------------------------------------------"

# =====================================================
#  Run training
# =====================================================
cd "${WORKDIR}" || exit 1
echo "Launching training script..."
srun python -u "${PY_SCRIPT}"

EXIT_CODE=$?

echo "GPU info (end):"
nvidia-smi || true

# =====================================================
#  Compute runtime
# =====================================================
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
RUNTIME_MIN=$(echo "scale=2; ${RUNTIME}/60" | bc)

echo "=============================================="
echo "Job finished at $(date)"
echo "Total runtime: ${RUNTIME}s  (~${RUNTIME_MIN} minutes)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
