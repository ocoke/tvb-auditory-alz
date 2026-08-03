#!/bin/bash -l

#$ -P riseprac
#$ -N tvb379
#$ -cwd
#$ -j y
#$ -o /projectnb/riseprac/tvb-auditory-alz/logs/
#$ -pe omp 28
#$ -l h_rt=12:00:00
#$ -l mem_per_core=3G

set -euo pipefail

PROJECT_DIR="/projectnb/riseprac/tvb-auditory-alz"
cd "$PROJECT_DIR"

module purge
module load python3/3.12.4

source "$PROJECT_DIR/venvs/tvb/bin/activate"

# Prevent each multiprocessing worker from spawning extra native threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WORKERS=$((NSLOTS - 1))

echo "========================================"
echo "Job ID: $JOB_ID"
echo "Allocated cores: $NSLOTS"
echo "TVB workers: $WORKERS"
echo "Requested memory: 3 GB/core, $((NSLOTS * 3)) GB total"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Start time: $(date)"
echo "========================================"

python -u main_extended.py --mode final --workers "$WORKERS"

echo "========================================"
echo "End time: $(date)"
echo "TVB experiment completed successfully."
echo "========================================"