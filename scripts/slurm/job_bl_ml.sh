#!/bin/bash
#SBATCH --job-name=bl_ml
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/bl_ml_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
source ~/.bashrc
conda init
conda activate hunaingpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /home/pramod/hunain/NSE_Portfolio_GNN
echo "========================================"
echo "  BL-ML Baseline"
echo "  Job  : $SLURM_JOB_ID"
echo "  Node : $(hostname)"
echo "  Start: $(date)"
echo "========================================"
PYTHONUNBUFFERED=1 python -u training/run_bl_ml.py
echo ""
echo "========================================"
echo "  Finished: $(date)"
echo "========================================"
