#!/bin/bash
#SBATCH --job-name=proto_blind
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/proto_blind_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
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
echo "  Quantile-Head Prototype — Blind Holdout"
echo "  Job  : $SLURM_JOB_ID"
echo "  Start: $(date)"
echo "========================================"
PYTHONUNBUFFERED=1 python -u training/prototype_quantile_blind.py
echo ""
echo "========================================"
echo "  Finished: $(date)"
echo "========================================"
