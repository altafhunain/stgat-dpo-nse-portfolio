#!/bin/bash
#SBATCH --job-name=proto_quant
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/proto_quant_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
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
echo "  Quantile-Head Prototype — Fold 5"
echo "  Job  : $SLURM_JOB_ID"
echo "  Start: $(date)"
echo "========================================"
PYTHONUNBUFFERED=1 python -u training/prototype_quantile_fold5.py
echo ""
echo "========================================"
echo "  Finished: $(date)"
echo "========================================"
