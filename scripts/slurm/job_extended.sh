#!/bin/bash
#SBATCH --job-name=stgat_ext
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/extended_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source ~/.bashrc
conda activate hunaingpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd /home/pramod/hunain/NSE_Portfolio_GNN
echo "Started: $(date)"
PYTHONUNBUFFERED=1 python -u training/train_stgat.py
echo "Finished: $(date)"
