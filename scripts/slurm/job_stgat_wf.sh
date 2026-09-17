#!/bin/bash
#SBATCH --job-name=stgat_wf
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --gres=gpu:h100_2g.24gb:1
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/stgat_wf_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

source ~/.bashrc
conda init
conda activate hunaingpu

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /home/pramod/hunain/NSE_Portfolio_GNN

echo "========================================"
echo "  STGAT-DPO Walk-Forward Training"
echo "  Job    : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU')"
echo "  Start  : $(date)"
echo "========================================"

python training/train_stgat.py

echo ""
echo "========================================"
echo "  Finished : $(date)"
echo "========================================"
