#!/bin/bash
#SBATCH --job-name=nse_fetch
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --gres=gpu:h100_2g.24gb:1
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/fetch_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

source ~/.bashrc
conda init
conda activate hunaingpu

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/pramod/hunain/NSE_Portfolio_GNN

echo "Job started  : $(date)"
echo "Node         : $(hostname)"
echo "Environment  : $CONDA_DEFAULT_ENV"
echo ""

python data/fetch_new_stocks.py

echo ""
echo "Job finished : $(date)"
