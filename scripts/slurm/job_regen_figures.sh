#!/bin/bash
#SBATCH --job-name=regen_figs
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/regen_figs_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
source ~/.bashrc
conda init
conda activate hunaingpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd /home/pramod/hunain/NSE_Portfolio_GNN
python training/generate_figures_final.py
