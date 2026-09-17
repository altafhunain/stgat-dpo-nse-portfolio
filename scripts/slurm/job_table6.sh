#!/bin/bash
#SBATCH --job-name=table6
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/table6_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
source ~/.bashrc
conda init
conda activate hunaingpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd /home/pramod/hunain/NSE_Portfolio_GNN
python training/build_table6.py
