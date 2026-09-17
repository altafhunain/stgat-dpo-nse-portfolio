#!/bin/bash
#SBATCH --job-name=ens_robust
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/ens_robust_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
source ~/.bashrc
conda init
conda activate hunaingpu
cd /home/pramod/hunain/NSE_Portfolio_GNN
python training/test_ensemble_robustness.py
