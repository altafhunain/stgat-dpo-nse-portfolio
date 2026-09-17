#!/bin/bash
#SBATCH --job-name=bootstrap_ci
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/bootstrap_ci_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
source ~/.bashrc
conda init
conda activate hunaingpu
cd /home/pramod/hunain/NSE_Portfolio_GNN
python training/compute_bootstrap_ci.py
