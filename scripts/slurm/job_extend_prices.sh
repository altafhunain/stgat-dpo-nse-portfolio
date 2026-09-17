#!/bin/bash
#SBATCH --job-name=extend_prices
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/extend_prices_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
source ~/.bashrc
conda init
conda activate hunaingpu
cd /home/pramod/hunain/NSE_Portfolio_GNN
python data/extend_prices_aug2026.py
