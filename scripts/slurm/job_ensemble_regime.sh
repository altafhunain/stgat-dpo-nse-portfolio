#!/bin/bash
#SBATCH --job-name=ens_regime
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/ens_regime_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
source ~/.bashrc
conda init
conda activate hunaingpu
cd /home/pramod/hunain/NSE_Portfolio_GNN
python training/test_ensemble_regime.py
