#!/bin/bash
#SBATCH --job-name=check_hdfc
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/check_hdfc_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
source ~/.bashrc
conda init
conda activate hunaingpu
cd /home/pramod/hunain/NSE_Portfolio_GNN
python training/check_hdfclife.py
