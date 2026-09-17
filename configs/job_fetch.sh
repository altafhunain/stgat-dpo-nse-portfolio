#!/usr/bin/env bash
#SBATCH --job-name=nse_fetch
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/fetch_%j.out
#SBATCH --error=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/fetch_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=short

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate nse_research

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── Run ───────────────────────────────────────────────────────────────────────
cd /home/pramod/hunain/NSE_Portfolio_GNN

echo "Job started  : $(date)"
echo "Node         : $(hostname)"
echo "Working dir  : $(pwd)"
echo ""

python data/fetch_new_stocks.py

echo ""
echo "Job finished : $(date)"
