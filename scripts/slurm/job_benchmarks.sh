#!/bin/bash
#SBATCH --job-name=nse_bench
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --output=/home/pramod/hunain/NSE_Portfolio_GNN/outputs/logs/bench_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source ~/.bashrc
conda init
conda activate hunaingpu

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /home/pramod/hunain/NSE_Portfolio_GNN

echo "Job started  : $(date)"
echo "Node         : $(hostname)"
echo ""

python training/run_benchmarks.py

echo ""
echo "Job finished : $(date)"
