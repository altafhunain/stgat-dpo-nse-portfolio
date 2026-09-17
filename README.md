# NSE Portfolio GNN -- STGAT-DPO

**Dynamic Portfolio Allocation via Spatio-Temporal Graph Attention Networks with Differentiable Optimization**

Hunain Altaf, Pramod Kumar Yadav -- National Institute of Technology Srinagar

This repository accompanies our manuscript submitted to IJIES (Paper ID 20265978) and contains the code, configurations, and result artifacts needed to reproduce the reported tables and figures.

## Project structure

Directories: data/ (fetch_new_stocks.py rebuilds prices_51.parquet), models/gnn/ (STGAT-DPO architecture, HMM regime detector), training/ (walk-forward trainer, baselines, stat tests), utils/ (feature engineering, graph construction), configs/ (fold boundaries, hyperparameters, baseline configs as YAML), outputs/results/ (per-fold weights, regime sequences, returns, statistical tests), scripts/slurm/ (SLURM job submission scripts).

## Environment

    conda activate hunaingpu
    pip install -r requirements.txt --break-system-packages

## Reproducing paper results

1. Data: python data/fetch_new_stocks.py rebuilds data/raw/prices_51.parquet (51 NSE large-cap tickers, Oct 2017-present, yfinance auto-adjusted OHLCV).
2. Fold structure: configs/folds.yaml -- 9 walk-forward folds (2021-2025) plus an 8-month blind holdout (Jan-Aug 2026), exact train/test date boundaries, validation-split method, and checkpoint-selection rule.
3. Model/training hyperparameters: configs/hyperparameters.yaml -- architecture, optimizer, loss, HMM regime detector (causal decoding via predict_causal, addressing look-ahead leakage identified in review), dynamic graph construction.
4. Baseline configurations: configs/baselines.yaml -- BL-ML, RL-PPO, and classical benchmarks (EW, MinVar, MaxSharpe, RiskParity, etc.) feature sets, hyperparameters, transaction costs (10 bps, shared across all strategies), and weight cap (15%, shared across all strategies).
5. Train: sbatch scripts/slurm/job_stgat_wf.sh for STGAT-DPO and ablations; python training/run_bl_ml.py, python training/run_rl_ppo.py, python training/run_benchmarks.py for baselines.
6. Statistical tests: python training/run_stat_tests.py (Newey-West HAC, Jobson-Korkie, Huberman-Kandel spanning, stochastic dominance, Diebold-Mariano); python training/compute_bootstrap_ci.py (fold-level percentile bootstrap, n=9, 10000 resamples, seed 42).

## Released result artifacts (outputs/results/)

weights_Fold1.csv through weights_Fold9.csv and weights_Blind2025.csv hold per-rebalance portfolio weights. regimes_Fold1.csv through regimes_Fold9.csv and regimes_Blind2025.csv hold daily causal HMM regime labels. Files ending in _returns.csv hold daily return series per strategy. statistical_tests_full.csv holds all hypothesis test results. ablation_results.csv, benchmark_fold_results.csv, bl_ml_results.csv, rl_ppo_results.csv, and stgat_fold_results.csv hold per-fold performance metrics. sensitivity_results_allfolds.csv holds the multi-fold hyperparameter sensitivity analysis.

## Universe

51 NSE large-cap stocks with continuous listing history (see manuscript Section 3.1 for the point-in-time eligibility mechanism and disclosed survivorship-bias limitation). Walk-forward evaluation spans 2021-2025 across 9 expanding-window folds with 6-month stepping. An independent blind holdout covers January-August 2026, evaluated once and not tuned on.

## Random seed

SEED = 42, shared across model training, all baselines, and the HMM regime detector, set in configs/folds.yaml and configs/hyperparameters.yaml.

## Contact

hunain_2022phacse009@nitsri.ac.in
