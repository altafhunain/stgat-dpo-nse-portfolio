"""
training/compute_bootstrap_ci.py
==================================
Percentile bootstrap 95% CI on mean walk-forward Information Ratio,
computed properly (fold-level resampling) for every strategy, replacing
the previously hardcoded, untraceable [0.040, 1.076] figure.
Method: resample the 9 walk-forward fold-level IR values with
replacement, recompute the mean IR each draw, repeat N_BOOT times,
take the 2.5th/97.5th percentiles as the 95% CI. This is a small-sample
bootstrap (n=9 folds) and is reported as such.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 10000

def bootstrap_ci(fold_irs, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)  # fresh identical seed per strategy: result independent of call order
    fold_irs = np.asarray(fold_irs)
    n = len(fold_irs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = fold_irs[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return fold_irs.mean(), lo, hi

def load_fold_irs(path, strategy_name, strategy_col="strategy"):
    df = pd.read_csv(path)
    df = df[df[strategy_col] == strategy_name]
    df = df[df["fold"].str.startswith("Fold")]
    return df["info_ratio"].values

sources = [
    ("STGAT-DPO",   "outputs/results/stgat_fold_results.csv",   "STGAT-DPO",   "2026-08-18 (confirmed final, point-in-time + interop fixes)"),
    ("STGAT-NoHMM", "outputs/results/ablation_results.csv",     "STGAT-NoHMM",       "2026-08-20 (confirmed final, both fixes)"),
    ("STGAT-NoGAT", "outputs/results/ablation_results.csv",     "STGAT-NoGAT",       "2026-08-20 (confirmed final, both fixes)"),
    ("BL-ML",       "outputs/results/bl_ml_results.csv",        "BL-ML",       "2026-08-06 (BLAS-thread fixed; interop-threading not separately re-tested)"),
    ("RL-PPO",      "outputs/results/rl_ppo_results.csv",       "RL-PPO",      "2026-08-06 (BLAS-thread fixed; interop-threading not separately re-tested)"),
]

print("="*90)
print("  BOOTSTRAP 95% CI ON MEAN WALK-FORWARD IR (fold-level percentile bootstrap, n=9 folds)")
print("="*90)
print(f"  {'Strategy':<14} {'Mean IR':>9} {'CI Low':>9} {'CI High':>9} {'vs zero':>8}  Data vintage")
print("  " + "-"*88)

results = []
for label, path, strat, vintage in sources:
    try:
        irs = load_fold_irs(path, strat)
        if len(irs) != 9:
            print(f"  {label:<14}  WARNING: found {len(irs)} folds, expected 9 -- skipping")
            continue
        mean_ir, lo, hi = bootstrap_ci(irs)
        excl = "above 0" if lo > 0 else ("below 0" if hi < 0 else "spans 0")
        print(f"  {label:<14} {mean_ir:>9.4f} {lo:>9.4f} {hi:>9.4f} {excl:>8}  {vintage}")
        results.append((label, mean_ir, lo, hi))
    except Exception as e:
        print(f"  {label:<14}  ERROR: {e}")

# Benchmark strategies (different schema: has 'name' column)
bench_df = pd.read_csv("outputs/results/benchmark_fold_results.csv")
bench_strats = bench_df["name"].unique() if "name" in bench_df.columns else bench_df["strategy"].unique()
print("\n  Benchmark strategies (2026-08-06 vintage, BLAS-thread fixed):")
for strat in bench_strats:
    col = "name" if "name" in bench_df.columns else "strategy"
    sub = bench_df[(bench_df[col] == strat) & (bench_df["fold"].str.startswith("Fold"))]
    irs = sub["info_ratio"].values
    if len(irs) != 9:
        print(f"  {strat:<14}  WARNING: found {len(irs)} folds, expected 9 -- skipping")
        continue
    mean_ir, lo, hi = bootstrap_ci(irs)
    excl = "above 0" if lo > 0 else ("below 0" if hi < 0 else "spans 0")
    print(f"  {strat:<14} {mean_ir:>9.4f} {lo:>9.4f} {hi:>9.4f} {excl:>8}")

print("\nDone.")
