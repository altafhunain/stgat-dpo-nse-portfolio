"""
training/test_causal_repeatability.py
========================================
Isolated repeatability test for HMMRegimeDetector.predict_causal():
fits once, then calls predict_causal() multiple times on the IDENTICAL
input array, checking for exact bit-for-bit agreement across calls.
No neural network training involved -- isolates whether the non-
determinism seen in full training runs originates in the causal HMM
decode itself, or elsewhere in the pipeline.
"""
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from pathlib import Path
from models.gnn.stgat_dpo import HMMRegimeDetector

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "results"

nifty_ser = pd.read_csv(OUT_DIR / "nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
nifty_ser = nifty_ser[~nifty_ser.index.duplicated(keep="first")].sort_index()

nifty_train = nifty_ser.loc[:"2020-12-31"].values
nifty_full = nifty_ser.loc[:"2021-12-31"].values  # Fold1's actual train+test range

print("="*70)
print("  Isolated repeatability test: predict_causal()")
print("="*70)

hmm_det = HMMRegimeDetector(n_states=3, seed=42)
hmm_det.fit(nifty_train)
print(f"  Fitted once. feat_mean={hmm_det.feat_mean}")

N_RUNS = 5
results = []
for run in range(N_RUNS):
    regimes = hmm_det.predict_causal(nifty_full)
    results.append(regimes)
    print(f"  Run {run+1}: bear={np.sum(regimes==0)} sideways={np.sum(regimes==1)} bull={np.sum(regimes==2)}")

print()
all_identical = all(np.array_equal(results[0], r) for r in results[1:])
print(f"  All {N_RUNS} runs bit-identical: {all_identical}")

if not all_identical:
    for i in range(1, N_RUNS):
        diff = np.where(results[0] != results[i])[0]
        print(f"  Run 1 vs Run {i+1}: {len(diff)} differing positions: {diff.tolist()[:30]}")

# Also test the underlying model.predict() call in isolation, repeated, on a FIXED window
print("\n  --- Sub-test: repeated model.predict() on a single FIXED window (no loop) ---")
X_test = np.random.RandomState(0).randn(500, 4)  # arbitrary fixed synthetic input, same every time
sub_results = []
for run in range(N_RUNS):
    raw = hmm_det.model.predict(X_test)
    sub_results.append(raw)
sub_identical = all(np.array_equal(sub_results[0], r) for r in sub_results[1:])
print(f"  Raw model.predict() on fixed synthetic input, {N_RUNS} runs identical: {sub_identical}")
if not sub_identical:
    for i in range(1, N_RUNS):
        diff = np.where(sub_results[0] != sub_results[i])[0]
        print(f"    Run 1 vs Run {i+1}: {len(diff)} differing positions")

print("\nDone.")
