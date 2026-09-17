"""
training/test_causal_hmm.py
=============================
Tests a causal (leakage-free) alternative to HMMRegimeDetector.predict():
instead of one Viterbi decode over the full array (which lets future
observations influence past state labels), re-decode an expanding window
ending at each day and keep only the terminal (most recent) state.

This is a standalone comparison script -- no training, no model changes.
Purpose: measure how much the regime LABELS actually differ between the
leaky (full-sequence) and causal (expanding-window) decoding, before
deciding whether a full pipeline retrain is warranted.
"""
import sys, os, time
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

# Use Fold 1's actual train/test split as a representative test case
# (matches train_stgat.py's FOLDS structure: fit on train only, decode on everything)
TRAIN_END = "2020-12-31"
TEST_END = "2021-06-30"

nifty_train = nifty_ser.loc[:TRAIN_END].values
nifty_full = nifty_ser.loc[:TEST_END].values  # train + test, as the leaky call currently uses
test_start_idx = len(nifty_train)  # index where the test period begins within nifty_full

print("="*70)
print("  Causal vs. leaky HMM regime decoding -- comparison")
print("="*70)
print(f"  Train period ends: {TRAIN_END}  ({len(nifty_train)} days)")
print(f"  Test period ends:  {TEST_END}  ({len(nifty_full)} days total, "
      f"{len(nifty_full)-test_start_idx} test days)")

hmm_det = HMMRegimeDetector(n_states=3, seed=42)
hmm_det.fit(nifty_train)

# --- Leaky (current) approach: single Viterbi call on the whole array ---
t0 = time.time()
regimes_leaky = hmm_det.predict(nifty_full)
t_leaky = time.time() - t0
print(f"\n  Leaky decode: {t_leaky:.2f}s")

# --- Causal approach: expanding-window Viterbi, terminal state only ---
def predict_causal(hmm_det, returns_full, min_window=30):
    """For each index i, decode returns_full[:i+1] and keep the last state."""
    from hmmlearn import hmm as hmmlearn_hmm
    T = len(returns_full)
    out = np.zeros(T, dtype=np.int64)
    r = returns_full.astype(np.float64)
    # Precompute the same feature transform used inside HMMRegimeDetector.predict
    vol = np.array([r[max(0,i-21):i+1].std()*(252**0.5) for i in range(T)])
    vov = np.array([vol[max(0,i-63):i+1].std() for i in range(T)])
    mom = np.array([r[max(0,i-21):i+1].sum() for i in range(T)])
    X_full = np.column_stack([r, vol, vov, mom])
    for i in range(T):
        if i < min_window:
            out[i] = out[max(0,i-1)] if i > 0 else 0
            continue
        X_window = X_full[:i+1]
        Xn = (X_window - X_window.mean(axis=0)) / (X_window.std(axis=0) + 1e-8)
        raw_path = hmm_det.model.predict(Xn)
        raw_last = raw_path[-1]
        out[i] = hmm_det.state_map[raw_last]
    return out

t0 = time.time()
regimes_causal = predict_causal(hmm_det, nifty_full)
t_causal = time.time() - t0
print(f"  Causal decode: {t_causal:.2f}s  ({len(nifty_full)} expanding windows)")

# --- Compare ---
agree = (regimes_leaky == regimes_causal)
print(f"\n  Overall agreement: {agree.mean():.1%}  ({agree.sum()}/{len(agree)} days)")

test_agree = agree[test_start_idx:]
print(f"  Test-period agreement (the part that matters for leakage): "
      f"{test_agree.mean():.1%}  ({test_agree.sum()}/{len(test_agree)} days)")

labels = ["bear", "sideways", "bull"]
leaky_counts = np.bincount(regimes_leaky[test_start_idx:], minlength=3)
causal_counts = np.bincount(regimes_causal[test_start_idx:], minlength=3)
print(f"\n  Test-period regime distribution:")
print(f"    Leaky:  " + "  ".join(f"{labels[i]}={leaky_counts[i]}" for i in range(3)))
print(f"    Causal: " + "  ".join(f"{labels[i]}={causal_counts[i]}" for i in range(3)))

print("\nDone.")
