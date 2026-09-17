"""
training/test_causal_hmm2.py
==============================
Corrected version of test_causal_hmm.py: freezes training-set feature
normalization statistics and reuses them identically for BOTH the leaky
(full-sequence Viterbi) and causal (expanding-window Viterbi) decodes,
so the comparison isolates only the decoding-direction effect the
reviewers flagged, not a confounded normalization difference.
"""
import sys, os, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn import hmm as hmmlearn_hmm

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "results"

nifty_ser = pd.read_csv(OUT_DIR / "nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
nifty_ser = nifty_ser[~nifty_ser.index.duplicated(keep="first")].sort_index()

TRAIN_END = "2020-12-31"
TEST_END = "2021-06-30"

nifty_train = nifty_ser.loc[:TRAIN_END].values
nifty_full = nifty_ser.loc[:TEST_END].values
test_start_idx = len(nifty_train)

print("="*70)
print("  Causal vs. leaky HMM decoding -- normalization-controlled comparison")
print("="*70)
print(f"  Train: {len(nifty_train)} days ending {TRAIN_END}")
print(f"  Full (train+test): {len(nifty_full)} days ending {TEST_END} "
      f"({len(nifty_full)-test_start_idx} test days)")

def make_features(r):
    r = r.astype(np.float64)
    T = len(r)
    vol = np.array([r[max(0,i-21):i+1].std()*(252**0.5) for i in range(T)])
    vov = np.array([vol[max(0,i-63):i+1].std() for i in range(T)])
    mom = np.array([r[max(0,i-21):i+1].sum() for i in range(T)])
    return np.column_stack([r, vol, vov, mom])

# --- Fit on training data, STORE training normalization stats ---
X_train_raw = make_features(nifty_train)
feat_mean = X_train_raw.mean(axis=0)
feat_std = X_train_raw.std(axis=0) + 1e-8
X_train_norm = (X_train_raw - feat_mean) / feat_std

model = hmmlearn_hmm.GaussianHMM(n_components=3, covariance_type="diag",
                                   n_iter=500, tol=1e-4, random_state=42)
model.fit(X_train_norm)
means = model.means_[:, 0]
order = np.argsort(means)
state_map = {order[i]: i for i in range(3)}

# --- Full feature array, normalized using FROZEN training stats (both methods) ---
X_full_raw = make_features(nifty_full)
X_full_norm = (X_full_raw - feat_mean) / feat_std   # SAME normalization for both decodes

# --- Leaky: single Viterbi call on the whole (frozen-normalized) array ---
t0 = time.time()
raw_leaky = model.predict(X_full_norm)
regimes_leaky = np.array([state_map[s] for s in raw_leaky])
t_leaky = time.time() - t0
print(f"\n  Leaky decode: {t_leaky:.2f}s")

# --- Causal: expanding-window Viterbi (frozen-normalized features), terminal state only ---
def predict_causal_frozen(model, state_map, X_full_norm, min_window=30):
    T = X_full_norm.shape[0]
    out = np.zeros(T, dtype=np.int64)
    for i in range(T):
        if i < min_window:
            out[i] = out[max(0,i-1)] if i > 0 else 0
            continue
        raw_path = model.predict(X_full_norm[:i+1])
        out[i] = state_map[raw_path[-1]]
    return out

t0 = time.time()
regimes_causal = predict_causal_frozen(model, state_map, X_full_norm)
t_causal = time.time() - t0
print(f"  Causal decode: {t_causal:.2f}s  ({X_full_norm.shape[0]} expanding windows)")

# --- Compare ---
agree = (regimes_leaky == regimes_causal)
print(f"\n  Overall agreement (normalization now identical): {agree.mean():.1%}  ({agree.sum()}/{len(agree)} days)")

test_agree = agree[test_start_idx:]
print(f"  Test-period agreement: {test_agree.mean():.1%}  ({test_agree.sum()}/{len(test_agree)} days)")

labels = ["bear", "sideways", "bull"]
leaky_counts = np.bincount(regimes_leaky[test_start_idx:], minlength=3)
causal_counts = np.bincount(regimes_causal[test_start_idx:], minlength=3)
print(f"\n  Test-period regime distribution:")
print(f"    Leaky:  " + "  ".join(f"{labels[i]}={leaky_counts[i]}" for i in range(3)))
print(f"    Causal: " + "  ".join(f"{labels[i]}={causal_counts[i]}" for i in range(3)))

# Show WHERE disagreements happen (early vs late in test period)
disagree_idx = np.where(~test_agree)[0]
print(f"\n  Disagreement positions within test period (0-indexed): {disagree_idx.tolist()}")

print("\nDone.")
