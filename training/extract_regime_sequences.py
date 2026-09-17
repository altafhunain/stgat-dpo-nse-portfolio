"""
training/extract_regime_sequences.py
======================================
Standalone script to persist the causal HMM regime sequence for every
walk-forward fold and the blind holdout, replicating the exact fit/
predict_causal logic embedded in train_stgat.py's main training loop
(same SEED, same fold boundaries) without re-running full model training.
Addresses R1#10: "HMM fitting/decoding code and regime sequences."
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '.')
from models.gnn.stgat_dpo import HMMRegimeDetector

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "results"
SEED = 42

FOLDS = [
    ("2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2018-01-01", "2021-06-30", "2021-07-01", "2022-06-30"),
    ("2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2018-01-01", "2022-06-30", "2022-07-01", "2023-06-30"),
    ("2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2018-01-01", "2023-06-30", "2023-07-01", "2024-06-30"),
    ("2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2018-01-01", "2024-06-30", "2024-07-01", "2025-06-30"),
    ("2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]
BLIND = ("2018-01-01", "2025-12-31", "2026-01-01", "2026-08-28")

nifty_ser = pd.read_csv(OUT_DIR / "nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
dates_ret = nifty_ser.index.values
nifty_arr = nifty_ser.values
labels = ["bear", "sideways", "bull"]

folds_to_run = FOLDS + [BLIND]
fold_names = [f"Fold{i+1}" for i in range(len(FOLDS))] + ["Blind2025"]

for fold_name, (tr_start, tr_end, te_start, te_end) in zip(fold_names, folds_to_run):
    tr_mask = (dates_ret >= np.datetime64(tr_start)) & (dates_ret <= np.datetime64(tr_end))
    nifty_tr = nifty_arr[tr_mask]

    hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
    hmm_det.fit(nifty_tr)
    regimes_full = hmm_det.predict_causal(nifty_arr)

    out = pd.DataFrame({
        "date": dates_ret,
        "regime": [labels[r] for r in regimes_full],
    })
    out_path = OUT_DIR / f"regimes_{fold_name}.csv"
    out.to_csv(out_path, index=False)
    print(f"  {fold_name:<12} saved -> {out_path}  ({len(out)} rows)")

print("\nDone — regime sequences persisted for all folds + blind holdout.")
