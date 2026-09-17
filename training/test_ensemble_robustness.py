import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from models.gnn.stgat_dpo import HMMRegimeDetector

SEED = 42
RF_DAILY = 0.065 / 252
ANN = 252

def compute_metrics(port_rets, nifty_ser):
    if len(port_rets) < 5:
        return {}
    excess = port_rets - RF_DAILY
    ann_ret = float(port_rets.mean() * ANN)
    sharpe = float(excess.mean() / (port_rets.std() + 1e-8) * ANN ** 0.5)
    common = port_rets.index.intersection(nifty_ser.index)
    active = port_rets.loc[common] - nifty_ser.loc[common]
    ir = float(active.mean() / (active.std() + 1e-8) * ANN ** 0.5)
    return dict(ann_return=round(ann_ret,4), sharpe=round(sharpe,4), info_ratio=round(ir,4))

stgat = pd.read_csv("outputs/results/stgat_returns.csv", index_col=0, parse_dates=True).iloc[:,0]
blml  = pd.read_csv("outputs/results/bl_ml_returns.csv", index_col=0, parse_dates=True).iloc[:,0]
nifty = pd.read_csv("outputs/results/nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:,0]

blind_start, blind_end = "2025-01-01", "2026-03-30"
stgat_b = stgat.loc[blind_start:blind_end]
blml_b  = blml.loc[blind_start:blind_end]
nifty_b = nifty.loc[blind_start:blind_end]

# Fixed rule from walk-forward evidence: bear->STGAT, sideways->BL-ML, bull->STGAT
rule = {0: 1.0, 1: 0.0, 2: 1.0}

fit_windows = [
    ("2018-01-01", "2019-12-31"),
    ("2018-01-01", "2020-12-31"),
    ("2018-01-01", "2021-12-31"),
    ("2018-01-01", "2022-12-31"),
    ("2018-01-01", "2024-09-30"),
]

print("="*78)
print("  ROBUSTNESS: ENSEMBLE IR ACROSS DIFFERENT HMM FIT WINDOWS")
print("="*78)
print(f"  {'Fit window':<25} {'Bear':>5} {'Side':>5} {'Bull':>5}   {'IR':>8}  {'Sharpe':>8}  {'AnnRet':>8}")
print(f"  {'-'*74}")

for start, end in fit_windows:
    hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
    hmm_det.fit(nifty.loc[start:end].fillna(0).values)
    regimes = hmm_det.predict(nifty.fillna(0).values)
    regime_ser = pd.Series(regimes, index=nifty.index[-len(regimes):])
    regime_b = regime_ser.reindex(stgat_b.index).ffill().fillna(1).astype(int)

    counts = regime_b.value_counts()
    n_bear = counts.get(0, 0)
    n_side = counts.get(1, 0)
    n_bull = counts.get(2, 0)

    w_stgat = regime_b.map(rule)
    blend = w_stgat * stgat_b + (1 - w_stgat) * blml_b
    m = compute_metrics(blend, nifty_b)

    label = f"{start} to {end}"
    print(f"  {label:<25} {n_bear:>5} {n_side:>5} {n_bull:>5}   "
          f"{m['info_ratio']:>8.4f}  {m['sharpe']:>8.4f}  {m['ann_return']:>8.4f}")

print(f"\n  Reference — pure BL-ML     ->  IR=1.1922  Sharpe=0.2088  AnnRet=0.1044")
print(f"  Reference — pure STGAT-DPO ->  IR=0.0407  Sharpe=-0.7067  AnnRet=-0.0268")
print("\nDone.")
