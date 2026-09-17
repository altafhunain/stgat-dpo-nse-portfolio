import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
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
    cum = np.exp(port_rets.cumsum())
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    common = port_rets.index.intersection(nifty_ser.index)
    active = port_rets.loc[common] - nifty_ser.loc[common]
    ir = float(active.mean() / (active.std() + 1e-8) * ANN ** 0.5)
    return dict(ann_return=round(ann_ret,4), sharpe=round(sharpe,4),
                info_ratio=round(ir,4), max_drawdown=round(max_dd,4))

stgat = pd.read_csv("outputs/results/stgat_returns.csv", index_col=0, parse_dates=True).iloc[:,0]
blml  = pd.read_csv("outputs/results/bl_ml_returns.csv", index_col=0, parse_dates=True).iloc[:,0]
nifty = pd.read_csv("outputs/results/nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:,0]

# Fit 3-state HMM on the same train window as production (2018-01-01 -> 2024-09-30)
tr_start, tr_end = "2018-01-01", "2024-09-30"
nifty_tr = nifty.loc[tr_start:tr_end].fillna(0).values

hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
hmm_det.fit(nifty_tr)

# Predict regimes over full nifty series (needed for regime labels within blind window)
nifty_full = nifty.fillna(0)
regimes = hmm_det.predict(nifty_full.values)
regime_ser = pd.Series(regimes, index=nifty_full.index[-len(regimes):])

blind_start, blind_end = "2025-01-01", "2026-03-30"
stgat_b = stgat.loc[blind_start:blind_end]
blml_b  = blml.loc[blind_start:blind_end]
nifty_b = nifty.loc[blind_start:blind_end]
regime_b = regime_ser.reindex(stgat_b.index).ffill().fillna(1).astype(int)

print("Regime distribution in blind holdout:")
print(regime_b.value_counts().sort_index())
print("(0=bear, 1=sideways, 2=bull)\n")

print("="*62)
print("  REGIME-CONDITIONED BLEND VARIANTS")
print("="*62)

# Variant A: bear/sideways -> favor STGAT, bull -> favor BL-ML (binary switch)
w_stgat_a = regime_b.map({0: 0.7, 1: 0.7, 2: 0.0})
blend_a = w_stgat_a * stgat_b + (1 - w_stgat_a) * blml_b
m = compute_metrics(blend_a, nifty_b)
print(f"  A) bear/sideways=0.7 STGAT, bull=0.0 STGAT  ->  IR={m['info_ratio']:.4f}  Sharpe={m['sharpe']:.4f}  AnnRet={m['ann_return']:.4f}")

# Variant B: full binary switch (100% STGAT in bear/sideways, 100% BL-ML in bull)
w_stgat_b = regime_b.map({0: 1.0, 1: 1.0, 2: 0.0})
blend_b = w_stgat_b * stgat_b + (1 - w_stgat_b) * blml_b
m = compute_metrics(blend_b, nifty_b)
print(f"  B) bear/sideways=1.0 STGAT, bull=0.0 STGAT  ->  IR={m['info_ratio']:.4f}  Sharpe={m['sharpe']:.4f}  AnnRet={m['ann_return']:.4f}")

# Variant C: mild tilt only
w_stgat_c = regime_b.map({0: 0.4, 1: 0.4, 2: 0.1})
blend_c = w_stgat_c * stgat_b + (1 - w_stgat_c) * blml_b
m = compute_metrics(blend_c, nifty_b)
print(f"  C) bear/sideways=0.4 STGAT, bull=0.1 STGAT  ->  IR={m['info_ratio']:.4f}  Sharpe={m['sharpe']:.4f}  AnnRet={m['ann_return']:.4f}")

print(f"\n  Reference — pure BL-ML          ->  IR=1.1922  Sharpe=0.2088  AnnRet=0.1044")
print(f"  Reference — pure STGAT-DPO      ->  IR=0.0407  Sharpe=-0.7067  AnnRet=-0.0268")
print("\nDone.")
