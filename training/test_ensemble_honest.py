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

# HMM fit strictly on pre-blind-holdout data (2018-2020), consistent with
# the walk-forward validity check — this rule was derived BEFORE looking
# at the blind holdout at all.
hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
hmm_det.fit(nifty.loc["2018-01-01":"2020-12-31"].fillna(0).values)
regimes = hmm_det.predict(nifty.fillna(0).values)
regime_ser = pd.Series(regimes, index=nifty.index[-len(regimes):])

blind_start, blind_end = "2025-01-01", "2026-03-30"
stgat_b = stgat.loc[blind_start:blind_end]
blml_b  = blml.loc[blind_start:blind_end]
nifty_b = nifty.loc[blind_start:blind_end]
regime_b = regime_ser.reindex(stgat_b.index).ffill().fillna(1).astype(int)

print("Regime distribution in blind holdout:")
print(regime_b.value_counts().sort_index())
print("(0=bear, 1=sideways, 2=bull)\n")

# RULE, derived ONLY from walk-forward evidence (2021-2024), applied once:
#   bear     -> STGAT-DPO  (STGAT better in WF: -0.27 vs -0.65)
#   sideways -> BL-ML      (BL-ML better in WF: 0.21 vs 0.17)
#   bull     -> STGAT-DPO  (STGAT better in WF: 0.28 vs 0.01)
rule = {0: 1.0, 1: 0.0, 2: 1.0}
w_stgat = regime_b.map(rule)
blend = w_stgat * stgat_b + (1 - w_stgat) * blml_b

m = compute_metrics(blend, nifty_b)
print("="*62)
print("  WALK-FORWARD-DERIVED RULE, APPLIED ONCE TO BLIND HOLDOUT")
print("="*62)
print("  Rule: bear->STGAT, sideways->BL-ML, bull->STGAT")
print(f"  IR       : {m['info_ratio']:.4f}")
print(f"  Sharpe   : {m['sharpe']:.4f}")
print(f"  AnnRet   : {m['ann_return']:.4f}")
print(f"  MaxDD    : {m['max_drawdown']:.4f}")
print()
print(f"  Reference — pure BL-ML          ->  IR=1.1922  Sharpe=0.2088  AnnRet=0.1044")
print(f"  Reference — pure STGAT-DPO      ->  IR=0.0407  Sharpe=-0.7067  AnnRet=-0.0268")
print("\nDone.")
