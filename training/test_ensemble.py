import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

RF_DAILY = 0.065 / 252
ANN = 252

def compute_metrics(port_rets, nifty_ser):
    if len(port_rets) < 5:
        return {}
    excess = port_rets - RF_DAILY
    ann_ret = float(port_rets.mean() * ANN)
    ann_vol = float(port_rets.std() * ANN ** 0.5)
    sharpe = float(excess.mean() / (port_rets.std() + 1e-8) * ANN ** 0.5)
    cum = np.exp(port_rets.cumsum())
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    common = port_rets.index.intersection(nifty_ser.index)
    if len(common) > 5:
        active = port_rets.loc[common] - nifty_ser.loc[common]
        ir = float(active.mean() / (active.std() + 1e-8) * ANN ** 0.5)
    else:
        ir = float("nan")
    cum_ret = float(np.exp(port_rets.sum()) - 1)
    return dict(ann_return=round(ann_ret,4), sharpe=round(sharpe,4),
                info_ratio=round(ir,4), max_drawdown=round(max_dd,4),
                cum_return=round(cum_ret,4))

stgat = pd.read_csv("outputs/results/stgat_returns.csv", index_col=0, parse_dates=True).iloc[:,0]
blml  = pd.read_csv("outputs/results/bl_ml_returns.csv", index_col=0, parse_dates=True).iloc[:,0]
nifty = pd.read_csv("outputs/results/nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:,0]

blind_start, blind_end = "2025-01-01", "2026-03-30"
stgat_b = stgat.loc[blind_start:blind_end]
blml_b  = blml.loc[blind_start:blind_end]
nifty_b = nifty.loc[blind_start:blind_end]

print("="*62)
print("  INDIVIDUAL STRATEGIES (blind holdout)")
print("="*62)
m_stgat = compute_metrics(stgat_b, nifty_b)
m_blml  = compute_metrics(blml_b, nifty_b)
print(f"  STGAT-DPO : IR={m_stgat['info_ratio']:.4f}  Sharpe={m_stgat['sharpe']:.4f}  AnnRet={m_stgat['ann_return']:.4f}")
print(f"  BL-ML     : IR={m_blml['info_ratio']:.4f}  Sharpe={m_blml['sharpe']:.4f}  AnnRet={m_blml['ann_return']:.4f}")

print("\n" + "="*62)
print("  STATIC BLENDS (blind holdout)")
print("="*62)
for w_stgat in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    blend = w_stgat * stgat_b + (1 - w_stgat) * blml_b
    m = compute_metrics(blend, nifty_b)
    print(f"  STGAT={w_stgat:.1f} / BL-ML={1-w_stgat:.1f}  ->  IR={m['info_ratio']:>8.4f}  Sharpe={m['sharpe']:>8.4f}  AnnRet={m['ann_return']:>8.4f}  MaxDD={m['max_drawdown']:>8.4f}")

print("\nDone.")
