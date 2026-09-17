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
stgat = pd.read_csv("outputs/results/stgat_returns.csv", index_col=0, parse_dates=True).iloc[:,0]
blml  = pd.read_csv("outputs/results/bl_ml_returns.csv", index_col=0, parse_dates=True).iloc[:,0]
nifty = pd.read_csv("outputs/results/nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:,0]

# Walk-forward evaluation window only (folds 1-6): 2021-01-01 to 2024-12-31
wf_start, wf_end = "2021-01-01", "2024-12-31"

# Fit HMM on data BEFORE the walk-forward window starts, to avoid look-ahead
# (earliest possible: use 2018-2020 as a pre-fit period, mirroring how the
#  production HMM is fit expanding-window per fold; here we use a single
#  fit on 2018-2020 as a simple, conservative pre-window fit)
hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
hmm_det.fit(nifty.loc["2018-01-01":"2020-12-31"].fillna(0).values)
regimes = hmm_det.predict(nifty.fillna(0).values)
regime_ser = pd.Series(regimes, index=nifty.index[-len(regimes):])

stgat_wf = stgat.loc[wf_start:wf_end]
blml_wf  = blml.loc[wf_start:wf_end]
regime_wf = regime_ser.reindex(stgat_wf.index).ffill().fillna(1).astype(int)

print("Regime distribution across walk-forward period (2021-2024):")
print(regime_wf.value_counts().sort_index())
print("(0=bear, 1=sideways, 2=bull)\n")

print("="*62)
print("  MEAN DAILY RETURN BY REGIME — WALK-FORWARD PERIOD")
print("="*62)
for r, label in [(0,"bear"), (1,"sideways"), (2,"bull")]:
    mask = regime_wf == r
    n = mask.sum()
    if n == 0:
        print(f"  {label:>10} (n={n:>4}): no days in this regime")
        continue
    s_mean = stgat_wf[mask].mean() * 252
    b_mean = blml_wf[mask].mean() * 252
    winner = "BL-ML" if b_mean > s_mean else "STGAT-DPO"
    print(f"  {label:>10} (n={n:>4}): STGAT ann={s_mean:>8.4f}   BL-ML ann={b_mean:>8.4f}   -> {winner} better")

print("\nDone.")
