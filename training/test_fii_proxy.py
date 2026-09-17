import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

usdinr = pd.read_csv("data/raw/usdinr.csv", skiprows=2, index_col=0, parse_dates=True)
usdinr.columns = ["close", "high", "low", "open", "volume"]
usdinr_ret = np.log(usdinr["close"] / usdinr["close"].shift(1))

nifty = pd.read_csv("outputs/results/nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:,0]
stgat = pd.read_csv("outputs/results/stgat_returns.csv", index_col=0, parse_dates=True).iloc[:,0]

common = usdinr_ret.index.intersection(nifty.index)
print(f"Overlapping dates: {len(common)}")

corr_full = usdinr_ret.loc[common].corr(nifty.loc[common])
print(f"\nFull-period correlation, USD/INR return vs Nifty return: {corr_full:.4f}")
print("(negative = rupee strengthens when Nifty rises, i.e. classic FII-inflow signature)")

# Quarterly breakdown matching Table 9's blind-holdout quarters
quarters = [
    ("2025-01-01","2025-03-31","Q1 2025"),
    ("2025-04-01","2025-06-30","Q2 2025"),
    ("2025-07-01","2025-09-30","Q3 2025"),
    ("2025-10-01","2025-12-31","Q4 2025"),
    ("2026-01-01","2026-03-30","Q1 2026"),
]
print("\n" + "="*70)
print("  QUARTERLY: USD/INR MOVE vs NIFTY MOVE vs STGAT-DPO PERFORMANCE")
print("="*70)
print(f"  {'Quarter':<10} {'USDINR chg':>12} {'Nifty ret':>10} {'STGAT ret':>10}  Interpretation")
for start, end, label in quarters:
    usdinr_chg = usdinr_ret.loc[start:end].sum()
    nifty_ret = np.exp(nifty.loc[start:end].sum()) - 1
    stgat_ret = np.exp(stgat.loc[start:end].sum()) - 1
    direction = "INR strengthened (FII inflow signature)" if usdinr_chg < 0 else "INR weakened (FII outflow signature)"
    print(f"  {label:<10} {usdinr_chg:>+12.4f} {nifty_ret:>+10.4f} {stgat_ret:>+10.4f}  {direction}")

print("\nDone.")
