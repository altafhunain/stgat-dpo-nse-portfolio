"""
training/rebuild_table6.py
============================
Rebuilds Table 6 (quarterly blind-holdout breakdown) from the confirmed
final stgat_returns.csv (point-in-time + interop-threading fixes applied),
computing quarterly returns AND quarterly Information Ratio properly
(previously left as "n/a" pending this).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

stgat = pd.read_csv("outputs/results/stgat_returns.csv", index_col=0, parse_dates=True)
stgat.columns = ["ret"]  # log returns
nifty = pd.read_csv("outputs/results/nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]

blind_start, blind_end = "2025-01-01", "2026-03-30"
stgat_b = stgat.loc[blind_start:blind_end, "ret"]
nifty_b = nifty.loc[blind_start:blind_end]

# Align on common dates
common = stgat_b.index.intersection(nifty_b.index)
stgat_b = stgat_b.loc[common]
nifty_b = nifty_b.loc[common]

print(f"Blind holdout rows: {len(stgat_b)}")
print(f"Date range: {stgat_b.index.min()} to {stgat_b.index.max()}")

# Sanity check against confirmed full-period numbers (Ann.Ret=-0.0449, IR=-0.3157)
full_cum = np.exp(stgat_b.sum()) - 1
full_ann = float(stgat_b.mean() * 252)
active_full = stgat_b - nifty_b
full_ir = float(active_full.mean() / (active_full.std() + 1e-8) * np.sqrt(252))
print(f"\nSanity check vs confirmed pipeline output:")
print(f"  Cumulative return : {full_cum:.4f}")
print(f"  Annualised return : {full_ann:.4f}  (confirmed: -0.0449)")
print(f"  Full-period IR    : {full_ir:.4f}  (confirmed: -0.3157)")

# Quarterly breakdown
quarters = [
    ("2025-01-01", "2025-03-31", "Q1 2025"),
    ("2025-04-01", "2025-06-30", "Q2 2025"),
    ("2025-07-01", "2025-09-30", "Q3 2025"),
    ("2025-10-01", "2025-12-31", "Q4 2025"),
    ("2026-01-01", "2026-03-30", "Q1 2026"),
]

print("\n" + "="*80)
print("  TABLE 6 (REBUILT) — Quarterly blind-holdout breakdown")
print("="*80)
print(f"  {'Period':<10} {'STGAT-DPO IR':>13} {'Nifty Ret':>10} {'STGAT-DPO Ret':>14} {'N days':>7}")
print("  " + "-"*76)

rows = []
for start, end, label in quarters:
    s = stgat_b.loc[start:end]
    n = nifty_b.loc[start:end]
    n_days = len(s)
    if n_days < 5:
        print(f"  {label:<10}  WARNING: only {n_days} days, skipping IR (too small for meaningful std)")
        continue
    stgat_ret = np.exp(s.sum()) - 1
    nifty_ret = np.exp(n.sum()) - 1 if not n.empty else float('nan')
    active = s - n
    ir = float(active.mean() / (active.std() + 1e-8) * np.sqrt(252))
    rows.append((label, ir, nifty_ret, stgat_ret, n_days))
    print(f"  {label:<10} {ir:>13.3f} {nifty_ret:>10.4f} {stgat_ret:>14.4f} {n_days:>7}")

print("\n  Note: quarterly IR uses ~50-65 daily observations per quarter; treat as a")
print("  noisy, small-sample estimate (same caveat as the 6-fold walk-forward CI).")
print("  Quarterly IRs will NOT average to the full-period IR (-0.3157) since that")
print("  figure is computed on the full 309-day pooled active-return series, not")
print("  as a mean of quarterly IRs -- same convention distinction as Section 4.4.")

df = pd.DataFrame(rows, columns=["period", "stgat_dpo_ir", "nifty_ret", "stgat_dpo_ret", "n_days"])
df.to_csv("outputs/results/table6_rebuilt.csv", index=False)
print("\nSaved: outputs/results/table6_rebuilt.csv")
print("Done.")
