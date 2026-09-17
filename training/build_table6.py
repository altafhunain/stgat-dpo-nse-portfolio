import pandas as pd
import numpy as np

df = pd.read_csv("outputs/results/stgat_returns.csv", index_col=0, parse_dates=True)
df.columns = ["ret"]  # log returns, per get_close_returns()

blind = df.loc["2025-01-01":"2026-03-30"]
print(f"Blind holdout rows: {len(blind)}")
print(f"Date range: {blind.index.min()} to {blind.index.max()}")

# Full-period: match compute_metrics() exactly (log-return compounding)
cum_ret = np.exp(blind["ret"].sum()) - 1
ann_ret = float(blind["ret"].mean() * 252)
print(f"\nFull blind period (log-return convention, matches training script):")
print(f"  Cumulative return : {cum_ret:.4f}")
print(f"  Annualized return : {ann_ret:.4f}")

# Quarterly: sum log returns within quarter, then exp() to display as simple return
quarterly_log = blind["ret"].resample("QE").sum()
quarterly_simple = np.exp(quarterly_log) - 1
print(f"\nQuarterly returns (simple-return equivalent):")
for date, q_ret in quarterly_simple.items():
    print(f"  {date.year}-Q{date.quarter}: {q_ret:.4f}")

# Verify: compounding the quarterly simple returns should reconcile to cum_ret
compounded = (1 + quarterly_simple).prod() - 1
print(f"\nCompounded from quarters : {compounded:.4f}")
print(f"Direct cumulative return : {cum_ret:.4f}")
print(f"Match: {abs(compounded - cum_ret) < 1e-6}")

out = quarterly_simple.copy()
out.index = [f"{d.year}-Q{d.quarter}" for d in quarterly_simple.index]
out.to_csv("outputs/results/table6_quarterly.csv")
print("\nSaved: outputs/results/table6_quarterly.csv")
