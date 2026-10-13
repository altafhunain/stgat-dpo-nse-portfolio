import numpy as np, pandas as pd
from pathlib import Path
OUT = Path('outputs/results')
rows = []
for f in [f"Fold{i}" for i in range(1, 10)] + ["Blind2025"]:
    w = pd.read_csv(OUT / f"weights_{f}.csv", index_col=0)
    d = w.diff().abs().sum(axis=1).iloc[1:]      # sum|dw| per rebalance, excluding initial buy-in
    rows.append(dict(fold=f, rebalances=len(w), traded_per_rebalance=d.mean(),
                     one_way_turnover=d.mean() / 2,
                     cost_bps_per_rebalance=d.mean() * 10,
                     cost_bps_per_year=d.mean() * 10 * 12))
df = pd.DataFrame(rows)
print(df.round(4).to_string(index=False))
wf = df[df.fold.str.startswith("Fold")]
print("\nWalk-forward mean: traded notional/rebalance = %.4f | one-way turnover = %.4f | cost drag = %.1f bps/yr"
      % (wf.traded_per_rebalance.mean(), wf.one_way_turnover.mean(), wf.cost_bps_per_year.mean()))
df.to_csv(OUT / "turnover_cost_drag_stgat_dpo.csv", index=False)
