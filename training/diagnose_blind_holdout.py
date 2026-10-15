"""
training/diagnose_blind_holdout.py
Reproduces the Section 4.6 diagnostics ("Diagnosing the Blind-Holdout Result") from released artifacts only:
  outputs/results/{regimes_Blind2025, nifty50_returns, weights_<fold>, stgat_returns, benchmark_returns, stgat_fold_results}
Usage: python training/diagnose_blind_holdout.py
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

R = Path(__file__).resolve().parent.parent / "outputs" / "results"
START, END = "2026-01-01", "2026-08-28"
FOLDS = [f"Fold{i}" for i in range(1, 10)] + ["Blind2025"]

regime = pd.read_csv(R / "regimes_Blind2025.csv", index_col=0, parse_dates=True)["regime"].loc[START:END]
nifty = pd.read_csv(R / "nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:, 0].loc[START:END]

print(f"1. Causal regime labels over the {len(regime)}-day blind holdout:", regime.value_counts().to_dict(),
      "| 'bear' days:", int((regime == 'bear').sum()))
for name, mask in (("Jan-Feb 2026 (sideways)", (nifty.index.month <= 2)),
                   ("bull-labelled days (Mar-Jul)", nifty.index.isin(regime[regime == 'bull'].index)),
                   ("Aug 2026 (sideways)", (nifty.index.month == 8))):
    x = nifty[mask]
    print(f"   Nifty 50 | {name:30s} n={len(x):3d}  cumulative {np.exp(x.sum()) - 1:+.1%}  annualised vol {x.std() * np.sqrt(252):.1%}")

W = {f: pd.read_csv(R / f"weights_{f}.csv", index_col=0, parse_dates=True) for f in FOLDS}
b = W["Blind2025"]; en = 1 / (b ** 2).sum(axis=1)
seg = {"Jan-Feb": en[b.index.month <= 2], "Mar-Jul": en[(b.index.month >= 3) & (b.index.month <= 7)], "Aug": en[b.index.month == 8]}
print("\n2. Portfolio concentration, blind holdout (effective-N = 1/sum w^2, out of 51):",
      {k: round(v.mean(), 1) for k, v in seg.items()}, "| per-rebalance range", round(en.min(), 1), "-", round(en.max(), 1))
print(f"   maximum single-stock weight per rebalance: {b.max(axis=1).min():.1%} - {b.max(axis=1).max():.1%}")

rt = pd.read_csv(R / "stgat_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
rt = rt[~rt.index.duplicated(keep="first")].sort_index()
ew = pd.read_parquet(R / "benchmark_returns.parquet")["EW"]; ew = ew[~ew.index.duplicated(keep="first")].sort_index()
reb = list(b.index) + [pd.Timestamp(END) + pd.Timedelta(days=1)]
diffs = []
for i in range(len(reb) - 1):
    idx = rt.index[(rt.index >= reb[i]) & (rt.index < reb[i + 1])]
    diffs.append((np.exp(rt[idx].sum()) - 1) - (np.exp(ew.reindex(idx).fillna(0).sum()) - 1))
print("\n3. STGAT-DPO minus Equal Weight, by rebalance period:", [f"{d:+.2%}" for d in diffs],
      f"| max |diff| {max(abs(d) for d in diffs):.2%} | negative in {sum(d < 0 for d in diffs)} of {len(diffs)}")

ir = pd.read_csv(R / "stgat_fold_results.csv").set_index("fold").info_ratio
effn = pd.Series({f: (1 / (W[f] ** 2).sum(axis=1)).mean() for f in FOLDS})
pr, sr = stats.pearsonr(effn, ir.loc[FOLDS]), stats.spearmanr(effn, ir.loc[FOLDS])
print(f"\n4. Mean effective-N vs fold IR across {len(FOLDS)} folds (9 walk-forward + blind): "
      f"Pearson r = {pr[0]:+.3f} (p = {pr[1]:.3f}); Spearman r = {sr[0]:+.3f} (p = {sr[1]:.3f})")

# ---- 5. quarterly and label-level breakdown of STGAT-DPO vs Nifty 50 (Table 9 of the full manuscript) ----
sr = rt.loc[START:END]; nr = nifty.reindex(sr.index).ffill().fillna(0)
def line(name, x, y, extra=""):
    a = x - y
    ir = a.mean() / a.std() * np.sqrt(252)
    print(f"   {name:34s} n={len(x):3d}  Nifty {np.exp(y.sum()) - 1:+.4f}  STGAT-DPO {np.exp(x.sum()) - 1:+.4f}  IR {ir:+.3f} {extra}")
print("\n5. STGAT-DPO vs Nifty 50 (compounded daily returns; IR = annualised mean/std of daily active return)")
for q, (s0, e0) in {"Q1 2026 (Jan-Mar)": ("2026-01-01", "2026-03-31"), "Q2 2026 (Apr-Jun)": ("2026-04-01", "2026-06-30"), "Q3 2026 (Jul-Aug)": ("2026-07-01", END)}.items():
    lab = regime.loc[s0:e0].value_counts().to_dict()
    line(q, sr.loc[s0:e0], nr.loc[s0:e0], f"| label days {lab}")
for L in ("sideways", "bull"):
    m = regime.reindex(sr.index) == L
    line(f"all days labelled '{L}'", sr[m], nr[m])
