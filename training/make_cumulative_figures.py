"""
training/make_cumulative_figures.py
Regenerates the cumulative-growth, drawdown and daily-return-distribution figures from released artifacts, correcting an error in the
earlier versions: the per-fold return files overlap by six months (988 duplicated dates), and the old figures compounded the stacked
series, counting overlapped days twice. Here each date is counted once (the earlier fold's value is kept, as in run_stat_tests.py).
Also writes a wide-aspect bootstrap-CI variant for two-column layouts.   Run from the repository root.
"""
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "outputs" / "results"; F = ROOT / "outputs" / "figures"; F.mkdir(parents=True, exist_ok=True)
START, END = "2021-01-01", "2026-08-28"
def load(path):
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].sort_index(); return s[~s.index.duplicated(keep="first")].loc[START:END]
S = {"STGAT-DPO": load(R/"stgat_returns.csv"), "STGAT-NoHMM": load(R/"ablation_NoHMM_returns.csv"), "STGAT-NoGAT": load(R/"ablation_NoGAT_returns.csv"),
     "BL-ML": load(R/"bl_ml_returns.csv"), "RL-PPO": load(R/"rl_ppo_returns.csv")}
B = pd.read_parquet(R/"benchmark_returns.parquet").sort_index(); B = B[~B.index.duplicated(keep="first")].loc[START:END]
for c, n in (("EW","Equal Weight"),("BH","Buy & Hold"),("RiskPar","Risk Parity"),("InvVol","Inv. Volatility"),("MinVar","Min Variance"),("Momentum","Momentum"),("MaxSharpe","Max Sharpe MVO")): S[n] = B[c]
nifty = pd.read_csv(R/"nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:, 0].sort_index().loc[START:END]
idx = S["STGAT-DPO"].index; S = {k: v.reindex(idx).fillna(0) for k, v in S.items()}; nifty = nifty.reindex(idx).fillna(0)
STYLE = {"Nifty 50 Index": dict(c="black", ls="--", lw=2.2, z=6), "STGAT-DPO": dict(c="#C0392B", ls="-", lw=3.2, z=7), "STGAT-NoHMM": dict(c="#E67E22", ls=":", lw=1.6, z=3),
         "STGAT-NoGAT": dict(c="#8E44AD", ls=":", lw=1.6, z=3), "BL-ML": dict(c="#1ABC9C", ls="-.", lw=1.6, z=3), "RL-PPO": dict(c="#2ECC71", ls=":", lw=1.6, z=3),
         "Equal Weight": dict(c="#5DADE2", ls="-", lw=1.4, z=4), "Min Variance": dict(c="#58D68D", ls="-", lw=1.2, z=3), "Risk Parity": dict(c="#16A085", ls="-", lw=1.2, z=3),
         "Momentum": dict(c="#F0B27A", ls="-", lw=1.2, z=3), "Buy & Hold": dict(c="#7F8C8D", ls="-", lw=1.2, z=3), "Inv. Volatility": dict(c="#34495E", ls="-", lw=1.2, z=3),
         "Max Sharpe MVO": dict(c="#A04000", ls="-", lw=1.4, z=3)}
series = {"Nifty 50 Index": nifty, **S}
# causal regime label per date: label from the first fold whose test window covers the date (blind holdout: its own fold)
folds = yaml.safe_load(open(ROOT/"configs"/"folds.yaml"))["folds"]; lab = pd.Series(index=idx, dtype=object)
for name in [f"Fold{i}" for i in range(1, 10)] + ["Blind2025"]:
    rg = pd.read_csv(R/f"regimes_{name}.csv", index_col=0, parse_dates=True)["regime"]; m = (idx >= folds[name]["test_start"]) & (idx <= folds[name]["test_end"]) & lab.isna().values
    lab[m] = rg.reindex(idx[m]).values
RC = {"bear": "#E74C3C", "sideways": "#F1C40F", "bull": "#2ECC71"}
def shade(ax):
    run_start, prev = idx[0], lab.iloc[0]
    for d, l in list(lab.items())[1:] + [(idx[-1], None)]:
        if l != prev:
            if prev in RC: ax.axvspan(run_start, d, color=RC[prev], alpha=0.07, lw=0)
            run_start, prev = d, l
    ax.axvline(pd.Timestamp("2026-01-01"), color="#555555", ls=":", lw=1.3)
cum = {k: 100 * np.exp(v.cumsum()) for k, v in series.items()}
fig, ax = plt.subplots(figsize=(12.9, 5.4)); shade(ax)
for k, v in cum.items(): ax.plot(v.index, v.values, label=k if k != "STGAT-DPO" else "STGAT-DPO (proposed)", color=STYLE[k]["c"], ls=STYLE[k]["ls"], lw=STYLE[k]["lw"], zorder=STYLE[k]["z"])
ax.text(pd.Timestamp("2026-01-12"), ax.get_ylim()[1]*0.93, "blind\nholdout", fontsize=8, color="#555555", va="top")
h, l = ax.get_legend_handles_labels(); h += [Patch(color=RC[r], alpha=0.25) for r in RC]; l += ["Bear", "Sideways", "Bull"]
ax.legend(h, l, fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9); ax.set_ylabel("Portfolio value (base = 100)"); ax.grid(alpha=0.3, ls="--"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(F/"fig3_cumulative_wealth_dedup.png", dpi=300, bbox_inches="tight"); plt.close()
fig, ax = plt.subplots(figsize=(12.9, 4.9)); shade(ax)
for k, v in cum.items(): ax.plot(v.index, (v / v.cummax() - 1) * 100, color=STYLE[k]["c"], ls=STYLE[k]["ls"], lw=STYLE[k]["lw"], zorder=STYLE[k]["z"], label=k if k != "STGAT-DPO" else "STGAT-DPO (proposed)")
ax.legend(fontsize=7.5, ncol=4, loc="lower left", framealpha=0.9); ax.set_ylabel("Drawdown (%)"); ax.grid(alpha=0.3, ls="--"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(F/"fig4_drawdown_dedup.png", dpi=300, bbox_inches="tight"); plt.close()
vio = ["Nifty 50 Index","STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT","BL-ML","RL-PPO","Equal Weight","Min Variance","Momentum"]
fig, ax = plt.subplots(figsize=(12.9, 5.4)); parts = ax.violinplot([series[k].values * 100 for k in vio], showmeans=True, showextrema=False, widths=0.85)
for pc, k in zip(parts["bodies"], vio): pc.set_facecolor(STYLE[k]["c"]); pc.set_alpha(0.75)
parts["cmeans"].set_color("black"); ax.set_xticks(range(1, len(vio)+1)); ax.set_xticklabels(["Nifty 50" if k.startswith("Nifty") else k for k in vio], rotation=20, ha="right")
ax.axhline(0, color="#666666", lw=0.8, ls="--"); ax.set_ylabel("Daily return (%)"); ax.grid(alpha=0.3, ls="--"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(F/"fig5_return_distribution_dedup.png", dpi=300, bbox_inches="tight"); plt.close()
# wide-aspect bootstrap CI variant (two-column layouts)
FOLDS = [f"Fold{i}" for i in range(1, 10)]
fr = [pd.read_csv(R/"stgat_fold_results.csv"), pd.read_csv(R/"ablation_results.csv"), pd.read_csv(R/"bl_ml_results.csv"), pd.read_csv(R/"rl_ppo_results.csv"), pd.read_csv(R/"benchmark_fold_results.csv").drop(columns=["name"])]
A = pd.concat(fr, ignore_index=True); A = A[A.fold.isin(FOLDS)]
ORDER = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT","BL-ML","RL-PPO","EW","BH","RiskPar","InvVol","MinVar","Momentum","MaxSharpe"]
LAB = {"STGAT-DPO":"STGAT-DPO (proposed)","EW":"Equal Weight","BH":"Buy & Hold","RiskPar":"Risk Parity","InvVol":"Inv. Volatility","MinVar":"Min Variance","MaxSharpe":"Max Sharpe MVO"}
rows = []
for s in ORDER:
    x = A[A.strategy == s].sort_values("fold").info_ratio.values; rng = np.random.default_rng(42)
    bm = np.array([x[rng.integers(0, 9, 9)].mean() for _ in range(10000)]); lo, hi = np.percentile(bm, [2.5, 97.5]); rows.append((s, x.mean(), lo, hi))
fig, ax = plt.subplots(figsize=(8.6, 5.2)); ax.axvspan(0, 1.4, color="#2e7d32", alpha=0.05)
for k, (s, m, lo, hi) in enumerate(rows):
    y = len(rows) - 1 - k; col = "#C0392B" if s == "STGAT-DPO" else ("#1f4e79" if hi < 0 else "#7f8c8d"); ax.plot([lo, hi], [y, y], color=col, lw=2.2); ax.plot(m, y, "o", color=col, ms=7)
ax.axvline(0, color="black", ls="--", lw=1.2); ax.set_yticks(range(len(rows))); ax.set_yticklabels([LAB.get(s, s) for s, *_ in rows][::-1], fontsize=9)
ax.set_xlabel("Mean walk-forward IR – 95% bootstrap CI (nine fold-level IRs, 10,000 resamples)", fontsize=8.5); ax.grid(axis="x", alpha=0.3, ls="--")
ax.annotate("Only CI entirely above zero", xy=(rows[0][2], len(rows)-1), xytext=(-1.45, len(rows)-2.0), fontsize=8, color="#C0392B", arrowprops=dict(arrowstyle="->", color="#C0392B"))
ax.annotate("Entirely below zero", xy=(rows[-1][3], 0), xytext=(-0.3, 0.7), fontsize=8, color="#1f4e79", arrowprops=dict(arrowstyle="->", color="#1f4e79"))
ax.spines[["top","right"]].set_visible(False); plt.tight_layout(); plt.savefig(F/"fig5_bootstrap_ci_12_wide.png", dpi=300, bbox_inches="tight"); plt.close()
print("saved: fig3_cumulative_wealth_dedup.png, fig4_drawdown_dedup.png, fig5_return_distribution_dedup.png, fig5_bootstrap_ci_12_wide.png ->", F)
for k, v in cum.items(): 
    dd = (v / v.cummax() - 1).min()
    print(f"  {k:16s} final value {v.iloc[-1]:7.1f}   max drawdown {dd:+.1%}")
