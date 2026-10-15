"""
training/make_full_paper_figures.py
Regenerates Fig. 2 (metrics heatmap), Fig. 7 (bootstrap CIs) and Fig. 10 (sensitivity) of the full manuscript
from released artifacts only (outputs/results/*.csv). Run from the repository root:  python training/make_full_paper_figures.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "outputs" / "results"; F = ROOT / "outputs" / "figures"; F.mkdir(parents=True, exist_ok=True)
FOLDS = [f"Fold{i}" for i in range(1, 10)]
frames = [pd.read_csv(R/"stgat_fold_results.csv"), pd.read_csv(R/"ablation_results.csv"), pd.read_csv(R/"bl_ml_results.csv"),
          pd.read_csv(R/"rl_ppo_results.csv"), pd.read_csv(R/"benchmark_fold_results.csv").drop(columns=["name"])]
A = pd.concat(frames, ignore_index=True); A = A[A.fold.isin(FOLDS)]
ORDER = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT","BL-ML","RL-PPO","EW","BH","RiskPar","InvVol","MinVar","Momentum","MaxSharpe"]
LABEL = {"STGAT-DPO":"STGAT-DPO (proposed)","STGAT-NoHMM":"STGAT-NoHMM","STGAT-NoGAT":"STGAT-NoGAT","BL-ML":"BL-ML","RL-PPO":"RL-PPO","EW":"Equal Weight",
         "BH":"Buy & Hold","RiskPar":"Risk Parity","InvVol":"Inv. Volatility","MinVar":"Min Variance","Momentum":"Momentum","MaxSharpe":"Max Sharpe MVO"}
M = A.groupby("strategy")[["info_ratio","sharpe","ann_return","max_drawdown","win_rate"]].mean().loc[ORDER]

# ---------------- Fig 2: heatmap of relative rank (green = better; max drawdown: less negative = better) ----------------
cols = [("info_ratio","Info Ratio","{:.3f}"),("sharpe","Sharpe","{:.3f}"),("ann_return","Ann. Return","{:.3f}"),("max_drawdown","Max Drawdown","{:.3f}"),("win_rate","Win Rate","{:.3f}")]
rank = pd.DataFrame({c: (M[c].rank(method="average") - 1) / (len(M) - 1) for c,_,_ in cols})
fig, ax = plt.subplots(figsize=(9.2, 6.4)); im = ax.imshow(rank.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
for i, s in enumerate(M.index):
    for j, (c,_,fmt) in enumerate(cols):
        v = rank.iloc[i, j]; ax.text(j, i, fmt.format(M.loc[s, c]), ha="center", va="center", fontsize=9, color="white" if v < 0.12 or v > 0.88 else "black", fontweight="bold" if i == 0 else "normal")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([n for _,n,_ in cols]); ax.set_yticks(range(len(M))); ax.set_yticklabels([LABEL[s] for s in M.index])
ax.add_patch(Rectangle((-0.5,-0.5),len(cols),1,fill=False,edgecolor="#C0392B",lw=2.5))
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03); cb.set_label("Relative rank (green = better)")
plt.tight_layout(); plt.savefig(F/"fig2_metrics_heatmap_12.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------------- Fig 7: bootstrap CIs, all twelve strategies (fresh identical seed per strategy) ----------------
rows = []
for s in ORDER:
    x = A[A.strategy == s].sort_values("fold").info_ratio.values; rng = np.random.default_rng(42)
    bm = np.array([x[rng.integers(0, 9, 9)].mean() for _ in range(10000)]); lo, hi = np.percentile(bm, [2.5, 97.5]); rows.append((s, x.mean(), lo, hi))
fig, ax = plt.subplots(figsize=(8.6, 6.6)); ax.axvspan(0, 1.4, color="#2e7d32", alpha=0.05)
for k, (s, m, lo, hi) in enumerate(rows):
    y = len(rows) - 1 - k; col = "#C0392B" if s == "STGAT-DPO" else ("#1f4e79" if hi < 0 else "#7f8c8d")
    ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round"); ax.plot(m, y, "o", color=col, ms=8)
ax.axvline(0, color="black", ls="--", lw=1.2); ax.set_yticks(range(len(rows))); ax.set_yticklabels([LABEL[s] for s,*_ in rows][::-1])
ax.set_xlabel("Mean walk-forward Information Ratio – 95% bootstrap CI (nine fold-level IRs, 10,000 resamples, seed 42)", fontsize=9); ax.grid(axis="x", alpha=0.3, ls="--")
ax.annotate("Only CI entirely above zero", xy=(rows[0][2], len(rows)-1), xytext=(-1.45, len(rows)-1.9), fontsize=8.5, color="#C0392B", arrowprops=dict(arrowstyle="->", color="#C0392B"))
ax.annotate("Entirely below zero", xy=(rows[-1][3], 0), xytext=(-0.25, 0.55), fontsize=8.5, color="#1f4e79", arrowprops=dict(arrowstyle="->", color="#1f4e79"))
ax.spines[["top","right"]].set_visible(False); plt.tight_layout(); plt.savefig(F/"fig7_bootstrap_ci_12.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------------- Fig 10: sensitivity (mean IR +/- 1 std, and mean Sharpe) across nine folds ----------------
S = pd.read_csv(R/"sensitivity_results_allfolds_v2.csv")
g = S.groupby(["sensitivity","param"]).agg(IR=("info_ratio","mean"), sd=("info_ratio","std"), Sh=("sharpe","mean")).reset_index()
cfg = [("transaction_cost","Transaction cost (bps per side)",10),("threshold_shift","Threshold shift added to 0.25 / 0.35 / 0.45",0.0),
       ("lookback","Lookback window (days)",20),("max_weight","Max weight cap (inference-time clip)",0.15)]
fig, axes = plt.subplots(2, 2, figsize=(12, 9)); fig.patch.set_facecolor("white")
for ax, (k, xl, base) in zip(axes.flatten(), cfg):
    d = g[g.sensitivity == k].sort_values("param"); x = d.param.values
    ax.plot(x, d.IR, color="#C0392B", lw=2, marker="o", ms=5, label="Mean IR across 9 folds", zorder=5)
    ax.fill_between(x, d.IR - d.sd, d.IR + d.sd, color="#C0392B", alpha=0.13, label="IR ± 1 std across folds")
    ax.plot(x, d.Sh, color="#1f4e79", lw=1.8, ls="--", marker="s", ms=4, label="Mean Sharpe across 9 folds", zorder=4)
    ax.axvline(base, color="#888888", lw=1.2, ls=":", label=f"Trained setting ({base})"); ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel("Mean ratio", fontsize=9); ax.legend(fontsize=7, framealpha=0.9, loc="best"); ax.grid(True, alpha=0.3, ls="--"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(F/"fig10_sensitivity_v2.png", dpi=300, bbox_inches="tight"); plt.close()
print("saved: fig2_metrics_heatmap_12.png, fig7_bootstrap_ci_12.png, fig10_sensitivity_v2.png ->", F)
for s,m,lo,hi in rows: print(f"  {s:12s} mean {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
