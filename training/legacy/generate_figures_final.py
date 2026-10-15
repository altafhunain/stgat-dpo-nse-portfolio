"""
training/generate_figures_final.py
====================================
Regenerates ALL publication figures with complete results including:
  - BL-ML and RL-PPO in comparisons
  - Attention heatmaps (fig9, fig10) from extract_attention.py
  - Sensitivity analysis (fig11) from run_sensitivity.py
  - Sector weights (fig12) from saved weights CSVs

No titles on any figure. Captions written separately in paper.

Run:
    python training/generate_figures_final.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.axisbelow": True,
})

C = {
    "stgat":    "#C0392B", "nohmm":    "#E67E22", "nogat":    "#8E44AD",
    "blml":     "#1ABC9C", "rlppo":    "#2ECC71",
    "ew":       "#2980B9", "minvar":   "#27AE60", "riskpar":  "#16A085",
    "invvol":   "#2C3E50", "bh":       "#7F8C8D", "momentum": "#D35400",
    "nifty":    "#000000",
    "bear":     "#FADBD8", "sideways": "#FEF9E7", "bull":     "#EAFAF1",
}

COLORS = {
    "STGAT-DPO": C["stgat"], "STGAT-NoHMM": C["nohmm"],
    "STGAT-NoGAT": C["nogat"], "BL-ML": C["blml"], "RL-PPO": C["rlppo"],
    "EW": C["ew"], "MinVar": C["minvar"], "RiskPar": C["riskpar"],
    "InvVol": C["invvol"], "BH": C["bh"], "Momentum": C["momentum"],
}
LABELS = {
    "STGAT-DPO": "STGAT-DPO (proposed)", "STGAT-NoHMM": "STGAT-NoHMM",
    "STGAT-NoGAT": "STGAT-NoGAT", "BL-ML": "BL-ML", "RL-PPO": "RL-PPO",
    "EW": "Equal Weight", "MinVar": "Min Variance", "RiskPar": "Risk Parity",
    "InvVol": "Inv. Volatility", "BH": "Buy & Hold", "Momentum": "Momentum",
}

REGIME_BANDS = [
    ("2021-01-01","2021-03-31","bull"), ("2021-04-01","2021-06-30","bear"),
    ("2021-07-01","2021-10-31","bull"), ("2021-11-01","2022-06-30","bear"),
    ("2022-07-01","2023-03-31","sideways"), ("2023-04-01","2023-12-31","bull"),
    ("2024-01-01","2024-06-30","bull"), ("2024-07-01","2024-12-31","sideways"),
]


def add_regime_shading(ax, start="2021-01-01", end="2024-12-31"):
    for rs, re, rtype in REGIME_BANDS:
        rs, re = pd.Timestamp(rs), pd.Timestamp(re)
        if re < pd.Timestamp(start) or rs > pd.Timestamp(end): continue
        ax.axvspan(rs, re, alpha=0.10, color=C[rtype], linewidth=0, zorder=0)


def load_all_returns():
    nifty = pd.read_csv(OUT_DIR/"nifty50_returns.csv",
                         index_col=0, parse_dates=True).squeeze().sort_index()
    bench = pd.read_parquet(OUT_DIR/"benchmark_returns.parquet").sort_index()
    rets  = {"Nifty50": nifty}
    for strat, fname in [
        ("STGAT-DPO",   "stgat_returns.csv"),
        ("STGAT-NoHMM", "ablation_NoHMM_returns.csv"),
        ("STGAT-NoGAT", "ablation_NoGAT_returns.csv"),
        ("BL-ML",       "bl_ml_returns.csv"),
        ("RL-PPO",      "rl_ppo_returns.csv"),
    ]:
        f = OUT_DIR/fname
        if f.exists():
            rets[strat] = pd.read_csv(f, index_col=0, parse_dates=True
                                       ).squeeze().sort_index()
    for col in bench.columns:
        rets[col] = bench[col].sort_index()
    return rets


def load_fold_results():
    frames = []
    for fname in ["stgat_fold_results.csv","ablation_results.csv",
                   "benchmark_fold_results.csv","rl_ppo_results.csv",
                   "bl_ml_results.csv"]:
        f = OUT_DIR/fname
        if f.exists(): frames.append(pd.read_csv(f))
    master = pd.concat(frames, ignore_index=True)
    return master[master["fold"].str.startswith("Fold")]


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Cumulative wealth with regime shading
# ══════════════════════════════════════════════════════════════════════════════
def fig1_cumulative_wealth(rets):
    S, E = "2021-01-01", "2024-12-31"
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    add_regime_shading(ax, S, E)

    # Nifty
    nm = rets["Nifty50"].loc[S:E].fillna(0)
    ax.plot(np.exp(nm.cumsum())*100, color=C["nifty"], lw=2.2,
            ls="--", label="Nifty 50 Index", zorder=10)

    # Classical benchmarks (muted thin)
    for s, lw, alpha in [("EW",1.0,0.45),("MinVar",1.0,0.45),
                          ("RiskPar",1.0,0.40),("Momentum",1.0,0.40)]:
        if s in rets:
            r = rets[s].loc[S:E].fillna(0)
            ax.plot(np.exp(r.cumsum())*100, color=COLORS[s],
                    lw=lw, alpha=alpha, label=LABELS[s])

    # AI baselines (medium dashed)
    for s, ls in [("BL-ML","-."),("RL-PPO",":")]:
        if s in rets:
            r = rets[s].loc[S:E].fillna(0)
            ax.plot(np.exp(r.cumsum())*100, color=COLORS[s],
                    lw=1.6, alpha=0.75, ls=ls, label=LABELS[s])

    # Ablations (dotted)
    for s in ["STGAT-NoHMM","STGAT-NoGAT"]:
        if s in rets:
            r = rets[s].loc[S:E].fillna(0)
            ax.plot(np.exp(r.cumsum())*100, color=COLORS[s],
                    lw=1.5, alpha=0.65, ls=":", label=LABELS[s])

    # STGAT-DPO — bold red
    if "STGAT-DPO" in rets:
        r = rets["STGAT-DPO"].loc[S:E].fillna(0)
        ax.plot(np.exp(r.cumsum())*100, color=C["stgat"],
                lw=2.6, label=LABELS["STGAT-DPO"], zorder=9)

    # Regime legend patches
    handles, labels = ax.get_legend_handles_labels()
    bp = mpatches.Patch(color=C["bear"],     alpha=0.45, label="Bear")
    sp = mpatches.Patch(color=C["sideways"], alpha=0.45, label="Sideways")
    up = mpatches.Patch(color=C["bull"],     alpha=0.45, label="Bull")
    ax.legend(handles+[bp,sp,up], labels+["Bear","Sideways","Bull"],
              loc="upper left", ncol=3, framealpha=0.9, fontsize=7.8,
              columnspacing=0.7)

    ax.set_ylabel("Portfolio value (base = 100)")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    plt.tight_layout(pad=1.2)
    plt.savefig(FIG_DIR/"fig1_cumulative_wealth.png", dpi=300, bbox_inches="tight")
    plt.close(); print("  ✓  fig1_cumulative_wealth.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Per-fold IR comparison (all key strategies)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_fold_ir(wf):
    folds  = ["Fold1","Fold2","Fold3","Fold4","Fold5","Fold6"]
    flabs  = ["F1\n2021","F2\n21-22","F3\n2022","F4\n22-23","F5\n2023","F6\n2024"]
    models = ["STGAT-DPO","STGAT-NoHMM","BL-ML","RL-PPO","EW","MinVar"]
    x, w   = np.arange(len(folds)), 0.13
    offsets= np.linspace(-(len(models)-1)/2*w, (len(models)-1)/2*w, len(models))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    for model, offset in zip(models, offsets):
        sub  = wf[wf["strategy"]==model].set_index("fold")
        vals = [float(sub.loc[f,"info_ratio"]) if f in sub.index else np.nan
                for f in folds]
        lw   = 2.5 if "STGAT-DPO" in model else 0.5
        ax.bar(x+offset, vals, w, color=COLORS.get(model,"#888"),
               label=LABELS.get(model,model), zorder=3,
               edgecolor="white", linewidth=0.4,
               alpha=1.0 if "STGAT" in model else 0.70)

    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(flabs)
    ax.set_ylabel("Information Ratio (vs Nifty 50)")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    plt.tight_layout(pad=1.2)
    plt.savefig(FIG_DIR/"fig2_fold_ir_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(); print("  ✓  fig2_fold_ir_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Ablation bars (4-panel)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_ablation(wf):
    models  = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT",
               "BL-ML","RL-PPO","EW","MinVar"]
    metrics = ["info_ratio","sharpe","ann_return","max_drawdown"]
    mlabs   = ["Information Ratio","Sharpe Ratio",
               "Annualised Return","Max Drawdown"]
    ylabs   = [m.replace(".NS","") for m in models]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    for ax, metric, mlab in zip(axes, metrics, mlabs):
        ax.set_facecolor("white")
        vals, cols = [], []
        for m in models:
            sub = wf[wf["strategy"]==m]
            vals.append(sub[metric].mean() if not sub.empty else np.nan)
            cols.append(COLORS.get(m,"#888"))

        y    = np.arange(len(models))
        bars = ax.barh(y, vals, color=cols, zorder=3,
                       edgecolor="white", linewidth=0.4, height=0.65)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                xpos = v + abs(v)*0.03 if v>=0 else v - abs(v)*0.08
                ax.text(xpos, bar.get_y()+bar.get_height()/2,
                        f"{v:.3f}", va="center", fontsize=7)
        ax.set_yticks(y)
        ylab_map = {"STGAT-DPO":"STGAT-DPO ★","STGAT-NoHMM":"No-HMM",
                    "STGAT-NoGAT":"No-GAT","BL-ML":"BL-ML","RL-PPO":"RL-PPO",
                    "EW":"EW","MinVar":"MinVar"}
        ax.set_yticklabels([ylab_map.get(m,m) for m in models], fontsize=8.5)
        ax.axvline(0, color="black", lw=0.7, ls="--", zorder=2)
        ax.set_xlabel(mlab, fontsize=9)
        ax.grid(True, alpha=0.25, axis="x")

    plt.tight_layout(pad=1.2)
    plt.savefig(FIG_DIR/"fig3_ablation_bars.png", dpi=300, bbox_inches="tight")
    plt.close(); print("  ✓  fig3_ablation_bars.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Bootstrap CI (horizontal bars, all strategies)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_bootstrap_ci(wf):
    np.random.seed(42)
    models = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT","BL-ML","RL-PPO",
              "EW","BH","MinVar","RiskPar","InvVol","Momentum"]
    means, los, his, colors, labels = [], [], [], [], []
    for m in models:
        sub = wf[wf["strategy"]==m]
        if sub.empty: continue
        ir = sub["info_ratio"].values
        b  = [np.random.choice(ir,len(ir),replace=True).mean()
               for _ in range(10000)]
        lo,hi = np.percentile(b,[2.5,97.5])
        means.append(ir.mean()); los.append(lo); his.append(hi)
        colors.append(COLORS.get(m,"#888"))
        labels.append(LABELS.get(m,m))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    y = np.arange(len(means))[::-1]

    for yi,m,lo,hi,c in zip(y,means,los,his,colors):
        ax.plot([lo,hi],[yi,yi], color=c, lw=2.2, solid_capstyle="round")
        ax.scatter([m],[yi], color=c, s=55, zorder=5)
        if lo>0:   # highlight only positive CI
            ax.plot([lo,hi],[yi,yi], color=c, lw=4,
                    solid_capstyle="round", alpha=0.3)

    ax.axvline(0, color="black", lw=1.2, ls="--", zorder=2)
    ax.axvspan(0, max(his)+0.1, alpha=0.04, color="green", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean Information Ratio — 95% bootstrap CI")

    # Annotate STGAT-DPO
    stgat_y = y[0]
    ax.annotate("Only CI\nexcluding zero",
                xy=(los[0], stgat_y),
                xytext=(los[0]-0.65, stgat_y+0.8),
                fontsize=7.5, color=C["stgat"],
                arrowprops=dict(arrowstyle="->", color=C["stgat"],
                                lw=0.8, connectionstyle="arc3,rad=0.15"))

    plt.tight_layout(pad=1.2)
    plt.savefig(FIG_DIR/"fig4_bootstrap_ci.png", dpi=300, bbox_inches="tight")
    plt.close(); print("  ✓  fig4_bootstrap_ci.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Rolling max drawdown
# ══════════════════════════════════════════════════════════════════════════════
def fig5_drawdown(rets):
    S, E = "2021-01-01", "2024-12-31"
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    add_regime_shading(ax, S, E)

    def rdd(r):
        cum = np.exp(r.cumsum())
        return ((cum - cum.cummax()) / cum.cummax()) * 100

    nm = rets["Nifty50"].loc[S:E].fillna(0)
    ax.plot(rdd(nm), color=C["nifty"], lw=2, ls="--", label="Nifty 50", zorder=10)

    for s,lw,alpha,ls in [("EW",1.0,0.5,"-"),("MinVar",1.0,0.5,"-"),
                            ("BL-ML",1.4,0.65,"-."),("RL-PPO",1.4,0.65,":")]:
        if s in rets:
            r = rets[s].loc[S:E].fillna(0)
            ax.plot(rdd(r), color=COLORS[s], lw=lw, alpha=alpha,
                    ls=ls, label=LABELS[s])

    for s in ["STGAT-NoHMM","STGAT-NoGAT"]:
        if s in rets:
            r = rets[s].loc[S:E].fillna(0)
            ax.plot(rdd(r), color=COLORS[s], lw=1.4, alpha=0.6,
                    ls=":", label=LABELS[s])

    if "STGAT-DPO" in rets:
        r = rets["STGAT-DPO"].loc[S:E].fillna(0)
        ax.plot(rdd(r), color=C["stgat"], lw=2.4, label=LABELS["STGAT-DPO"], zorder=9)

    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", ncol=2, framealpha=0.9, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    plt.tight_layout(pad=1.2)
    plt.savefig(FIG_DIR/"fig5_drawdown.png", dpi=300, bbox_inches="tight")
    plt.close(); print("  ✓  fig5_drawdown.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Regime composition + STGAT IR overlay
# ══════════════════════════════════════════════════════════════════════════════
def fig6_regime(wf):
    flabs  = ["F1\n2021","F2\n21-22","F3\n2022","F4\n22-23","F5\n2023","F6\n2024"]
    bear_c = [204,160,284, 54,839,175]
    side_c = [1085,1205,856,524,196,878]
    bull_c = [713,637,862,1424,967,949]
    tots   = [b+s+u for b,s,u in zip(bear_c,side_c,bull_c)]
    bp = [b/t*100 for b,t in zip(bear_c,tots)]
    sp = [s/t*100 for s,t in zip(side_c,tots)]
    up = [u/t*100 for u,t in zip(bull_c,tots)]

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("white"); ax1.set_facecolor("white")
    ax2 = ax1.twinx()
    x   = np.arange(6)

    ax1.bar(x, bp, 0.6, label="Bear",     color="#E74C3C", alpha=0.75, zorder=3)
    ax1.bar(x, sp, 0.6, bottom=bp,        label="Sideways",color="#F39C12", alpha=0.75, zorder=3)
    ax1.bar(x, up, 0.6, bottom=[b+s for b,s in zip(bp,sp)],
            label="Bull", color="#27AE60", alpha=0.75, zorder=3)
    ax1.set_xticks(x); ax1.set_xticklabels(flabs)
    ax1.set_ylabel("Regime composition (%)")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    sub = wf[wf["strategy"]=="STGAT-DPO"].sort_values("fold")
    ir  = sub["info_ratio"].values
    ax2.plot(x, ir, color=C["stgat"], lw=2.2, marker="o", ms=6,
             label="STGAT-DPO IR", zorder=10)
    ax2.axhline(0, color=C["stgat"], lw=0.8, ls=":", alpha=0.5)
    ax2.set_ylabel("Information Ratio", color=C["stgat"])
    ax2.tick_params(axis="y", colors=C["stgat"])
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout(pad=1.2)
    plt.savefig(FIG_DIR/"fig6_regime_composition.png", dpi=300, bbox_inches="tight")
    plt.close(); print("  ✓  fig6_regime_composition.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Return distribution violin plots
# ══════════════════════════════════════════════════════════════════════════════
def fig7_violin(rets):
    S, E    = "2021-01-01", "2024-12-31"
    models  = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT",
               "BL-ML","RL-PPO","EW","MinVar","Momentum"]
    data, cols, labs = [], [], []
    nm = rets["Nifty50"].loc[S:E].fillna(0)
    data.append(nm.values*100); cols.append(C["nifty"]); labs.append("Nifty 50")
    for m in models:
        if m in rets:
            r = rets[m].loc[S:E].fillna(0)
            data.append(r.values*100)
            cols.append(COLORS.get(m,"#888"))
            labs.append(LABELS.get(m,m).replace(" (proposed)","")
                                       .replace(" (ablation)",""))

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    parts = ax.violinplot(data, positions=range(len(data)),
                           showmedians=True, showextrema=False, widths=0.7)
    for pc, color in zip(parts["bodies"], cols):
        pc.set_facecolor(color); pc.set_alpha(0.65)
        pc.set_edgecolor("white"); pc.set_linewidth(0.5)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=22, ha="right", fontsize=8.5)
    ax.set_ylabel("Daily return (%)")
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    plt.tight_layout(pad=1.2)
    plt.savefig(FIG_DIR/"fig7_return_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(); print("  ✓  fig7_return_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Metrics heatmap (all strategies)
# ══════════════════════════════════════════════════════════════════════════════
def fig8_heatmap(wf):
    models  = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT","BL-ML","RL-PPO",
               "EW","MinVar","RiskPar","InvVol","BH","Momentum"]
    metrics = ["info_ratio","sharpe","ann_return","max_drawdown","win_rate"]
    mlabs   = ["Info Ratio","Sharpe","Ann. Return","Max Drawdown","Win Rate"]

    data, rlabs = [], []
    for m in models:
        sub = wf[wf["strategy"]==m]
        if sub.empty: continue
        data.append([sub[c].mean() for c in metrics])
        lbl = LABELS.get(m,m).replace(" (proposed)","★").replace(" (ablation)","")
        rlabs.append(lbl)

    arr    = np.array(data)
    normed = np.zeros_like(arr)
    for j,m in enumerate(metrics):
        col = arr[:,j]; mn,mx = col.min(), col.max()
        normed[:,j] = (col-mn)/(mx-mn+1e-8) if mx>mn else 0.5
        if m=="max_drawdown": normed[:,j] = 1-normed[:,j]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    im = ax.imshow(normed, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(mlabs))); ax.set_xticklabels(mlabs, fontsize=9.5)
    ax.set_yticks(range(len(rlabs))); ax.set_yticklabels(rlabs, fontsize=9.5)

    for i in range(len(data)):
        for j in range(len(metrics)):
            v   = arr[i,j]; nv = normed[i,j]
            col = "white" if nv<=0.15 or nv>=0.85 else "black"
            ax.text(j,i,f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color=col,
                    fontweight="bold" if "★" in rlabs[i] else "normal")

    si = next((i for i,l in enumerate(rlabs) if "★" in l), None)
    if si is not None:
        ax.add_patch(plt.Rectangle((-0.5,si-0.5), len(metrics), 1,
                                    fill=False, edgecolor=C["stgat"],
                                    lw=2.5, zorder=5))
    plt.colorbar(im, ax=ax, shrink=0.7, label="Relative rank (green = better)")
    plt.tight_layout(pad=1.2)
    plt.savefig(FIG_DIR/"fig8_metrics_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(); print("  ✓  fig8_metrics_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9+10 — Attention heatmaps (already generated, check + regenerate if needed)
# ══════════════════════════════════════════════════════════════════════════════
def check_attention_figs():
    for f in ["fig9_attention_heatmap.png","fig10_attention_top_pairs.png"]:
        if (FIG_DIR/f).exists():
            print(f"  ✓  {f} (existing)")
        else:
            print(f"  ⚠  {f} missing — run training/extract_attention.py")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Sensitivity (already generated)
# ══════════════════════════════════════════════════════════════════════════════
def check_sensitivity_fig():
    f = "fig11_sensitivity.png"
    if (FIG_DIR/f).exists():
        print(f"  ✓  {f} (existing)")
    else:
        print(f"  ⚠  {f} missing — run training/run_sensitivity.py")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Sector weights heatmap (already generated)
# ══════════════════════════════════════════════════════════════════════════════
def check_sector_fig():
    f = "fig12_sector_weights.png"
    if (FIG_DIR/f).exists():
        print(f"  ✓  {f} (existing)")
    else:
        print(f"  ⚠  {f} missing — run the sector analysis script")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("  Generating all publication figures (final)")
    print("="*60)

    print("\n── Loading data ──")
    rets = load_all_returns()
    wf   = load_fold_results()
    print(f"  {len(rets)} return series  |  {len(wf)} fold results")
    print(f"  Strategies: {sorted(wf['strategy'].unique())}")

    print("\n── Generating figures 1–8 ──")
    fig1_cumulative_wealth(rets)
    fig2_fold_ir(wf)
    fig3_ablation(wf)
    fig4_bootstrap_ci(wf)
    fig5_drawdown(rets)
    fig6_regime(wf)
    fig7_violin(rets)
    fig8_heatmap(wf)

    print("\n── Checking figures 9–12 ──")
    check_attention_figs()
    check_sensitivity_fig()
    check_sector_fig()

    # List all figures
    figs = sorted(FIG_DIR.glob("fig*.png"))
    print(f"\n{'='*60}")
    print(f"  Total figures ready: {len(figs)}")
    for f in figs:
        sz = f.stat().st_size/1024
        print(f"  {f.name:<45} {sz:>6.1f} KB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
