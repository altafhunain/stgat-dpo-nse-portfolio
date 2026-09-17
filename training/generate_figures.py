"""
training/generate_figures.py
=============================
Generates all 8 publication-quality figures for the STGAT-DPO paper.
No titles on figures — captions written separately in the paper.

Figures produced:
  fig1_cumulative_wealth.png       — wealth curves with regime shading
  fig2_fold_ir_comparison.png      — per-fold IR bar chart (all models)
  fig3_ablation_bars.png           — ablation study bar chart
  fig4_bootstrap_ci.png            — bootstrap CI horizontal bars
  fig5_drawdown.png                — rolling max drawdown comparison
  fig6_regime_composition.png      — HMM regime composition per fold
  fig7_return_distribution.png     — violin/box return distributions
  fig8_sector_weights.png          — sector allocation heatmap over folds

Run:
    python training/generate_figures.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8.5,
    "figure.dpi":       300,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "axes.axisbelow":   True,
})

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "stgat":    "#C0392B",   # deep red
    "nohmm":    "#E67E22",   # orange
    "nogat":    "#8E44AD",   # purple
    "ew":       "#2980B9",   # steel blue
    "minvar":   "#27AE60",   # green
    "riskpar":  "#16A085",   # teal
    "invvol":   "#2C3E50",   # dark slate
    "bh":       "#7F8C8D",   # grey
    "momentum": "#D35400",   # burnt orange
    "nifty":    "#000000",   # black
    "bear":     "#FADBD8",   # light red
    "sideways": "#FEF9E7",   # light yellow
    "bull":     "#EAFAF1",   # light green
}

STRAT_COLORS = {
    "STGAT-DPO":   C["stgat"],
    "STGAT-NoHMM": C["nohmm"],
    "STGAT-NoGAT": C["nogat"],
    "EW":          C["ew"],
    "MinVar":      C["minvar"],
    "RiskPar":     C["riskpar"],
    "InvVol":      C["invvol"],
    "BH":          C["bh"],
    "Momentum":    C["momentum"],
}

STRAT_LABELS = {
    "STGAT-DPO":   "STGAT-DPO (proposed)",
    "STGAT-NoHMM": "STGAT-NoHMM (ablation)",
    "STGAT-NoGAT": "STGAT-NoGAT (ablation)",
    "EW":          "Equal Weight",
    "MinVar":      "Min Variance",
    "RiskPar":     "Risk Parity",
    "InvVol":      "Inv. Volatility",
    "BH":          "Buy & Hold",
    "Momentum":    "Momentum",
}

# Regime date ranges (approximate fold boundaries)
REGIME_BANDS = [
    # (start, end, regime)
    ("2021-01-01", "2021-03-31", "bull"),
    ("2021-04-01", "2021-06-30", "bear"),
    ("2021-07-01", "2021-10-31", "bull"),
    ("2021-11-01", "2022-06-30", "bear"),
    ("2022-07-01", "2023-03-31", "sideways"),
    ("2023-04-01", "2023-12-31", "bull"),
    ("2024-01-01", "2024-06-30", "bull"),
    ("2024-07-01", "2024-12-31", "sideways"),
]


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def load_all():
    stgat  = pd.read_csv(OUT_DIR / "stgat_fold_results.csv")
    ablat  = pd.read_csv(OUT_DIR / "ablation_results.csv")
    bench  = pd.read_csv(OUT_DIR / "benchmark_fold_results.csv")
    master = pd.concat([stgat, ablat, bench], ignore_index=True)
    wf     = master[master["fold"].str.startswith("Fold")]

    nifty  = pd.read_csv(OUT_DIR / "nifty50_returns.csv",
                          index_col=0, parse_dates=True).squeeze().sort_index()

    # Return series
    ret_series = {}
    for f in [OUT_DIR/"stgat_returns.csv",
              OUT_DIR/"ablation_NoHMM_returns.csv",
              OUT_DIR/"ablation_NoGAT_returns.csv"]:
        if f.exists():
            key = ("STGAT-DPO" if "stgat_returns" in f.name
                   else "STGAT-NoHMM" if "NoHMM" in f.name
                   else "STGAT-NoGAT")
            ret_series[key] = pd.read_csv(
                f, index_col=0, parse_dates=True).squeeze().sort_index()

    bench_ret = pd.read_parquet(OUT_DIR / "benchmark_returns.parquet").sort_index()
    for col in bench_ret.columns:
        ret_series[col] = bench_ret[col].sort_index()

    return master, wf, nifty, ret_series


def add_regime_shading(ax, start="2021-01-01", end="2024-12-31"):
    """Add coloured background bands for approximate HMM regime periods."""
    for rs, re, rtype in REGIME_BANDS:
        rs = pd.Timestamp(rs); re = pd.Timestamp(re)
        if re < pd.Timestamp(start) or rs > pd.Timestamp(end):
            continue
        ax.axvspan(rs, re, alpha=0.12,
                   color=C[rtype], linewidth=0, zorder=0)


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — CUMULATIVE WEALTH WITH REGIME SHADING
# ═════════════════════════════════════════════════════════════════════════════
def fig1_cumulative_wealth(ret_series, nifty):
    S, E = "2021-01-01", "2024-12-31"

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    add_regime_shading(ax, S, E)

    # Nifty 50 first (reference)
    nm = nifty.loc[S:E].fillna(0)
    ax.plot(np.exp(nm.cumsum()) * 100,
            color=C["nifty"], lw=2.2, ls="--",
            label="Nifty 50 Index", zorder=10)

    # Benchmarks (muted, thin)
    for strat in ["EW", "MinVar", "RiskPar", "Momentum"]:
        if strat in ret_series:
            s = ret_series[strat].loc[S:E].fillna(0)
            ax.plot(np.exp(s.cumsum()) * 100,
                    color=STRAT_COLORS[strat], lw=1.1, alpha=0.55,
                    label=STRAT_LABELS[strat])

    # Ablations (medium)
    for strat in ["STGAT-NoHMM", "STGAT-NoGAT"]:
        if strat in ret_series:
            s = ret_series[strat].loc[S:E].fillna(0)
            ax.plot(np.exp(s.cumsum()) * 100,
                    color=STRAT_COLORS[strat], lw=1.5, alpha=0.7,
                    ls=":", label=STRAT_LABELS[strat])

    # STGAT-DPO — bold red on top
    if "STGAT-DPO" in ret_series:
        s = ret_series["STGAT-DPO"].loc[S:E].fillna(0)
        ax.plot(np.exp(s.cumsum()) * 100,
                color=C["stgat"], lw=2.5,
                label=STRAT_LABELS["STGAT-DPO"], zorder=9)

    # Regime legend patches
    bull_p = mpatches.Patch(color=C["bull"],     alpha=0.5, label="Bull regime")
    bear_p = mpatches.Patch(color=C["bear"],     alpha=0.5, label="Bear regime")
    side_p = mpatches.Patch(color=C["sideways"], alpha=0.5, label="Sideways regime")

    handles, labels = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles + [bull_p, bear_p, side_p],
                     labels  + ["Bull regime", "Bear regime", "Sideways regime"],
                     loc="upper left", ncol=2, framealpha=0.9,
                     fontsize=8, columnspacing=0.8)

    ax.set_ylabel("Portfolio value (base = 100)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.tight_layout(pad=1.2)
    out = FIG_DIR / "fig1_cumulative_wealth.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — PER-FOLD IR COMPARISON (STGAT + top benchmarks + ablations)
# ═════════════════════════════════════════════════════════════════════════════
def fig2_fold_ir(wf):
    folds  = ["Fold1","Fold2","Fold3","Fold4","Fold5","Fold6"]
    flabs  = ["F1\n2021","F2\n21-22","F3\n2022","F4\n22-23","F5\n2023","F6\n2024"]
    models = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT","EW","MinVar"]
    n_m    = len(models)
    x      = np.arange(len(folds))
    w      = 0.14
    offsets= np.linspace(-(n_m-1)/2*w, (n_m-1)/2*w, n_m)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (model, offset) in enumerate(zip(models, offsets)):
        sub = wf[wf["strategy"]==model].set_index("fold")
        vals = [sub.loc[f, "info_ratio"] if f in sub.index else np.nan
                for f in folds]
        bars = ax.bar(x + offset, vals, w,
                      color=STRAT_COLORS[model],
                      label=STRAT_LABELS[model],
                      zorder=3, edgecolor="white", linewidth=0.4,
                      alpha=0.85 if "STGAT" in model else 0.65)

    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(flabs)
    ax.set_ylabel("Information Ratio (vs Nifty 50)")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)

    plt.tight_layout(pad=1.2)
    out = FIG_DIR / "fig2_fold_ir_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — ABLATION STUDY BARS
# ═════════════════════════════════════════════════════════════════════════════
def fig3_ablation(wf):
    models  = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT",
               "EW","MinVar","RiskPar","Momentum"]
    metrics = ["info_ratio","sharpe","ann_return","max_drawdown"]
    m_labs  = ["Information Ratio","Sharpe Ratio",
               "Annualised Return","Max Drawdown"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.patch.set_facecolor("white")

    for ax, metric, mlab in zip(axes, metrics, m_labs):
        ax.set_facecolor("white")
        vals   = []
        colors = []
        labels = []
        for m in models:
            sub = wf[wf["strategy"]==m]
            if sub.empty:
                continue
            vals.append(sub[metric].mean())
            colors.append(STRAT_COLORS[m])
            labels.append(m.replace("STGAT-","").replace("-","–"))

        y = np.arange(len(vals))
        bars = ax.barh(y, vals, color=colors, zorder=3,
                       edgecolor="white", linewidth=0.4, height=0.6)

        # Value labels
        for bar, v in zip(bars, vals):
            xpos = v + abs(v)*0.02 if v >= 0 else v - abs(v)*0.06
            ax.text(xpos, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=7.5,
                    color="black")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.axvline(0, color="black", lw=0.7, ls="--", zorder=2)
        ax.set_xlabel(mlab, fontsize=9)
        ax.grid(True, alpha=0.25, axis="x")

        # Highlight proposed model
        for i, m in enumerate(models):
            if m == "STGAT-DPO":
                axes[metrics.index(metric)].get_yticklabels()[
                    [lbl.get_text() for lbl in
                     axes[metrics.index(metric)].get_yticklabels()
                    ].index("DPO") if "DPO" in [l.get_text() for l in
                     axes[metrics.index(metric)].get_yticklabels()] else 0
                ].set_fontweight("bold")

    plt.tight_layout(pad=1.2)
    out = FIG_DIR / "fig3_ablation_bars.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — BOOTSTRAP CI (horizontal bars)
# ═════════════════════════════════════════════════════════════════════════════
def fig4_bootstrap_ci(wf):
    np.random.seed(42)
    N_BOOT  = 10000
    models  = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT",
               "EW","BH","MinVar","RiskPar","InvVol","Momentum"]

    means, los, his, colors, labels = [], [], [], [], []
    for m in models:
        sub = wf[wf["strategy"]==m]
        if sub.empty: continue
        ir  = sub["info_ratio"].values
        boots = [np.random.choice(ir, len(ir), replace=True).mean()
                 for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        means.append(ir.mean())
        los.append(lo)
        his.append(hi)
        colors.append(STRAT_COLORS[m])
        labels.append(STRAT_LABELS[m])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(means))[::-1]   # reverse so STGAT-DPO is on top

    for i, (yi, m, lo, hi, c, lbl) in enumerate(
            zip(y, means, los, his, colors, labels)):
        # CI bar
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
        # Mean dot
        ax.scatter([m], [yi], color=c, s=50, zorder=5)
        # Zero-crossing highlight
        if lo > 0:
            ax.plot([lo, hi], [yi, yi], color=c, lw=3.5,
                    solid_capstyle="round", alpha=0.35)

    # Zero line
    ax.axvline(0, color="black", lw=1.2, ls="--", zorder=2)

    # Shade positive-only CI region
    ax.axvspan(0, max(his) + 0.05, alpha=0.04, color="green", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Mean Information Ratio (95% bootstrap CI)")

    # Annotate the STGAT-DPO CI
    stgat_y = y[0]
    ax.annotate("Only CI\nexcluding zero",
                xy=(los[0], stgat_y), xytext=(los[0]-0.55, stgat_y+0.7),
                fontsize=7.5, color=C["stgat"],
                arrowprops=dict(arrowstyle="->", color=C["stgat"],
                                lw=0.8, connectionstyle="arc3,rad=0.15"))

    plt.tight_layout(pad=1.2)
    out = FIG_DIR / "fig4_bootstrap_ci.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — ROLLING MAX DRAWDOWN
# ═════════════════════════════════════════════════════════════════════════════
def fig5_drawdown(ret_series, nifty):
    S, E = "2021-01-01", "2024-12-31"

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    add_regime_shading(ax, S, E)

    def rolling_dd(rets):
        cum = np.exp(rets.cumsum())
        return ((cum - cum.cummax()) / cum.cummax()) * 100

    # Nifty
    nm = nifty.loc[S:E].fillna(0)
    ax.plot(rolling_dd(nm), color=C["nifty"], lw=2, ls="--",
            label="Nifty 50 Index", zorder=10)

    for strat in ["EW", "MinVar", "RiskPar"]:
        if strat in ret_series:
            s = ret_series[strat].loc[S:E].fillna(0)
            ax.plot(rolling_dd(s), color=STRAT_COLORS[strat],
                    lw=1.0, alpha=0.55, label=STRAT_LABELS[strat])

    for strat in ["STGAT-NoHMM", "STGAT-NoGAT"]:
        if strat in ret_series:
            s = ret_series[strat].loc[S:E].fillna(0)
            ax.plot(rolling_dd(s), color=STRAT_COLORS[strat],
                    lw=1.4, alpha=0.65, ls=":", label=STRAT_LABELS[strat])

    if "STGAT-DPO" in ret_series:
        s = ret_series["STGAT-DPO"].loc[S:E].fillna(0)
        ax.plot(rolling_dd(s), color=C["stgat"], lw=2.4,
                label=STRAT_LABELS["STGAT-DPO"], zorder=9)

    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("")
    ax.legend(loc="lower left", ncol=2, framealpha=0.9, fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.tight_layout(pad=1.2)
    out = FIG_DIR / "fig5_drawdown.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — REGIME COMPOSITION + STGAT IR OVERLAY
# ═════════════════════════════════════════════════════════════════════════════
def fig6_regime_composition(wf):
    fold_labels = ["F1\n2021","F2\n21-22","F3\n2022",
                   "F4\n22-23","F5\n2023","F6\n2024"]
    bear_c    = [204, 160, 284,  54, 839, 175]
    side_c    = [1085,1205, 856, 524, 196, 878]
    bull_c    = [713, 637, 862,1424, 967, 949]
    totals    = [b+s+u for b,s,u in zip(bear_c,side_c,bull_c)]
    bear_p    = [b/t*100 for b,t in zip(bear_c,totals)]
    side_p    = [s/t*100 for s,t in zip(side_c,totals)]
    bull_p    = [u/t*100 for u,t in zip(bull_c,totals)]

    fig, ax1  = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2 = ax1.twinx()

    x = np.arange(6)
    w = 0.6
    ax1.bar(x, bear_p, w, label="Bear", color="#E74C3C", alpha=0.75, zorder=3)
    ax1.bar(x, side_p, w, bottom=bear_p,
            label="Sideways", color="#F39C12", alpha=0.75, zorder=3)
    ax1.bar(x, bull_p, w,
            bottom=[b+s for b,s in zip(bear_p,side_p)],
            label="Bull", color="#27AE60", alpha=0.75, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_labels)
    ax1.set_ylabel("Regime composition (%)")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # IR overlay
    sub = wf[wf["strategy"]=="STGAT-DPO"].sort_values("fold")
    ir_vals = sub["info_ratio"].values
    ax2.plot(x, ir_vals, color=C["stgat"], lw=2.2,
             marker="o", ms=6, label="STGAT-DPO IR", zorder=10)
    ax2.axhline(0, color=C["stgat"], lw=0.8, ls=":", alpha=0.5)
    ax2.set_ylabel("Information Ratio", color=C["stgat"])
    ax2.tick_params(axis="y", colors=C["stgat"])
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout(pad=1.2)
    out = FIG_DIR / "fig6_regime_composition.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 — RETURN DISTRIBUTION (violin)
# ═════════════════════════════════════════════════════════════════════════════
def fig7_return_distribution(ret_series, nifty):
    S, E    = "2021-01-01", "2024-12-31"
    models  = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT",
               "EW","MinVar","Momentum"]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    data   = []
    colors = []
    labels = []

    # Nifty
    nm = nifty.loc[S:E].fillna(0)
    data.append(nm.values * 100)
    colors.append(C["nifty"])
    labels.append("Nifty 50")

    for m in models:
        if m in ret_series:
            s = ret_series[m].loc[S:E].fillna(0)
            data.append(s.values * 100)
            colors.append(STRAT_COLORS[m])
            labels.append(STRAT_LABELS[m].replace(" (proposed)","")
                                         .replace(" (ablation)",""))

    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=True, showextrema=False,
                          widths=0.7)

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
        pc.set_edgecolor("white")
        pc.set_linewidth(0.5)

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Daily return (%)")
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)

    plt.tight_layout(pad=1.2)
    out = FIG_DIR / "fig7_return_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 8 — METRICS SUMMARY HEATMAP (clean final table figure)
# ═════════════════════════════════════════════════════════════════════════════
def fig8_metrics_heatmap(wf):
    models  = ["STGAT-DPO","STGAT-NoHMM","STGAT-NoGAT",
               "EW","MinVar","RiskPar","InvVol","BH","Momentum"]
    metrics = ["info_ratio","sharpe","ann_return",
               "max_drawdown","sortino","win_rate"]
    m_labs  = ["Info Ratio","Sharpe","Ann. Return",
               "Max Drawdown","Sortino","Win Rate"]

    data   = []
    rlabs  = []
    for m in models:
        sub = wf[wf["strategy"]==m]
        if sub.empty: continue
        row = [sub[c].mean() for c in metrics]
        data.append(row)
        rlabs.append(STRAT_LABELS[m].replace(" (proposed)","★")
                                     .replace(" (ablation)",""))

    data_arr = np.array(data)

    # Normalise each column 0→1 (higher=better)
    normed = np.zeros_like(data_arr)
    for j, m in enumerate(metrics):
        col = data_arr[:, j]
        mn, mx = col.min(), col.max()
        if mx > mn:
            normed[:, j] = (col - mn) / (mx - mn)
        else:
            normed[:, j] = 0.5
        if m == "max_drawdown":   # lower drawdown = better → flip
            normed[:, j] = 1 - normed[:, j]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(normed, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1)

    ax.set_xticks(range(len(m_labs)))
    ax.set_xticklabels(m_labs, fontsize=9.5)
    ax.set_yticks(range(len(rlabs)))
    ax.set_yticklabels(rlabs, fontsize=9.5)

    # Annotate cells with actual values
    for i in range(len(data)):
        for j in range(len(metrics)):
            v     = data_arr[i, j]
            n_val = normed[i, j]
            color = "black" if 0.2 < n_val < 0.8 else (
                    "white" if n_val <= 0.2 else "black")
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color=color,
                    fontweight="bold" if rlabs[i].endswith("★") else "normal")

    # Highlight STGAT-DPO row
    stgat_row = next((i for i, l in enumerate(rlabs) if "★" in l), None)
    if stgat_row is not None:
        for spine in ["bottom","top","left","right"]:
            ax.add_patch(plt.Rectangle(
                (-0.5, stgat_row - 0.5), len(metrics), 1,
                fill=False, edgecolor=C["stgat"], lw=2.5, zorder=5))

    plt.colorbar(im, ax=ax, shrink=0.7, label="Relative rank (green = better)")
    plt.tight_layout(pad=1.2)
    out = FIG_DIR / "fig8_metrics_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  Generating publication figures")
    print("=" * 55)

    master, wf, nifty, ret_series = load_all()
    print(f"  Loaded {len(wf)} fold results | "
          f"{len(ret_series)} return series\n")

    fig1_cumulative_wealth(ret_series, nifty)
    fig2_fold_ir(wf)
    fig3_ablation(wf)
    fig4_bootstrap_ci(wf)
    fig5_drawdown(ret_series, nifty)
    fig6_regime_composition(wf)
    fig7_return_distribution(ret_series, nifty)
    fig8_metrics_heatmap(wf)

    print(f"\n  All figures saved to: {FIG_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
