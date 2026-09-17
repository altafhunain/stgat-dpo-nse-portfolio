"""
generate_figures.py

Generates figures for a Word-format research paper from the evaluation
results already collected. No titles on the figures themselves (caption
goes in Word below the image, per standard convention) -- clear axis
labels and legends only.

    Fig 1: Test IC by fold/year, per horizon (grouped bar chart)
    Fig 2: Average test IC by horizon (bar chart with error bars)
    Fig 3: Test IC vs. average India VIX per year (scatter + trend line)
    Fig 4: Ensemble vs. individual-fold IC on the true 2026 holdout
    Fig 5: Classifier-derived vs regression-derived Sharpe ratio

Saved as high-resolution PNG (primary, for Word embedding) and PDF
(vector, in case a LaTeX version is also needed).

Usage:
    python generate_figures.py \
        --fold-summary fold_evaluation_summary_tb_v2.csv \
        --ensemble-results ensemble_holdout_results.csv \
        --regime-analysis regime_analysis_by_year.csv \
        --output-dir ./figures
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_figures")

COLORS = {"1m": "#1f77b4", "2m": "#ff7f0e", "3m": "#2ca02c"}
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_fig(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)
    logger.info("Saved %s.png and %s.pdf", name, name)


def fig1_ic_by_fold(fold_summary, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    years = sorted(fold_summary["test_year"].unique())
    x = np.arange(len(years))
    width = 0.25

    for i, h in enumerate(["1m", "2m", "3m"]):
        vals = [fold_summary[(fold_summary["test_year"] == y) & (fold_summary["horizon"] == h)]["test_ic"].values[0]
                for y in years]
        ax.bar(x + (i - 1) * width, vals, width, label=h, color=COLORS[h])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Test Year")
    ax.set_ylabel("Information Coefficient (Spearman IC)")
    ax.legend(title="Horizon", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, output_dir, "fig1_ic_by_fold")


def fig2_avg_ic_by_horizon(fold_summary, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    horizons = ["1m", "2m", "3m"]
    means = [fold_summary[fold_summary["horizon"] == h]["test_ic"].mean() for h in horizons]
    stds = [fold_summary[fold_summary["horizon"] == h]["test_ic"].std() for h in horizons]

    bars = ax.bar(horizons, means, yerr=stds, capsize=6,
                   color=[COLORS[h] for h in horizons], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Mean Test IC (\u00b1 1 SD across 8 folds)")
    ax.grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)
    save_fig(fig, output_dir, "fig2_avg_ic_by_horizon")


def fig3_ic_vs_vix(regime_analysis, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    df = regime_analysis.dropna(subset=["test_ic", "avg_vix"])

    ax.scatter(df["avg_vix"], df["test_ic"], s=80, color="#1f77b4", zorder=3)
    for _, row in df.iterrows():
        ax.annotate(str(int(row["year"])), (row["avg_vix"], row["test_ic"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    z = np.polyfit(df["avg_vix"], df["test_ic"], 1)
    x_line = np.linspace(df["avg_vix"].min(), df["avg_vix"].max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.7, zorder=2)

    corr = df["avg_vix"].corr(df["test_ic"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Average India VIX")
    ax.set_ylabel(f"Average Test IC across Horizons (r = {corr:.2f})")
    ax.grid(alpha=0.3)
    save_fig(fig, output_dir, "fig3_ic_vs_vix")


def fig4_ensemble_vs_individual(ensemble_results, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = ["1m", "2m", "3m"]
    x = np.arange(len(horizons))
    width = 0.35

    individual_avg = [ensemble_results[(ensemble_results["horizon"] == h) &
                                        (ensemble_results["model"] != "ENSEMBLE_ALL_8")]["ic"].mean()
                       for h in horizons]
    ensemble_ic = [ensemble_results[(ensemble_results["horizon"] == h) &
                                     (ensemble_results["model"] == "ENSEMBLE_ALL_8")]["ic"].values[0]
                    for h in horizons]

    ax.bar(x - width / 2, individual_avg, width, label="Average of 8 individual folds", color="#7f7f7f")
    ax.bar(x + width / 2, ensemble_ic, width, label="8-fold ensemble", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Information Coefficient (2026 YTD Holdout)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, output_dir, "fig4_ensemble_vs_individual")


def fig5_classifier_vs_regression_sharpe(fold_summary, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = ["1m", "2m", "3m"]
    x = np.arange(len(horizons))
    width = 0.35

    cls_sharpe = [fold_summary[fold_summary["horizon"] == h]["naive_sharpe"].mean() for h in horizons]
    reg_sharpe = [fold_summary[fold_summary["horizon"] == h]["naive_sharpe_reg"].mean() for h in horizons]

    ax.bar(x - width / 2, cls_sharpe, width, label="Classifier-derived signal", color="#9467bd")
    ax.bar(x + width / 2, reg_sharpe, width, label="Regression-derived signal", color="#8c564b")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Mean Naive Annualized Sharpe (across 8 folds)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, output_dir, "fig5_classifier_vs_regression_sharpe")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-summary", required=True)
    parser.add_argument("--ensemble-results", default=None)
    parser.add_argument("--regime-analysis", default=None)
    parser.add_argument("--output-dir", default="./figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_summary = pd.read_csv(args.fold_summary)
    logger.info("Loaded fold summary: %d rows", len(fold_summary))

    fig1_ic_by_fold(fold_summary, output_dir)
    fig2_avg_ic_by_horizon(fold_summary, output_dir)

    if "naive_sharpe_reg" in fold_summary.columns:
        fig5_classifier_vs_regression_sharpe(fold_summary, output_dir)
    else:
        logger.warning("fold_summary missing regression-derived columns -- skipping Fig 5")

    if args.regime_analysis:
        regime_analysis = pd.read_csv(args.regime_analysis)
        fig3_ic_vs_vix(regime_analysis, output_dir)

    if args.ensemble_results:
        ensemble_results = pd.read_csv(args.ensemble_results)
        fig4_ensemble_vs_individual(ensemble_results, output_dir)

    logger.info("All figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
