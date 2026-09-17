"""
training/rebuild_sector_weights.py
=====================================
Rebuilds the sector-allocation heatmap (fig12_sector_weights.png) from
the confirmed final weights_Fold*.csv files (regenerated 2026-08-18
under the corrected, reproducible pipeline), using the model's own
authoritative SECTOR_MAP from models/gnn/stgat_dpo.py.
"""
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from models.gnn.stgat_dpo import SECTOR_MAP

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "outputs" / "figures"

# Group the 6 smallest sectors into "Other" to match the existing 9-row figure format
MAIN_SECTORS = ["IT", "Banking", "FMCG", "Auto", "Pharma", "Energy", "Infrastructure", "Consumer"]
ticker_to_group = {}
for sector, members in SECTOR_MAP.items():
    group = sector if sector in MAIN_SECTORS else "Other"
    for t in members:
        ticker_to_group[t] = group

fold_labels = ["F1\n2021", "F2\n21-22", "F3\n2022", "F4\n22-23", "F5\n2023", "F6\n2024"]
groups = MAIN_SECTORS + ["Other"]
heatmap = np.zeros((len(groups), 6))

print("Rebuilding sector-weight heatmap from confirmed final weights_Fold*.csv files")
print("="*70)
for fold_i in range(1, 7):
    df = pd.read_csv(RESULTS_DIR / f"weights_Fold{fold_i}.csv", index_col=0)
    mean_weights = df.mean(axis=0)  # average weight per stock across all rebalance dates in this fold
    for g_i, group in enumerate(groups):
        tickers_in_group = [t for t, grp in ticker_to_group.items() if grp == group and t in mean_weights.index]
        heatmap[g_i, fold_i-1] = mean_weights[tickers_in_group].sum()
    print(f"  Fold{fold_i}: total weight check = {mean_weights.sum():.4f} (should be ~1.0)")

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0)
ax.set_xticks(range(6)); ax.set_xticklabels(fold_labels)
ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups)
for i in range(len(groups)):
    for j in range(6):
        ax.text(j, i, f"{heatmap[i,j]:.2f}", ha="center", va="center", fontsize=8)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Portfolio weight")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig12_sector_weights.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {FIG_DIR / 'fig12_sector_weights.png'}")

print("\nSector weight table (mean across dates within each fold):")
sw_df = pd.DataFrame(heatmap, index=groups, columns=[f"Fold{i}" for i in range(1,7)])
print(sw_df.round(4))
sw_df.to_csv(RESULTS_DIR / "sector_weights_rebuilt.csv")
print(f"\nSaved: {RESULTS_DIR / 'sector_weights_rebuilt.csv'}")
print("Done.")
