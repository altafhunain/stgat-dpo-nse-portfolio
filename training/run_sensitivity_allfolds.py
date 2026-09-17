"""
training/run_sensitivity_allfolds.py
======================================
Sensitivity and robustness analysis for STGAT-DPO, repeated across
ALL 9 walk-forward folds (not just Fold 5), addressing the reviewer
concern that single-fold sensitivity cannot establish global robustness.

Tests, per fold:
  1. Transaction cost sensitivity  — 0, 5, 10, 20, 30, 50 bps
  2. Graph threshold sensitivity   — 0.10-0.45
  3. Lookback window sensitivity   — 5-30 days
  4. Max weight cap sensitivity    — 0.05-0.25

Reuses evaluate_with_params() from run_sensitivity.py unchanged.
No retraining -- uses the 9 saved stgat_fold{1..9}.pt checkpoints.

Output:
  outputs/results/sensitivity_results_allfolds.csv   (fold-tagged, every run)
  outputs/results/sensitivity_crossfold_summary.csv  (mean/std across folds per param value)
  outputs/figures/fig11_sensitivity_allfolds.png      (mean line + std band, all 4 sweeps)

Run:
    python training/run_sensitivity_allfolds.py
"""

import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from utils.features import build_feature_tensor, N_FEATURES
from models.gnn.stgat_dpo import (
    STGATDPOModel, HMMRegimeDetector,
    build_sector_adj, build_dynamic_adj, REGIME_LAMBDA
)
from training.train_stgat import (
    LOOKBACK, MAX_WEIGHT, SEED, DEVICE, PRICES_FILE, FOLDS
)
from training.run_sensitivity import evaluate_with_params

torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "outputs" / "results"
FIG_DIR  = ROOT / "outputs" / "figures"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 62)
    print(f"  Sensitivity Analysis — STGAT-DPO (all {len(FOLDS)} walk-forward folds)")
    print("=" * 62)

    # ── Load shared data once ────────────────────────────────────────────
    feat_arr, tickers, dates = build_feature_tensor(PRICES_FILE, verbose=True)
    N = len(tickers)

    df_raw   = pd.read_parquet(PRICES_FILE)
    close_df = pd.DataFrame(
        {t: df_raw[(t, t, "Close")].squeeze() for t in tickers}
    ).reindex(dates).ffill(limit=5)
    ret_df   = np.log(close_df / close_df.shift(1)).iloc[1:]
    ret_full = np.zeros((len(dates), N), dtype=np.float32)
    ret_full[1:] = ret_df.fillna(0).values.astype(np.float32)

    nifty_ser = pd.read_csv(OUT_DIR / "nifty50_returns.csv",
                             index_col=0, parse_dates=True
                             ).iloc[:, 0].reindex(dates[1:]).ffill().fillna(0)
    nifty_full = np.zeros(len(dates), dtype=np.float32)
    nifty_full[1:] = nifty_ser.values

    sector_adj = build_sector_adj(tickers)
    dates_ret  = dates[1:]

    all_results = []

    # ── Loop over all 9 folds ────────────────────────────────────────────
    for fold_idx, (tr_start_s, tr_end_s, te_start_s, te_end_s) in enumerate(FOLDS, start=1):
        fold_name = f"Fold{fold_idx}"
        print(f"\n{'='*62}")
        print(f"  {fold_name}  train->{tr_end_s}  test {te_start_s}->{te_end_s}")
        print(f"{'='*62}")

        tr_start = pd.Timestamp(tr_start_s)
        tr_end   = pd.Timestamp(tr_end_s)
        tr_mask  = (dates_ret >= tr_start) & (dates_ret <= tr_end)
        nifty_tr = nifty_ser.reindex(dates_ret[tr_mask]).fillna(0).values

        hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
        hmm_det.fit(nifty_tr)
        regimes_raw  = hmm_det.predict_causal(nifty_ser.reindex(dates_ret).fillna(0).values)
        regimes_full = np.zeros(len(dates), dtype=np.int32)
        regimes_full[1:] = regimes_raw

        ckpt_path = CKPT_DIR / f"stgat_fold{fold_idx}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP: checkpoint not found: {ckpt_path}")
            continue

        model = STGATDPOModel(
            n_stocks=N, n_features=N_FEATURES, lookback=LOOKBACK,
            lstm_hidden=64, lstm_layers=2, gat_hidden=64, gat_heads=2,
            dropout=0.0, max_weight=MAX_WEIGHT,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        te_start = te_start_s
        te_end   = te_end_s

        base = evaluate_with_params(
            model, feat_arr, ret_full, nifty_full,
            regimes_full, sector_adj, dates,
            te_start, te_end,
            tc_bps=10, corr_threshold=0.25,
            lookback=20, max_weight=0.15)
        print(f"  Baseline IR={base.get('info_ratio', float('nan')):.4f}  "
              f"Sharpe={base.get('sharpe', float('nan')):.4f}")

        sweeps = [
            ("transaction_cost", [0, 5, 10, 20, 30, 50],
             lambda v: dict(tc_bps=v, corr_threshold=0.25, lookback=20, max_weight=0.15)),
            ("graph_threshold", [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45],
             lambda v: dict(tc_bps=10, corr_threshold=v, lookback=20, max_weight=0.15)),
            ("lookback", [5, 10, 15, 20, 25, 30],
             lambda v: dict(tc_bps=10, corr_threshold=0.25, lookback=v, max_weight=0.15)),
            ("max_weight", [0.05, 0.08, 0.10, 0.15, 0.20, 0.25],
             lambda v: dict(tc_bps=10, corr_threshold=0.25, lookback=20, max_weight=v)),
        ]

        for sens_type, values, kwargs_fn in sweeps:
            for v in values:
                kwargs = kwargs_fn(v)
                m = evaluate_with_params(
                    model, feat_arr, ret_full, nifty_full,
                    regimes_full, sector_adj, dates,
                    te_start, te_end, **kwargs)
                m.update({"sensitivity": sens_type, "param": v, "fold": fold_name})
                all_results.append(m)
        print(f"  {fold_name}: {sum(len(v) for _,v,_ in sweeps)} sensitivity runs complete")

    # ── Save full fold-tagged results ────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(OUT_DIR / "sensitivity_results_allfolds.csv", index=False)
    print(f"\n  Saved: sensitivity_results_allfolds.csv  ({len(df)} rows)")

    # ── Cross-fold aggregate: mean/std IR at each parameter value ───────
    agg = (df.groupby(["sensitivity", "param"])["info_ratio"]
             .agg(["mean", "std", "min", "max", "count"])
             .reset_index()
             .sort_values(["sensitivity", "param"]))
    agg.to_csv(OUT_DIR / "sensitivity_crossfold_summary.csv", index=False)
    print(f"  Saved: sensitivity_crossfold_summary.csv")

    # ── Figure: mean line + std band, across all 4 sweeps ───────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("white")

    configs = [
        ("transaction_cost", "Transaction cost (bps)", 10),
        ("graph_threshold", "Correlation threshold", 0.25),
        ("lookback", "Lookback window (days)", 20),
        ("max_weight", "Max weight cap", 0.15),
    ]

    for ax, (sens_type, xlabel, baseline_val) in zip(axes.flatten(), configs):
        ax.set_facecolor("white")
        sub = agg[agg["sensitivity"] == sens_type].sort_values("param")
        params = sub["param"].values
        mean_ir = sub["mean"].values
        std_ir  = sub["std"].values

        ax.plot(params, mean_ir, color="#C0392B", lw=2, marker="o", ms=5,
                label="Mean IR across 9 folds", zorder=5)
        ax.fill_between(params, mean_ir - std_ir, mean_ir + std_ir,
                         color="#C0392B", alpha=0.15, label="+/- 1 std across folds")
        ax.axvline(baseline_val, color="#888888", lw=1.2, ls=":", alpha=0.7,
                   label=f"Baseline ({baseline_val})")
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Mean Information Ratio", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.3, ls="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=1.5)
    out = FIG_DIR / "fig11_sensitivity_allfolds.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig11_sensitivity_allfolds.png")

    # ── Stability summary ────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  CROSS-FOLD STABILITY SUMMARY")
    print("="*62)
    for sens_type, xlabel, baseline_val in configs:
        sub = agg[agg["sensitivity"] == sens_type]
        base_row = sub[sub["param"] == baseline_val]
        base_mean = base_row["mean"].values[0] if len(base_row) else float("nan")
        print(f"  {xlabel:<28}  baseline mean IR={base_mean:.3f}  "
              f"mean-IR range across param values [{sub['mean'].min():.3f}, {sub['mean'].max():.3f}]  "
              f"typical cross-fold std={sub['std'].mean():.3f}")

    print(f"\n{'='*62}")
    print("  Multi-fold sensitivity analysis complete.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
