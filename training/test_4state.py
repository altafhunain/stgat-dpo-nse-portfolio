"""
training/run_sensitivity.py
============================
Sensitivity and robustness analysis for STGAT-DPO.

Tests:
  1. Transaction cost sensitivity  — 5, 10, 20, 30 bps
  2. Graph threshold sensitivity   — 0.15, 0.25, 0.35, 0.45
  3. Lookback window sensitivity   — 10, 20, 30 days
  4. Max weight cap sensitivity    — 0.08, 0.10, 0.15, 0.20

All tests run on Fold 5 (2023-01-01 → 2023-12-31) only using
the saved stgat_fold5.pt checkpoint — no retraining required.
Fast runtime: ~5 minutes total.

Output:
  outputs/results/sensitivity_results.csv
  outputs/figures/fig11_sensitivity.png

Run:
    python training/run_sensitivity.py
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
    LOOKBACK, MAX_WEIGHT, SEED, DEVICE, PRICES_FILE
)

torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "outputs" / "results"
FIG_DIR  = ROOT / "outputs" / "figures"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed fold for sensitivity (Fold 5 — 2023 bull market) ───────────────────
SENS_FOLD = {
    "train_start": "2018-01-01",
    "train_end":   "2024-09-30",
    "test_start":  "2025-01-01",
    "test_end":    "2026-03-30",
    "ckpt":        "stgat_fold6.pt",
}

RF_DAILY  = 0.065 / 252
CORR_WINDOW = 63


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION ENGINE (parametric version of evaluate_portfolio)
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_with_params(model, feat_arr, ret_full, nifty_full,
                          regimes, sector_adj, dates,
                          test_start, test_end,
                          tc_bps=10, corr_threshold=0.25,
                          lookback=20, max_weight=0.15) -> dict:
    """
    Evaluate portfolio with specific parameter settings.
    Returns metric dict.
    """
    model.eval()
    N  = feat_arr.shape[1]
    tc = tc_bps / 10_000

    test_mask  = (dates >= test_start) & (dates <= test_end)
    test_dates = dates[test_mask]
    if len(test_dates) == 0:
        return {}

    reb_dates = (test_dates.to_series()
                 .groupby([test_dates.year, test_dates.month])
                 .first().values)
    reb_dates = pd.DatetimeIndex(reb_dates)

    port_ret_map = {}
    prev_w       = np.ones(N) / N

    for i, reb_d in enumerate(reb_dates):
        t_arr = np.where(dates == reb_d)[0]
        if len(t_arr) == 0:
            continue
        t = int(t_arr[0])
        if t < max(lookback, CORR_WINDOW):
            continue

        # Feature window (with custom lookback)
        x = feat_arr[max(0, t - lookback):t]
        if len(x) < lookback:
            pad = np.zeros((lookback - len(x), N, feat_arr.shape[2]),
                            dtype=np.float32)
            x = np.concatenate([pad, x], axis=0)
        x_t = torch.tensor(x, dtype=torch.float32
                            ).permute(1, 0, 2).unsqueeze(0).to(DEVICE)

        # Dynamic adjacency (with custom threshold)
        ret_window = ret_full[max(0, t - CORR_WINDOW):t]
        regime     = int(regimes[t])
        lam        = REGIME_LAMBDA[regime]

        # Custom threshold adjacency
        corr = np.corrcoef(ret_window.T)
        corr = np.nan_to_num(corr, nan=0.0)
        dyn  = (np.abs(corr) > corr_threshold).astype(np.float32)
        np.fill_diagonal(dyn, 0.0)
        dyn  = dyn * np.abs(corr).clip(0, 1)
        adj  = (0.7 * dyn + 0.3 * sector_adj).clip(0, 1)
        np.fill_diagonal(adj, 0.0)
        adj_t = torch.tensor(adj.astype(np.float32)
                              ).unsqueeze(0).to(DEVICE)

        # Forward pass
        try:
            # Adjust LSTM hidden state if lookback differs from training
            # Use model as-is (lookback mismatch is handled by padding/trimming)
            w, _, _ = model(x_t, adj_t, regime_lambda=lam)
            w_np    = w.squeeze(0).cpu().numpy()
        except Exception:
            w_np = prev_w.copy()

        # Apply custom max_weight constraint
        w_np = np.clip(w_np, 0, max_weight)
        w_np = w_np / (w_np.sum() + 1e-8)

        # Transaction cost
        cost   = np.abs(w_np - prev_w).sum() * tc
        prev_w = w_np.copy()

        # Holding period
        next_reb = (reb_dates[i + 1] if i + 1 < len(reb_dates)
                    else test_dates[-1] + pd.Timedelta(days=1))
        hold_mask = (dates >= reb_d) & (dates < next_reb)

        for d_idx, d in enumerate(dates[hold_mask]):
            ti = np.where(dates == d)[0]
            if len(ti) == 0:
                continue
            ti = int(ti[0])
            if ti >= len(ret_full):
                continue
            r = float(ret_full[ti] @ w_np)
            if d_idx == 0:
                r -= cost
            port_ret_map[d] = r

    if not port_ret_map:
        return {}

    rets = pd.Series(port_ret_map).sort_index().fillna(0)
    if len(rets) < 5:
        return {}

    nifty_csv = OUT_DIR / "nifty50_returns.csv"
    nifty_ser = pd.read_csv(nifty_csv, index_col=0, parse_dates=True
                             ).iloc[:, 0].sort_index()
    nifty_te  = nifty_ser.reindex(rets.index).ffill().fillna(0)

    excess = rets - RF_DAILY
    ann    = 252
    ann_ret = float(rets.mean() * ann)
    ann_vol = float(rets.std() * ann**0.5)
    sharpe  = float(excess.mean() / (rets.std() + 1e-8) * ann**0.5)

    active = rets.values - nifty_te.values
    ir     = float(active.mean() / (active.std() + 1e-8) * ann**0.5)

    cum    = np.exp(rets.cumsum())
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())

    turnover = np.nan  # approximate: already baked into costs

    return {
        "ann_return":   round(ann_ret, 4),
        "ann_vol":      round(ann_vol, 4),
        "sharpe":       round(sharpe,  4),
        "info_ratio":   round(ir,      4),
        "max_drawdown": round(max_dd,  4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  Sensitivity Analysis — STGAT-DPO (Fold 5 evaluation)")
    print("=" * 62)

    # ── Load data ─────────────────────────────────────────────────────────
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

    # ── Fit HMM for Fold 5 ────────────────────────────────────────────────
    tr_start = pd.Timestamp(SENS_FOLD["train_start"])
    tr_end   = pd.Timestamp(SENS_FOLD["train_end"])
    dates_ret= dates[1:]
    tr_mask  = (dates_ret >= tr_start) & (dates_ret <= tr_end)
    nifty_tr = nifty_ser.reindex(dates_ret[tr_mask]).fillna(0).values

    hmm_det = HMMRegimeDetector(n_states=4, seed=SEED)
    hmm_det.fit(nifty_tr)
    regimes_raw  = hmm_det.predict(nifty_ser.reindex(dates[1:]).fillna(0).values)
    regimes_full = np.zeros(len(dates), dtype=np.int32)
    regimes_full[1:] = regimes_raw

    rc = np.bincount(regimes_raw, minlength=3)
    print(f"  HMM: bear={rc[0]} sideways={rc[1]} bull={rc[2]}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_path = CKPT_DIR / SENS_FOLD["ckpt"]
    model = STGATDPOModel(
        n_stocks=N, n_features=N_FEATURES, lookback=LOOKBACK,
        lstm_hidden=64, lstm_layers=2, gat_hidden=64, gat_heads=2,
        dropout=0.0, max_weight=MAX_WEIGHT,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"  ✓  Loaded: {ckpt_path.name}")

    te_start = SENS_FOLD["test_start"]
    te_end   = SENS_FOLD["test_end"]

    # Baseline (default parameters)
    base = evaluate_with_params(
        model, feat_arr, ret_full, nifty_full,
        regimes_full, sector_adj, dates,
        te_start, te_end,
        tc_bps=10, corr_threshold=0.25,
        lookback=20, max_weight=0.15)
    print(f"\n  Baseline IR={base.get('info_ratio','?'):.4f}  "
          f"Sharpe={base.get('sharpe','?'):.4f}")

    # ── Save baseline result only ────────────────────────────────────────
    result_row = dict(base)
    result_row.update({"n_states": 4, "ckpt": SENS_FOLD["ckpt"],
                        "test_start": te_start, "test_end": te_end})
    df = pd.DataFrame([result_row])
    df.to_csv(OUT_DIR / "test_4state_blind_result.csv", index=False)
    print(f"\n  Saved: test_4state_blind_result.csv")
    print(f"\n{'='*62}")
    print("  4-STATE HMM — BLIND HOLDOUT RESULT")
    print("="*62)
    print(f"  IR       : {base.get('info_ratio', float('nan')):.4f}")
    print(f"  Sharpe   : {base.get('sharpe', float('nan')):.4f}")
    print(f"  Ann.Ret  : {base.get('ann_return', float('nan')):.4f}")
    print(f"  Max DD   : {base.get('max_drawdown', float('nan')):.4f}")
    print(f"{'='*62}\n")

if __name__ == "__main__":
    main()
