"""
training/prototype_quantile_fold5.py
=====================================
Single-fold prototype test of the quantile-head / skew-adjusted-mu
architecture change. Trains ONLY Fold 5 from scratch under the new
architecture, saving to a separate checkpoint so nothing in the
existing (old-architecture) pipeline is touched or overwritten.
Compares against the confirmed baseline: Ann.Return=0.2615,
Sharpe=2.1396, IR=1.6596, MaxDD=-0.0663.
Run:
    python training/prototype_quantile_fold5.py
"""
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from utils.features import build_feature_tensor
from models.gnn.stgat_dpo import HMMRegimeDetector, build_sector_adj
from training.train_stgat import (
    STGATDataset, collate_fn, STGATTrainer, evaluate_portfolio, compute_metrics,
    FOLDS, LOOKBACK, BATCH_SIZE, SEED, DEVICE, PRICES_FILE, OUT_DIR, N_FEATURES
)

print("="*65)
print("  PROTOTYPE: Quantile-head / skew-adjusted-mu — Fold 5 only")
print("="*65)

feat_arr, tickers, dates = build_feature_tensor(PRICES_FILE, verbose=True)
N = len(tickers)
T = len(dates)

df_raw = pd.read_parquet(PRICES_FILE)
close_df = pd.DataFrame(
    {t: df_raw[(t, t, "Close")].squeeze() for t in tickers}
).reindex(dates).ffill(limit=5)
ret_df = np.log(close_df / close_df.shift(1)).iloc[1:]
ret_arr = ret_df.fillna(0).values.astype(np.float32)

# Point-in-time eligibility mask (same as production fix)
first_valid_idx = close_df.notna().values.argmax(axis=0)
never_valid = ~close_df.notna().values.any(axis=0)
eligible_from = first_valid_idx + 63
eligible_from[never_valid] = 10**9
idx_grid = np.arange(T)[:, None]
eligible_mask_full = idx_grid >= eligible_from[None, :]

nifty_csv = OUT_DIR / "nifty50_returns.csv"
nifty_ser = pd.read_csv(nifty_csv, index_col=0, parse_dates=True).iloc[:, 0]
nifty_ser = nifty_ser.reindex(dates[1:]).ffill().fillna(0)
nifty_arr = nifty_ser.values.astype(np.float32)

sector_adj = build_sector_adj(tickers)

# Fold 5 = index 4
tr_start, tr_end, te_start, te_end = FOLDS[4]
print(f"\nFold5  train->{tr_end}  test {te_start}->{te_end}")

dates_ret = dates[1:]
tr_ret_mask = (dates_ret >= tr_start) & (dates_ret <= tr_end)
nifty_tr = nifty_arr[tr_ret_mask]

hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
hmm_det.fit(nifty_tr)
regimes_full = hmm_det.predict(nifty_arr)
regimes_full_aligned = np.zeros(T, dtype=np.int32)
regimes_full_aligned[1:] = regimes_full
regime_counts = np.bincount(regimes_full, minlength=3)
print(f"Regime distribution: bear={regime_counts[0]} sideways={regime_counts[1]} bull={regime_counts[2]}")

ret_for_feat = np.zeros((T, N), dtype=np.float32)
ret_for_feat[1:] = ret_arr
nifty_for_feat = np.zeros(T, dtype=np.float32)
nifty_for_feat[1:] = nifty_arr

tr_start_ts = pd.Timestamp(tr_start)
tr_end_ts = pd.Timestamp(tr_end)
tr_sample_dates_mask = np.array([
    (dates[t] >= tr_start_ts) and (dates[t] <= tr_end_ts) for t in range(T)
])

train_dataset = STGATDataset(
    feat_arr=feat_arr, ret_arr=ret_for_feat, nifty_rets=nifty_for_feat,
    regimes=regimes_full_aligned, sector_adj=sector_adj, tickers=tickers,
    lookback=LOOKBACK, corr_window=63, eligible_mask=eligible_mask_full,
)
train_indices = [i for i, t in enumerate(train_dataset.valid_idx) if tr_sample_dates_mask[t]]
val_split = int(0.85 * len(train_indices))
val_indices = train_indices[val_split:]
train_indices = train_indices[:val_split]

from torch.utils.data import Subset
train_sub = Subset(train_dataset, train_indices)
val_sub = Subset(train_dataset, val_indices)
train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

trainer = STGATTrainer(n_stocks=N, n_features=N_FEATURES)
trainer.train(train_loader, val_loader, "Fold5_prototype")

ckpt_path = Path("outputs/checkpoints/stgat_fold5_QUANTILE_PROTOTYPE_v2.pt")
trainer.save(ckpt_path)
print(f"Saved prototype checkpoint: {ckpt_path}")

print("\n-- Evaluating Fold5 --")
port_rets, weights_df = evaluate_portfolio(
    model=trainer.model, feat_arr=feat_arr, ret_arr=ret_for_feat,
    nifty_rets=nifty_for_feat, regimes=regimes_full_aligned,
    sector_adj=sector_adj, dates=dates, test_start=te_start, test_end=te_end,
)
nifty_ser_eval = pd.Series(nifty_for_feat, index=dates)
metrics = compute_metrics(port_rets, nifty_ser_eval)

print("\n" + "="*65)
print("  PROTOTYPE RESULT vs CONFIRMED BASELINE (mean-variance only)")
print("="*65)
print(f"  {'Metric':<15} {'Baseline':>12} {'Prototype':>12}")
print(f"  {'Ann.Return':<15} {0.2615:>12.4f} {metrics.get('ann_return', float('nan')):>12.4f}")
print(f"  {'Sharpe':<15} {2.1396:>12.4f} {metrics.get('sharpe', float('nan')):>12.4f}")
print(f"  {'IR':<15} {1.6596:>12.4f} {metrics.get('info_ratio', float('nan')):>12.4f}")
print(f"  {'MaxDD':<15} {-0.0663:>12.4f} {metrics.get('max_drawdown', float('nan')):>12.4f}")
print(f"\n  Learned skew_kappa: {trainer.model.skew_kappa.item():.4f}")
print("\nDone.")
