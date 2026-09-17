"""
training/build_gate_features.py
=================================
Builds market-state features for the STGAT-DPO / BL-ML gating network.
IMPORTANT: build_feature_tensor()'s output is cross-sectionally z-scored
per day (see utils/features.py), which destroys the absolute meaning of
breadth/volume-surge/dispersion signals. Those three are therefore
recomputed here directly from raw prices/volume, not pulled from the
normalized model-input tensor. Nifty-level and relative-performance
features are computed from separate, unnormalized return series.
Output: outputs/results/gate_features.csv (indexed by date)
"""
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from pathlib import Path

from utils.features import build_feature_tensor, compute_stock_features
from models.gnn.stgat_dpo import HMMRegimeDetector

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "results"
PRICES_FILE = ROOT / "data" / "raw" / "prices_51.parquet"

print("=" * 70)
print("  Building gate features for STGAT-DPO / BL-ML gating network")
print("=" * 70)

# ── Get the canonical date range / tickers from the normalized tensor ───
# (used only for tickers/dates alignment, NOT for breadth/vol/dispersion)
feat_arr, tickers, dates = build_feature_tensor(PRICES_FILE, verbose=True)
T, N, F = feat_arr.shape

# ── Recompute RAW (non-normalized) per-stock features directly ──────────
print("\n  Recomputing raw (non-z-scored) ema_dev_20 / vol_ratio / ret_1d "
      "for breadth, volume-surge, and dispersion signals...")
df_raw = pd.read_parquet(PRICES_FILE)
ema_dev_raw = np.full((T, N), np.nan, dtype=np.float64)
vol_ratio_raw = np.full((T, N), np.nan, dtype=np.float64)
ret1d_raw = np.full((T, N), np.nan, dtype=np.float64)

for j, tkr in enumerate(tickers):
    close = df_raw[(tkr, tkr, "Close")].squeeze()
    volume = df_raw[(tkr, tkr, "Volume")].squeeze()
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    if isinstance(volume, pd.DataFrame): volume = volume.iloc[:, 0]
    close = close.ffill(limit=5)
    volume = volume.fillna(0)
    fdf = compute_stock_features(close, volume).reindex(dates)
    ema_dev_raw[:, j] = fdf["ema_dev_20"].values
    vol_ratio_raw[:, j] = fdf["vol_ratio"].values
    ret1d_raw[:, j] = fdf["ret_1d"].values

# Forward-fill any residual NaN (late listings) so aggregates are well-defined
ema_dev_raw = pd.DataFrame(ema_dev_raw, index=dates).ffill().bfill().values
vol_ratio_raw = pd.DataFrame(vol_ratio_raw, index=dates).ffill().bfill().values
ret1d_raw = pd.DataFrame(ret1d_raw, index=dates).fillna(0).values

breadth = (ema_dev_raw > 0).mean(axis=1)                # [T] fraction above own 20d EMA
vol_surge = np.median(vol_ratio_raw, axis=1)              # [T] median volume/20d-avg (raw, >=0)
cs_disp_daily = ret1d_raw.std(axis=1)                      # [T] daily cross-sectional return std (raw)
cs_disp = pd.Series(cs_disp_daily, index=dates).rolling(21, min_periods=5).mean().values

print(f"  breadth range: [{breadth.min():.3f}, {breadth.max():.3f}]  (sanity: should span well beyond a tight band around 0.5)")
print(f"  vol_surge range: [{vol_surge.min():.3f}, {vol_surge.max():.3f}]  (sanity: should be >= 0)")
print(f"  cs_dispersion range: [{np.nanmin(cs_disp):.5f}, {np.nanmax(cs_disp):.5f}]  (sanity: should vary meaningfully, not be ~constant)")

# ── Nifty-level vol/return (unaffected by the normalization issue) ──────
nifty_csv = OUT_DIR / "nifty50_returns.csv"
nifty_ser = pd.read_csv(nifty_csv, index_col=0, parse_dates=True).iloc[:, 0]
nifty_ser = nifty_ser.reindex(dates).ffill().fillna(0)
nifty_vol_21d = nifty_ser.rolling(21, min_periods=5).std().values
nifty_vol_63d = nifty_ser.rolling(63, min_periods=10).std().values
nifty_ret_21d = nifty_ser.rolling(21, min_periods=5).sum().values
nifty_ret_63d = nifty_ser.rolling(63, min_periods=10).sum().values

# ── HMM regime (one-hot) ─────────────────────────────────────────────────
dates_ret = dates[1:]
hmm_det = HMMRegimeDetector(n_states=3, seed=42)
hmm_det.fit(nifty_ser.reindex(dates_ret).values)
regime_pred = hmm_det.predict(nifty_ser.reindex(dates_ret).values)
regime_full = np.zeros(T, dtype=np.int32)
regime_full[1:] = regime_pred
regime_onehot = np.eye(3)[regime_full]

# ── Trailing relative performance: STGAT-DPO vs BL-ML ───────────────────
stgat_ret = pd.read_csv(OUT_DIR / "stgat_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
blml_ret = pd.read_csv(OUT_DIR / "bl_ml_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
stgat_ret = stgat_ret[~stgat_ret.index.duplicated(keep="first")].sort_index()
blml_ret = blml_ret[~blml_ret.index.duplicated(keep="first")].sort_index()
stgat_ret = stgat_ret.reindex(dates).fillna(0)
blml_ret = blml_ret.reindex(dates).fillna(0)
rel_daily = stgat_ret - blml_ret
rel_perf_21d = rel_daily.rolling(21, min_periods=5).sum().values
rel_perf_63d = rel_daily.rolling(63, min_periods=10).sum().values

# ── Assemble ──────────────────────────────────────────────────────────────
gate_df = pd.DataFrame({
    "nifty_vol_21d":  nifty_vol_21d,
    "nifty_vol_63d":  nifty_vol_63d,
    "nifty_ret_21d":  nifty_ret_21d,
    "nifty_ret_63d":  nifty_ret_63d,
    "regime_bear":    regime_onehot[:, 0],
    "regime_sideways":regime_onehot[:, 1],
    "regime_bull":    regime_onehot[:, 2],
    "rel_perf_21d":   rel_perf_21d,
    "rel_perf_63d":   rel_perf_63d,
    "breadth":        breadth,
    "cs_dispersion":  cs_disp,
    "vol_surge":      vol_surge,
}, index=dates)

gate_df = gate_df.fillna(0)
out_path = OUT_DIR / "gate_features.csv"
gate_df.to_csv(out_path)

print(f"\n  Gate feature matrix: {gate_df.shape}")
print(f"  Saved: {out_path}")
print("\n  Sample (last 5 rows):")
print(gate_df.tail())
print("\nDone.")
