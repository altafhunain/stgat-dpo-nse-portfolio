"""
utils/features.py
==================
Compute technical analysis feature matrix from raw OHLCV data.
Produces a 3D tensor: [N_days, N_stocks, N_features]

Features (8 per stock):
  0  ret_1d      — daily log return
  1  ret_5d      — 5-day log return
  2  ret_21d     — 21-day log return
  3  vol_21d     — 21-day rolling volatility
  4  rsi_14      — 14-day RSI
  5  ema_dev_20  — % deviation from 20-day EMA
  6  vol_ratio   — volume / 20-day avg volume
  7  mom_63d     — 63-day price momentum
"""

import numpy as np
import pandas as pd
from pathlib import Path


FEATURE_NAMES = [
    "ret_1d", "ret_5d", "ret_21d",
    "vol_21d", "rsi_14", "ema_dev_20",
    "vol_ratio", "mom_63d",
    "macd_signal", "bb_width", "obv_roc",
]
N_FEATURES = len(FEATURE_NAMES)
WARMUP     = 200     # trading days needed before first valid feature row


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(returns: pd.Series, period: int = 14) -> pd.Series:
    gain = returns.clip(lower=0)
    loss = (-returns).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs  = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_stock_features(close: pd.Series,
                            volume: pd.Series) -> pd.DataFrame:
    """
    Compute all 8 features for a single stock.
    Returns DataFrame indexed by date, columns = FEATURE_NAMES.
    """
    log_ret = np.log(close / close.shift(1))

    # ── Returns ───────────────────────────────────────────────────────────
    ret_1d  = log_ret
    ret_5d  = np.log(close / close.shift(5))
    ret_21d = np.log(close / close.shift(21))

    # ── Volatility ────────────────────────────────────────────────────────
    vol_21d = log_ret.rolling(21).std()

    # ── RSI ───────────────────────────────────────────────────────────────
    rsi_14  = _rsi(log_ret, 14)

    # ── EMA deviation ─────────────────────────────────────────────────────
    ema20      = _ema(close, 20)
    ema_dev_20 = (close - ema20) / (ema20 + 1e-8)

    # ── Volume ratio ──────────────────────────────────────────────────────
    vol_20ma  = volume.rolling(20).mean()
    vol_ratio = volume / (vol_20ma + 1e-8)
    vol_ratio = vol_ratio.clip(0, 10)        # cap extreme volume spikes

    # ── Medium-term momentum ──────────────────────────────────────────────
    mom_63d = np.log(close / close.shift(63))

    # MACD signal
    ema12 = _ema(close, 12); ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9) / (close.abs() + 1e-8)
    # Bollinger Band width
    ma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
    bb_width = (2 * std20) / (ma20 + 1e-8)
    # OBV rate of change
    obv = (np.sign(log_ret) * volume).cumsum()
    obv_roc = obv.pct_change(21).clip(-5, 5)

    df = pd.DataFrame({
        "ret_1d":      ret_1d,
        "ret_5d":      ret_5d,
        "ret_21d":     ret_21d,
        "vol_21d":     vol_21d,
        "rsi_14":      rsi_14,
        "ema_dev_20":  ema_dev_20,
        "vol_ratio":   vol_ratio,
        "mom_63d":     mom_63d,
        "macd_signal": macd_signal,
        "bb_width":    bb_width,
        "obv_roc":     obv_roc,
    }, index=close.index)

    return df


def build_feature_tensor(prices_path: str | Path,
                          start_date: str = "2017-10-03",
                          end_date:   str = "2026-08-28",
                          verbose: bool = True
                          ) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
    """
    Load prices_51.parquet and compute full feature tensor.

    Returns
    -------
    feat_arr  : np.ndarray  shape [T, N, F]  (NaN-free after WARMUP)
    tickers   : list of ticker strings
    dates     : DatetimeIndex aligned with feat_arr rows
    """
    import pandas as pd

    df = pd.read_parquet(prices_path)
    tickers = sorted(set(df.columns.get_level_values(0)))
    N = len(tickers)

    if verbose:
        print(f"  Computing features for {N} stocks...")

    stock_feat = {}
    for t in tickers:
        close  = df[(t, t, "Close")].squeeze()
        volume = df[(t, t, "Volume")].squeeze()

        # Ensure Series (not DataFrame)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]

        close  = close.ffill(limit=5)
        volume = volume.fillna(0)

        feat_df = compute_stock_features(close, volume)
        stock_feat[t] = feat_df

    # Align all stocks to common date index
    all_dates = stock_feat[tickers[0]].index
    for t in tickers[1:]:
        all_dates = all_dates.intersection(stock_feat[t].index)

    # Filter to requested range
    all_dates = all_dates[
        (all_dates >= start_date) & (all_dates <= end_date)
    ]
    all_dates = all_dates.sort_values()

    # Build tensor [T, N, F]
    T = len(all_dates)
    feat_arr = np.full((T, N, N_FEATURES), np.nan, dtype=np.float32)

    for j, t in enumerate(tickers):
        fdf = stock_feat[t].reindex(all_dates)
        feat_arr[:, j, :] = fdf[FEATURE_NAMES].values

    if verbose:
        nan_pct = np.isnan(feat_arr).mean() * 100
        print(f"  Raw tensor shape : {feat_arr.shape}  "
              f"({nan_pct:.1f}% NaN before warmup trim)")

    # Trim warmup period
    # Allow up to 10% of stocks to have NaN (handles late-IPO stocks)
    # This lets us use 2010 data even though 3 insurance stocks only
    # list in 2017-18. Their NaN values are forward-filled after trim.
    nan_per_day = np.isnan(feat_arr).any(axis=2).sum(axis=1)  # [T]
    valid_mask  = nan_per_day <= int(feat_arr.shape[1] * 0.10)
    first_valid = int(np.argmax(valid_mask))
    feat_arr    = feat_arr[first_valid:]
    all_dates   = all_dates[first_valid:]
    # Forward-fill any remaining NaN (late-IPO stocks)
    for j in range(feat_arr.shape[1]):
        for f in range(feat_arr.shape[2]):
            col = feat_arr[:, j, f]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                idx = np.where(~nan_mask)[0]
                if len(idx) > 0:
                    feat_arr[:idx[0], j, f] = col[idx[0]]  # backfill start
                    for k in range(len(col)):
                        if np.isnan(feat_arr[k, j, f]) and k > 0:
                            feat_arr[k, j, f] = feat_arr[k-1, j, f]

    if verbose:
        print(f"  Clean tensor     : {feat_arr.shape}  "
              f"({all_dates[0].date()} → {all_dates[-1].date()})")

    # Cross-sectional z-score normalisation per day per feature
    # This removes market-wide effects and focuses on relative ranking
    feat_arr = _cross_sectional_normalise(feat_arr)

    return feat_arr, tickers, all_dates


def _cross_sectional_normalise(arr: np.ndarray,
                                clip: float = 3.0) -> np.ndarray:
    """
    Z-score each feature across stocks at every timestep.
    arr : [T, N, F]
    """
    # Compute mean and std across N dimension (axis=1)
    mu  = arr.mean(axis=1, keepdims=True)        # [T, 1, F]
    std = arr.std(axis=1, keepdims=True) + 1e-8  # [T, 1, F]
    normed = (arr - mu) / std
    # Clip outliers
    normed = np.clip(normed, -clip, clip)
    return normed.astype(np.float32)


def get_close_returns(prices_path: str | Path,
                       dates: pd.DatetimeIndex,
                       tickers: list[str]) -> np.ndarray:
    """
    Extract aligned daily log returns matrix for the given dates/tickers.
    Returns np.ndarray [T, N] — to be used as realised returns for loss.
    """
    df = pd.read_parquet(prices_path)
    close_dict = {}
    for t in tickers:
        c = df[(t, t, "Close")].squeeze()
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        close_dict[t] = c.ffill(limit=5)

    close_df = pd.DataFrame(close_dict).reindex(dates).ffill(limit=5)
    log_ret  = np.log(close_df / close_df.shift(1)).iloc[1:].values

    return log_ret.astype(np.float32)
