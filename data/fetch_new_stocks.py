"""
NSE Portfolio GNN — Data Fetch Script
======================================
Fetches 21 new NSE stocks via yfinance, merges with the existing
prices_30.parquet (30 stocks from the classification paper), and
writes prices_51.parquet.

Run on HPC login node or via job script:
    conda activate nse_research
    python data/fetch_new_stocks.py

Output:
    data/raw/prices_51.parquet   ← complete 51-stock OHLCV dataset
    data/raw/fetch_report.txt    ← per-ticker quality report
"""

import os
import sys
import time
import warnings
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent   # NSE_Portfolio_GNN/
RAW_DIR    = ROOT / "data" / "raw"
PRICES_30  = RAW_DIR / "prices_30.parquet"
PRICES_51  = RAW_DIR / "prices_51.parquet"
REPORT     = RAW_DIR / "fetch_report.txt"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Date range — matches existing prices_30.parquet exactly ──────────────────
FETCH_START = "2017-10-01"
FETCH_END   = "2026-03-31"

# ── 21 new tickers (Phase-2 expansion) ───────────────────────────────────────
NEW_STOCKS = {
    # FMCG
    "NESTLEIND.NS":  {"name": "Nestle India",           "sector": "FMCG",       "beta": "Low",    "type": "Private"},
    "BRITANNIA.NS":  {"name": "Britannia Industries",   "sector": "FMCG",       "beta": "Low",    "type": "Private"},
    "DABUR.NS":      {"name": "Dabur India",            "sector": "FMCG",       "beta": "Low",    "type": "Private"},
    "GODREJCP.NS":   {"name": "Godrej Consumer",        "sector": "FMCG",       "beta": "Low",    "type": "Private"},
    "TATACONSUM.NS": {"name": "Tata Consumer Products", "sector": "FMCG",       "beta": "Low",    "type": "Private"},
    "UBL.NS":        {"name": "United Breweries",       "sector": "FMCG",       "beta": "Medium", "type": "Private"},
    # Consumer / Chemicals
    "PIDILITIND.NS": {"name": "Pidilite Industries",    "sector": "Chemicals",  "beta": "Low",    "type": "Private"},
    "BERGEPAINT.NS": {"name": "Berger Paints",          "sector": "Consumer",   "beta": "Low",    "type": "Private"},
    "HAVELLS.NS":    {"name": "Havells India",          "sector": "Consumer",   "beta": "Medium", "type": "Private"},
    "VOLTAS.NS":     {"name": "Voltas",                 "sector": "Consumer",   "beta": "Medium", "type": "Private"},
    # Insurance
    "SBILIFE.NS":    {"name": "SBI Life Insurance",     "sector": "Insurance",  "beta": "Low",    "type": "Private"},
    "ICICIPRULI.NS": {"name": "ICICI Prudential Life",  "sector": "Insurance",  "beta": "Low",    "type": "Private"},
    # Diversified / Cement
    "GRASIM.NS":     {"name": "Grasim Industries",      "sector": "Diversified","beta": "Medium", "type": "Private"},
    "SHREECEM.NS":   {"name": "Shree Cement",           "sector": "Cement",     "beta": "Medium", "type": "Private"},
    "AMBUJACEM.NS":  {"name": "Ambuja Cements",         "sector": "Cement",     "beta": "Medium", "type": "Private"},
    # Energy / Power
    "TATAPOWER.NS":  {"name": "Tata Power",             "sector": "Power",      "beta": "High",   "type": "Private"},
    "BPCL.NS":       {"name": "BPCL",                   "sector": "Energy",     "beta": "High",   "type": "PSU"},
    "ONGC.NS":       {"name": "ONGC",                   "sector": "Energy",     "beta": "Medium", "type": "PSU"},
    # IT
    "TECHM.NS":      {"name": "Tech Mahindra",          "sector": "IT",         "beta": "Medium", "type": "Private"},
    "MPHASIS.NS":    {"name": "Mphasis",                "sector": "IT",         "beta": "Medium", "type": "Private"},
    # Banking
    "SBIN.NS":       {"name": "State Bank of India",    "sector": "Banking",    "beta": "Medium", "type": "PSU"},
}

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ─────────────────────────────────────────────────────────────────────────────
def banner(msg: str):
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


def fetch_single(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    """Download OHLCV for a single ticker with retry logic."""
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df.empty:
                return None
            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[PRICE_COLS].copy()
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)   # exponential backoff
            else:
                print(f"    ✗  {ticker}: {e}")
                return None


def build_multiindex_df(ticker: str, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrap a single-stock OHLCV df into the same 3-level MultiIndex format
    as the existing prices_30.parquet:  (Ticker, Ticker, Price)
    """
    cols = pd.MultiIndex.from_tuples(
        [(ticker, ticker, col) for col in PRICE_COLS],
        names=["Ticker", "Ticker", "Price"]
    )
    out = pd.DataFrame(ohlcv_df.values, index=ohlcv_df.index, columns=cols)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main():
    banner("STEP 1 — Load existing prices_30.parquet")

    if not PRICES_30.exists():
        print(f"  ✗  Not found: {PRICES_30}")
        print("     Copy your prices.parquet from the classification project:")
        print(f"     cp /home/pramod/hunain/TechnicalSentimentRegime_Second/"
              "nse_classifier/nse_classifier/outputs/data/prices.parquet")
        print(f"        {PRICES_30}")
        sys.exit(1)

    df30 = pd.read_parquet(PRICES_30)
    existing_tickers = sorted(set(df30.columns.get_level_values(0)))
    print(f"  ✓  Loaded: {df30.shape[0]:,} rows × {len(existing_tickers)} stocks")
    print(f"  ✓  Date range: {df30.index.min().date()} → {df30.index.max().date()}")
    print(f"  ✓  Index type: {type(df30.index.dtype)}")

    # ── STEP 2: Fetch 21 new tickers ─────────────────────────────────────────
    banner("STEP 2 — Fetch 21 new NSE stocks from Yahoo Finance")

    report_lines = [
        f"NSE Portfolio GNN — Stock Fetch Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Date range: {FETCH_START} → {FETCH_END}",
        "=" * 70,
        f"{'#':>3}  {'Ticker':<16}  {'Name':<28}  {'Days':>6}  {'Missing':>8}  {'First date':<12}  Status",
        "─" * 100,
    ]

    new_frames = []
    results = {}

    for i, (ticker, meta) in enumerate(NEW_STOCKS.items(), 1):
        print(f"  [{i:>2}/{len(NEW_STOCKS)}]  {ticker:<16}  {meta['name']}", end="", flush=True)

        df_raw = fetch_single(ticker, FETCH_START, FETCH_END)

        if df_raw is None or df_raw.empty:
            status  = "FAILED"
            days_ok = 0
            missing = "—"
            first   = "—"
            print(f"  ✗  FAILED")
        else:
            days_ok  = int(df_raw["Close"].notna().sum())
            missing  = int(df_raw["Close"].isna().sum())
            first    = df_raw["Close"].dropna().index.min().date()

            # Forward-fill up to 5 days for early IPO NaN gaps
            df_raw = df_raw.ffill(limit=5)
            # Any remaining NaNs (true early absence) → fill Volume=0, prices=ffill
            df_raw["Volume"] = df_raw["Volume"].fillna(0)
            df_raw[["Open","High","Low","Close"]] = (
                df_raw[["Open","High","Low","Close"]].ffill()
            )

            mi_df = build_multiindex_df(ticker, df_raw)
            new_frames.append(mi_df)

            gap_flag = "  ← GAPS" if missing > 50 else ""
            status = f"OK  {days_ok:,} days" + gap_flag
            print(f"  ✓  {days_ok:,} trading days  |  {missing} missing  |  first: {first}")

        results[ticker] = {"days": days_ok, "missing": missing, "first": str(first), "status": status}
        report_lines.append(
            f"{i:>3}. {ticker:<16}  {meta['name']:<28}  {str(days_ok):>6}  "
            f"{str(missing):>8}  {str(first):<12}  {status}"
        )
        time.sleep(0.3)   # polite rate limiting

    # ── STEP 3: Merge ─────────────────────────────────────────────────────────
    banner("STEP 3 — Merge into prices_51.parquet")

    if not new_frames:
        print("  ✗  No new data fetched. Check internet connectivity.")
        sys.exit(1)

    df_new = pd.concat(new_frames, axis=1)
    print(f"  New stocks shape : {df_new.shape}")

    # Align on common date index — union, then sort
    df_all = pd.concat([df30, df_new], axis=1).sort_index()

    # Ensure consistent date coverage — reindex to match original
    df_all = df_all.reindex(df30.index)   # keep exactly the original trading days

    # Final forward-fill pass (max 3 days) for any alignment gaps
    df_all = df_all.ffill(limit=3)

    total_stocks = len(set(df_all.columns.get_level_values(0)))
    print(f"  Combined shape   : {df_all.shape[0]:,} rows × {total_stocks} stocks")
    print(f"  Date range       : {df_all.index.min().date()} → {df_all.index.max().date()}")

    # ── STEP 4: Validate ──────────────────────────────────────────────────────
    banner("STEP 4 — Data quality validation")

    all_tickers = sorted(set(df_all.columns.get_level_values(0)))
    print(f"\n  {'Ticker':<18}  {'Close days':>10}  {'NaN %':>8}  {'Status'}")
    print("  " + "─" * 52)

    bad_stocks = []
    for t in all_tickers:
        # Use xs to safely extract a single stock-price column
        try:
            close = df_all.xs((t, "Close"), level=(0, 2), axis=1).iloc[:, 0]
        except Exception:
            close = df_all[(t, t, "Close")].iloc[:, 0] if hasattr(df_all[(t, t, "Close")], "iloc") else df_all[(t, t, "Close")]
        close = close.squeeze()
        n_ok    = int(close.notna().sum().item() if hasattr(close.notna().sum(), "item") else close.notna().sum())
        pct_nan = float(close.isna().mean().item() if hasattr(close.isna().mean(), "item") else close.isna().mean()) * 100
        if pct_nan > 5:
            status = f"⚠  {pct_nan:.1f}% missing"
            bad_stocks.append(t)
        else:
            status = "✓"
        print(f"  {t:<18}  {n_ok:>10,}  {pct_nan:>7.2f}%  {status}")

    print(f"\n  Total stocks   : {len(all_tickers)}")
    print(f"  Stocks w/ >5%  : {len(bad_stocks)}")
    if bad_stocks:
        print(f"  Flagged        : {', '.join(bad_stocks)}")
        print(f"  Action         : These stocks will use forward-filled prices.")
        print(f"                   Verify listing dates — early NaN is expected.")

    # ── STEP 5: Save ──────────────────────────────────────────────────────────
    banner("STEP 5 — Save prices_51.parquet")

    df_all.to_parquet(PRICES_51, compression="snappy")
    size_mb = PRICES_51.stat().st_size / 1e6
    print(f"  ✓  Saved: {PRICES_51}")
    print(f"  ✓  File size : {size_mb:.2f} MB")
    print(f"  ✓  Shape     : {df_all.shape[0]:,} rows × {total_stocks} stocks × 5 OHLCV fields")

    # ── Write report ──────────────────────────────────────────────────────────
    report_lines += [
        "─" * 100,
        f"",
        f"Final dataset",
        f"  Stocks     : {total_stocks}",
        f"  Rows       : {df_all.shape[0]:,}",
        f"  Date range : {df_all.index.min().date()} → {df_all.index.max().date()}",
        f"  File       : {PRICES_51}",
        f"  Size       : {size_mb:.2f} MB",
    ]
    REPORT.write_text("\n".join(report_lines))
    print(f"  ✓  Report  : {REPORT}")

    banner("DONE — prices_51.parquet ready")
    print(f"  Next step: python training/run_benchmarks.py")
    print()


if __name__ == "__main__":
    main()
