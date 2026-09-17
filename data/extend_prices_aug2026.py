"""
data/extend_prices_aug2026.py
================================
Extends the existing prices_51.parquet (currently ending 2026-03-30)
forward through 2026-08-30, for all 51 stocks already in the dataset.
Fetches with a few days of overlap before the current end date (safety
against any final-day adjustments), then appends new rows only.
"""
import sys, time, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PRICES_51 = RAW_DIR / "prices_51.parquet"
REPORT = RAW_DIR / "extend_report_aug2026.txt"

# Fetch with overlap from 10 days before current end, through new target end
FETCH_START = "2026-03-20"
FETCH_END = "2026-08-31"  # yfinance end is exclusive, so this captures through Aug 30

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]

def fetch_single(ticker, start, end, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df[PRICE_COLS].copy()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    x  {ticker}: {e}")
                return None

def build_multiindex_df(ticker, ohlcv_df):
    cols = pd.MultiIndex.from_tuples(
        [(ticker, ticker, col) for col in PRICE_COLS], names=["Ticker", "Ticker", "Price"]
    )
    return pd.DataFrame(ohlcv_df.values, index=ohlcv_df.index, columns=cols)

def main():
    print("="*70)
    print("  Extending prices_51.parquet through 2026-08-30")
    print("="*70)

    df_old = pd.read_parquet(PRICES_51)
    tickers = sorted(set(df_old.columns.get_level_values(0)))
    print(f"  Existing: {df_old.shape[0]:,} rows, {len(tickers)} tickers")
    print(f"  Existing date range: {df_old.index.min().date()} to {df_old.index.max().date()}")
    print(f"  Fetching new data: {FETCH_START} to {FETCH_END} (exclusive)")
    print()

    report_lines = [f"Extend report — generated {datetime.now()}", "="*70]
    new_frames = []
    n_ok, n_fail = 0, 0

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:>2}/{len(tickers)}]  {ticker:<16}", end="", flush=True)
        df_raw = fetch_single(ticker, FETCH_START, FETCH_END)
        if df_raw is None or df_raw.empty:
            print("  FAILED")
            report_lines.append(f"{ticker}: FAILED")
            n_fail += 1
            continue
        days = int(df_raw["Close"].notna().sum())
        last_date = df_raw.index.max().date()
        print(f"  OK  {days} days  last: {last_date}")
        report_lines.append(f"{ticker}: OK {days} days, last={last_date}")
        new_frames.append(build_multiindex_df(ticker, df_raw))
        n_ok += 1
        time.sleep(0.3)

    print(f"\n  Fetched OK: {n_ok}/{len(tickers)}  Failed: {n_fail}")
    if n_fail > 0:
        print("  WARNING: some tickers failed -- stopping before merge. Investigate before proceeding.")
        REPORT.write_text("\n".join(report_lines))
        sys.exit(1)

    df_new = pd.concat(new_frames, axis=1).sort_index()
    print(f"\n  New data shape: {df_new.shape}")
    print(f"  New date range: {df_new.index.min().date()} to {df_new.index.max().date()}")

    # Combine: keep all old rows, add new rows (dedupe on overlap by keeping OLD values
    # for any overlapping dates, since old data is the confirmed, already-used version)
    combined = pd.concat([df_old, df_new[~df_new.index.isin(df_old.index)]]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    print(f"\n  Combined shape: {combined.shape}")
    print(f"  Combined date range: {combined.index.min().date()} to {combined.index.max().date()}")
    added_rows = combined.shape[0] - df_old.shape[0]
    print(f"  Rows added: {added_rows}")

    out_path = RAW_DIR / "prices_51_aug2026.parquet"
    combined.to_parquet(out_path, compression="snappy")
    print(f"\n  Saved to NEW file (not overwriting original): {out_path}")
    print(f"  Review before replacing prices_51.parquet")

    report_lines += ["", f"Combined: {combined.shape[0]} rows, {combined.index.min().date()} to {combined.index.max().date()}",
                      f"Rows added: {added_rows}", f"Saved: {out_path}"]
    REPORT.write_text("\n".join(report_lines))
    print(f"  Report: {REPORT}")

if __name__ == "__main__":
    main()
