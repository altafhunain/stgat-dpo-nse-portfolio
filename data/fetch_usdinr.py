import yfinance as yf
import pandas as pd

FETCH_START = "2017-10-01"
FETCH_END   = "2026-08-11"

print("Fetching USD/INR (INR=X)...")
df = yf.download("INR=X", start=FETCH_START, end=FETCH_END, progress=False)
print(f"Rows fetched: {len(df)}")
print(df.head())
print(df.tail())

df.to_csv("data/raw/usdinr.csv")
print("Saved: data/raw/usdinr.csv")
