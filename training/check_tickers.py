import pandas as pd

df = pd.read_parquet("data/raw/prices_51.parquet")
tickers = sorted(set(df.columns.get_level_values(0)))
print(f"Total tickers: {len(tickers)}")
for t in tickers:
    print(t)
