import pandas as pd

df = pd.read_parquet("data/raw/prices_51.parquet")
tickers = sorted(set(df.columns.get_level_values(0)))

print(f"{'Ticker':<16} {'First valid date':<18} {'Leading NaN days'}")
print("-" * 55)
results = []
for t in tickers:
    close = df[t][t]["Close"].sort_index()
    first_valid = close.dropna().index.min()
    leading_nan = close.loc[:close.index[close.index.get_loc(first_valid)-1]].isna().sum() if close.index.get_loc(first_valid) > 0 else 0
    results.append((t, first_valid, leading_nan))
    print(f"{t:<16} {str(first_valid.date()):<18} {leading_nan}")

late_listings = [r for r in results if r[2] > 0]
print(f"\nTotal tickers with any leading NaN (late listing): {len(late_listings)}")
print(f"Latest listing date among all 51: {max(r[1] for r in results)}")
