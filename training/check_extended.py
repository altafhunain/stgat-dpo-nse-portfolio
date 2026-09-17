import pandas as pd

df51 = pd.read_parquet("data/raw/prices_51.parquet")
df51x = pd.read_parquet("data/raw/prices_51_extended.parquet")
t51 = sorted(set(df51.columns.get_level_values(0)))
t51x = sorted(set(df51x.columns.get_level_values(0)))
print(f"prices_51: {len(t51)} tickers")
print(f"prices_51_extended: {len(t51x)} tickers")
print(f"In extended but not in 51: {set(t51x) - set(t51)}")
print(f"In 51 but not in extended: {set(t51) - set(t51x)}")
