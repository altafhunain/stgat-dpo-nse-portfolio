import pandas as pd

df = pd.read_parquet("data/raw/prices_51.parquet")
hdfclife = df["HDFCLIFE.NS"]["HDFCLIFE.NS"]["Close"]
hdfclife = hdfclife.sort_index()

print("First 10 non-null HDFCLIFE.NS Close prices:")
print(hdfclife.dropna().head(10))

print("\nFirst 15 rows of raw series (showing any pre-listing values):")
print(hdfclife.head(15))

print(f"\nTotal NaN count before 2017-11-17: {hdfclife.loc[:'2017-11-16'].isna().sum()}")
print(f"Total non-NaN count before 2017-11-17: {hdfclife.loc[:'2017-11-16'].notna().sum()}")

first_valid = hdfclife.dropna().index.min()
print(f"\nFirst valid (non-NaN) date for HDFCLIFE.NS: {first_valid}")
