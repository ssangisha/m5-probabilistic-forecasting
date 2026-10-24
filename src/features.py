import duckdb
import pandas as pd

con = duckdb.connect()

# Get daily sales and calendar columns for the 720 series in scope
query = """
SELECT id, cat_id, store_id, state_id, date, sales, sell_price,
       wday, month, event_name_1, event_type_1,
       snap_CA, snap_TX, snap_WI
FROM 'data/processed/sales_long.parquet'
WHERE id IN (SELECT id FROM 'data/scope/scope_skus.csv')
ORDER BY id, date
"""
df = con.execute(query).df()
df["date"] = pd.to_datetime(df["date"])

# SNAP flag for the store's own state
df["snap"] = 0
for state in ["CA", "TX", "WI"]:
    this_state = df["state_id"] == state
    df.loc[this_state, "snap"] = df.loc[this_state, f"snap_{state}"]

# Empty events become the word "none"
df["event_name_1"] = df["event_name_1"].fillna("none")
df["event_type_1"] = df["event_type_1"].fillna("none")

# Sales and zero-flag as they were 28 days ago
by_series = df.groupby("id")
df["sales_28ago"] = by_series["sales"].shift(28)
df["is_zero"] = (df["sales"] == 0).astype(float)
df["zero_28ago"] = df.groupby("id")["is_zero"].shift(28)

# Lag features (always at least 28 days back)
df["lag_28"] = df["sales_28ago"]
df["lag_35"] = df.groupby("id")["sales"].shift(35)
df["lag_42"] = df.groupby("id")["sales"].shift(42)

# Rolling features, measured on the data as it was 28 days ago
for window in [7, 28, 56]:
    df[f"mean_{window}"] = df.groupby("id")["sales_28ago"].transform(
        lambda s: s.rolling(window).mean())
df["std_28"] = df.groupby("id")["sales_28ago"].transform(
    lambda s: s.rolling(28).std())
df["zero_rate_28"] = df.groupby("id")["zero_28ago"].transform(
    lambda s: s.rolling(28).mean())

# Price compared with the price 28 days ago
df["price_28ago"] = df.groupby("id")["sell_price"].shift(28)
df["price_ratio"] = df["sell_price"] / df["price_28ago"]

# The first weeks of each series have no history yet, so drop them
df = df.dropna(subset=["mean_56", "lag_42", "price_ratio"])

keep = ["id", "date", "sales", "cat_id", "store_id", "state_id",
        "wday", "month", "event_name_1", "event_type_1", "snap",
        "sell_price", "price_ratio",
        "lag_28", "lag_35", "lag_42",
        "mean_7", "mean_28", "mean_56", "std_28", "zero_rate_28"]
df = df[keep]
df.to_parquet("data/processed/features.parquet", index=False)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Dates:", df["date"].min().date(), "to", df["date"].max().date())
print("Missing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])