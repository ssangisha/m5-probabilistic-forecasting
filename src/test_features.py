import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")

# Take one series and one date well inside its history
one = df[df["id"] == df["id"].iloc[0]].set_index("date")
day = one.index[300]

# lag_28 must equal the sales exactly 28 days earlier
sales_28_days_before = one.loc[day - pd.Timedelta(days=28), "sales"]
assert one.loc[day, "lag_28"] == sales_28_days_before

# lag_42 must equal the sales exactly 42 days earlier
assert one.loc[day, "lag_42"] == one.loc[day - pd.Timedelta(days=42), "sales"]

# mean_7 must equal the average of the 7 days ending 28 days earlier
window = one.loc[day - pd.Timedelta(days=34): day - pd.Timedelta(days=28), "sales"]
assert abs(one.loc[day, "mean_7"] - window.mean()) < 1e-9

# zero_rate_28 must equal the share of zero-sales days in the 28 days ending 28 days earlier
zero_window = one.loc[day - pd.Timedelta(days=55): day - pd.Timedelta(days=28), "sales"]
assert abs(one.loc[day, "zero_rate_28"] - (zero_window == 0).mean()) < 1e-9

# Every test row for fold 1 must have all its features
test_rows = df[(df["date"] >= "2015-06-20") & (df["date"] <= "2015-07-17")]
assert len(test_rows) == 720 * 28

print("All feature tests passed")