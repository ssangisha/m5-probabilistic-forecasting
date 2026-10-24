import duckdb
import matplotlib.pyplot as plt

CUTOFF = "2015-06-20"
con = duckdb.connect()

# Daily sales for our 720 series, before the cutoff only
query = f"""
SELECT date, wday, cat_id, sales
FROM 'data/processed/sales_long.parquet'
WHERE date < DATE '{CUTOFF}'
  AND id IN (SELECT id FROM 'data/scope/scope_skus.csv')
"""
sales = con.execute(query).df()

# Add up all series in each category, one total per day
daily = sales.groupby(["date", "cat_id", "wday"], as_index=False)["sales"].sum()

# Weekly pattern: average sales on each weekday, divided by the overall average
# (1.0 = normal day, 1.2 = 20% above normal)
avg_by_wday = daily.groupby(["cat_id", "wday"])["sales"].mean().unstack(0)
overall_avg = daily.groupby("cat_id")["sales"].mean()
weekly_index = avg_by_wday / overall_avg
print("Weekly pattern (1 = Saturday, 2 = Sunday, ... 7 = Friday):")
print(weekly_index.round(2))

# Yearly pattern: 28-day rolling average of daily totals
totals = daily.pivot(index="date", columns="cat_id", values="sales")
smooth = totals.rolling(28).mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
weekly_index.plot(ax=axes[0], marker="o")
axes[0].set_title("Weekly pattern by category")
axes[0].set_xlabel("wday (1 = Saturday)")
axes[0].set_ylabel("Sales vs average day")

smooth.plot(ax=axes[1])
axes[1].set_title("Total daily sales (28-day average)")
axes[1].set_xlabel("")

plt.savefig("reports/figures/seasonality.png", dpi=150, bbox_inches="tight")
print("Saved reports/figures/seasonality.png")