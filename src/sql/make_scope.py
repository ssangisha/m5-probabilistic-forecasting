import duckdb
import pandas as pd

# We only look at history before this date, so the final year
# stays untouched for testing later.
CUTOFF = "2015-06-20"

con = duckdb.connect()

# Step 1: for every series, count its days of history and its average daily sales
query = f"""
SELECT id, item_id, cat_id, store_id,
       COUNT(*) AS n_days,
       AVG(sales) AS mean_sales
FROM 'data/processed/sales_long.parquet'
WHERE date < DATE '{CUTOFF}'
GROUP BY id, item_id, cat_id, store_id
HAVING COUNT(*) >= 730
"""
stats = con.execute(query).df()
print("Eligible series:", len(stats))

# Step 2: split series into 4 volume groups inside each category and store
def make_quartiles(sales):
    ranks = sales.rank(method="first")
    return pd.qcut(ranks, 4, labels=[1, 2, 3, 4])

stats["vol_quartile"] = stats.groupby(["cat_id", "store_id"])["mean_sales"].transform(make_quartiles)

# Step 3: pick 6 random series from each group (fixed seed = same result every run)
groups = stats.groupby(["cat_id", "store_id", "vol_quartile"])
scope = groups.sample(n=6, random_state=42)

# Step 4: save the list
scope = scope.sort_values(["cat_id", "store_id", "vol_quartile", "id"])
scope.to_csv("data/scope/scope_skus.csv", index=False)

# Step 5: quick check
print("Total series in scope:", len(scope))
summary = scope.groupby(["cat_id", "vol_quartile"], observed=True)["mean_sales"].agg(["count", "mean"])
print(summary.round(2))