# check_data_integrity.py
# Double-checks the cleaned data, the scope list and every forecast file.

import duckdb
import numpy as np
import pandas as pd
from splits import TRAIN_ENDS, HORIZON

con = duckdb.connect()

# 1. Cleaning must not have removed any real sales
raw_total = con.execute(r"""
    SELECT SUM(sales) FROM (
        UNPIVOT read_csv_auto('data/raw/sales_train_evaluation.csv')
        ON COLUMNS('^d_\d+$') INTO NAME d VALUE sales)
""").fetchone()[0]
clean_total = con.execute(
    "SELECT SUM(sales) FROM 'data/processed/sales_long.parquet'").fetchone()[0]
print(f"Total units sold, raw file: {raw_total:,}   cleaned file: {clean_total:,}")
assert raw_total == clean_total, "Cleaning removed real sales"

# 2. One row per series per day
duplicates = con.execute("""
    SELECT COUNT(*) FROM (
        SELECT id, date FROM 'data/processed/sales_long.parquet'
        GROUP BY id, date HAVING COUNT(*) > 1)
""").fetchone()[0]
print("Duplicate series-days:", duplicates)
assert duplicates == 0

# 3. The frozen scope list
scope = pd.read_csv("data/scope/scope_skus.csv")
assert len(scope) == 720 and scope["id"].is_unique
assert (scope.groupby(["cat_id", "vol_quartile"]).size() == 60).all()
history = con.execute("""
    SELECT id, COUNT(*) AS days FROM 'data/processed/sales_long.parquet'
    WHERE date < DATE '2015-06-20' AND id IN (SELECT id FROM 'data/scope/scope_skus.csv')
    GROUP BY id
""").df()
assert len(history) == 720 and history["days"].min() >= 730
print("Scope list ok: 720 unique series, at least 730 days of history each")

# 4. Forecast files: no gaps, no negatives, quantiles in order, correct dates
quantile_columns = ["q0.025", "q0.05", "q0.1", "q0.5", "q0.9", "q0.95", "q0.975"]
files = {
    "quantile baseline": "data/processed/baseline_quantiles.parquet",
    "LightGBM untuned": "data/processed/lgbm_quantiles.parquet",
    "LightGBM tuned": "data/processed/lgbm_tuned_quantiles.parquet",
    "N-HiTS": "data/processed/nhits_quantiles.parquet",
}
for name, path in files.items():
    forecasts = pd.read_parquet(path)
    values = forecasts[quantile_columns].to_numpy()
    assert not np.isnan(values).any(), f"{name}: missing values"
    assert (values >= 0).all(), f"{name}: negative forecasts"
    assert (np.diff(values, axis=1) >= 0).all(), f"{name}: quantiles out of order"
    assert (forecasts.groupby("fold").size() == 720 * HORIZON).all(), f"{name}: wrong row count"
    assert not forecasts.duplicated(["fold", "id", "date"]).any(), f"{name}: duplicates"
    for fold, group in forecasts.groupby("fold"):
        first_day = pd.Timestamp(TRAIN_ENDS[fold]) + pd.Timedelta(days=1)
        last_day = pd.Timestamp(TRAIN_ENDS[fold]) + pd.Timedelta(days=HORIZON)
        assert group["date"].min() == first_day and group["date"].max() == last_day, \
            f"{name}: fold {fold} has the wrong dates"
    print(f"{name}: ok")

print("All integrity checks passed")