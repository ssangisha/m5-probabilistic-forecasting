import pandas as pd
from splits import load_scope_sales, get_fold, TRAIN_ENDS
from metrics import mae, wape, pinball_loss


def seasonal_naive(train, test):
    """Forecast = same weekday last week, repeated across the 28 days."""
    last7 = train.groupby("id").tail(7)
    last7_values = last7.groupby("id")["sales"].apply(list).to_dict()

    test = test.copy()
    test["day_number"] = test.groupby("id").cumcount()   # 0, 1, 2, ... 27
    test["forecast"] = [
        last7_values[series_id][day % 7]
        for series_id, day in zip(test["id"], test["day_number"])
    ]
    return test


df = load_scope_sales()
scope = pd.read_csv("data/scope/scope_skus.csv")[["id", "vol_quartile"]]

all_results = []
for fold in TRAIN_ENDS:
    train, test = get_fold(df, fold)
    result = seasonal_naive(train, test)
    result["fold"] = fold
    all_results.append(result)

    a = result["sales"]
    f = result["forecast"]
    print(f"Fold {fold}: rows={len(result)}  MAE={mae(a, f):.3f}  "
          f"WAPE={wape(a, f):.3f}  "
          f"pinball(0.5)={pinball_loss(a, f, 0.5):.3f}  MAE/2={mae(a, f) / 2:.3f}")

# Results by volume quartile, all folds together
all_results = pd.concat(all_results).merge(scope, on="id")
print()
for quartile, group in all_results.groupby("vol_quartile"):
    print(f"Quartile {quartile}: WAPE={wape(group['sales'], group['forecast']):.3f}")
all_results.to_parquet("data/processed/baseline_forecasts.parquet", index=False)