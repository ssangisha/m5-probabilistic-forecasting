import numpy as np
import pandas as pd
from splits import load_scope_sales, get_fold, TRAIN_ENDS
from metrics import mae, wape, pinball_loss, coverage, interval_width

TAUS = [0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975]
N_WEEKS = 8   # how many past weeks we use for each weekday


def quantile_baseline(train, test):
    """For each series and weekday, take quantiles of the last 8 same weekdays."""
    last = train.groupby("id").tail(N_WEEKS * 7)
    history = last.groupby("id")["sales"].apply(list).to_dict()

    test = test.copy()
    test["day_number"] = test.groupby("id").cumcount()   # 0, 1, 2, ... 27

    all_quantiles = []
    for series_id, day in zip(test["id"], test["day_number"]):
        # every 7th value, starting at this weekday's position
        same_weekday_values = history[series_id][day % 7::7]
        all_quantiles.append(np.quantile(same_weekday_values, TAUS))

    columns = [f"q{tau}" for tau in TAUS]
    quantiles = pd.DataFrame(all_quantiles, columns=columns, index=test.index)
    return pd.concat([test, quantiles], axis=1)


df = load_scope_sales()

all_results = []
for fold in TRAIN_ENDS:
    train, test = get_fold(df, fold)
    result = quantile_baseline(train, test)
    result["fold"] = fold
    all_results.append(result)

    a = result["sales"]
    pinballs = [pinball_loss(a, result[f"q{tau}"], tau) for tau in TAUS]
    print(f"Fold {fold}: "
          f"mean pinball={np.mean(pinballs):.3f}  "
          f"WAPE of median={wape(a, result['q0.5']):.3f}  "
          f"coverage 80%={coverage(a, result['q0.1'], result['q0.9']):.3f}  "
          f"90%={coverage(a, result['q0.05'], result['q0.95']):.3f}  "
          f"95%={coverage(a, result['q0.025'], result['q0.975']):.3f}  "
          f"width 90%={interval_width(result['q0.05'], result['q0.95']):.2f}")

all_results = pd.concat(all_results)
all_results.to_parquet("data/processed/baseline_quantiles.parquet", index=False)