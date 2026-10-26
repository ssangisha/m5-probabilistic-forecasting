# tune_lightgbm.py
# Compares a small set of LightGBM settings using folds 1-3 only.
# Fold 4 is left alone so it stays an honest final test.

import numpy as np
import pandas as pd
import lightgbm as lgb
from splits import get_fold
from metrics import pinball_loss

# Same feature list as train_lgbm.py
FEATURES = ["cat_id", "store_id", "state_id", "wday", "month",
            "event_name_1", "event_type_1", "snap", "sell_price", "price_ratio",
            "lag_28", "lag_35", "lag_42",
            "mean_7", "mean_28", "mean_56", "std_28", "zero_rate_28"]
CATEGORY_COLUMNS = ["cat_id", "store_id", "state_id", "event_name_1", "event_type_1"]

TUNING_FOLDS = [1, 2, 3]
TUNING_QUANTILES = [0.05, 0.5, 0.95]   # three levels keep the search fast
assert 4 not in TUNING_FOLDS

SETTINGS = [
    {"name": "current",           "n_estimators": 200, "num_leaves": 31, "min_child_samples": 100},
    {"name": "smaller trees",     "n_estimators": 200, "num_leaves": 15, "min_child_samples": 100},
    {"name": "bigger trees",      "n_estimators": 200, "num_leaves": 63, "min_child_samples": 100},
    {"name": "more regularized",  "n_estimators": 200, "num_leaves": 31, "min_child_samples": 300},
    {"name": "more trees",        "n_estimators": 400, "num_leaves": 31, "min_child_samples": 100},
    {"name": "small, regularized, more trees",
     "n_estimators": 400, "num_leaves": 15, "min_child_samples": 300},
]

data = pd.read_parquet("data/processed/features.parquet")
for column in CATEGORY_COLUMNS:
    data[column] = data[column].astype("category")

results = []
for setting in SETTINGS:
    fold_losses = []
    for fold in TUNING_FOLDS:
        train, test = get_fold(data, fold)
        quantile_losses = []
        for quantile in TUNING_QUANTILES:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=setting["n_estimators"],
                learning_rate=0.05,
                num_leaves=setting["num_leaves"],
                min_child_samples=setting["min_child_samples"],
                random_state=42,
                verbose=-1,
            )
            model.fit(train[FEATURES], train["sales"])
            prediction = np.maximum(model.predict(test[FEATURES]), 0)
            quantile_losses.append(pinball_loss(test["sales"], prediction, quantile))
        fold_losses.append(np.mean(quantile_losses))

    average_loss = np.mean(fold_losses)
    results.append({"setting": setting["name"], "mean_pinball_folds_1_to_3": average_loss})
    print(f"{setting['name']}: {average_loss:.4f}")

results = pd.DataFrame(results).sort_values("mean_pinball_folds_1_to_3").round(4)
print()
print(results.to_string(index=False))
results.to_csv("reports/tuning_results.csv", index=False)