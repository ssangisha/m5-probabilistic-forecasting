import numpy as np
import pandas as pd
import lightgbm as lgb
from splits import get_fold, TRAIN_ENDS
from scoring import TAUS

FEATURES = ["cat_id", "store_id", "state_id", "wday", "month",
            "event_name_1", "event_type_1", "snap", "sell_price", "price_ratio",
            "lag_28", "lag_35", "lag_42",
            "mean_7", "mean_28", "mean_56", "std_28", "zero_rate_28"]
CATEGORY_COLUMNS = ["cat_id", "store_id", "state_id", "event_name_1", "event_type_1"]

df = pd.read_parquet("data/processed/features.parquet")
for column in CATEGORY_COLUMNS:
    df[column] = df[column].astype("category")

all_results = []
for fold in TRAIN_ENDS:
    train, test = get_fold(df, fold)

    result = test[["id", "date", "sales"]].copy()
    result["fold"] = fold

    for tau in TAUS:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=tau,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=100,
            random_state=42,
            verbose=-1,
        )
        model.fit(train[FEATURES], train["sales"])
        result[f"q{tau}"] = model.predict(test[FEATURES])
        print(f"fold {fold}, tau {tau} done")

    # Sales cannot be negative, and a higher quantile must not be below a lower one
    columns = [f"q{tau}" for tau in TAUS]
    values = np.maximum(result[columns].values, 0)
    result[columns] = np.sort(values, axis=1)

    all_results.append(result)

all_results = pd.concat(all_results)
all_results.to_parquet("data/processed/lgbm_tuned_quantiles.parquet", index=False)
print(all_results.groupby("fold").size())