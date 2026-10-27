# compare_models.py
# Scores every model on the same rows and prints one table.

import pandas as pd
from scoring import score

MODEL_FILES = {
    "quantile baseline": "data/processed/baseline_quantiles.parquet",
    "LightGBM untuned": "data/processed/lgbm_quantiles.parquet",
    "LightGBM tuned": "data/processed/lgbm_tuned_quantiles.parquet",
    "N-HiTS": "data/processed/nhits_quantiles.parquet",
}

rows = []
reference_keys = None
for model_name, path in MODEL_FILES.items():
    forecasts = pd.read_parquet(path)

    # Every model must be scored on exactly the same rows
    keys = forecasts[["fold", "id", "date"]].sort_values(["fold", "id", "date"])
    keys = keys.reset_index(drop=True)
    if reference_keys is None:
        reference_keys = keys
    assert keys.equals(reference_keys), f"{model_name} was scored on different rows"

    for fold, group in forecasts.groupby("fold"):
        row = score(group)
        row["model"] = model_name
        row["fold"] = fold
        rows.append(row)

columns = ["fold", "model", "mean_pinball", "wape_median",
           "cov_80", "cov_90", "cov_95", "width_90"]
table = pd.DataFrame(rows)[columns].sort_values(["fold", "model"]).round(3)

print(table.to_string(index=False))
table.to_csv("reports/model_comparison.csv", index=False)