# score_by_quartile.py
# Scores each model separately for each volume quartile of series.

import pandas as pd
from scoring import score

MODEL_FILES = {
    "quantile baseline": "data/processed/baseline_quantiles.parquet",
    "LightGBM untuned": "data/processed/lgbm_quantiles.parquet",
    "LightGBM tuned": "data/processed/lgbm_tuned_quantiles.parquet",
}

series_quartiles = pd.read_csv("data/scope/scope_skus.csv")[["id", "vol_quartile"]]

rows = []
for model_name, path in MODEL_FILES.items():
    forecasts = pd.read_parquet(path).merge(series_quartiles, on="id")
    for quartile, group in forecasts.groupby("vol_quartile"):
        row = score(group)
        row["model"] = model_name
        row["vol_quartile"] = quartile
        rows.append(row)

columns = ["vol_quartile", "model", "mean_pinball", "wape_median",
           "cov_80", "cov_90", "cov_95", "width_90"]
scorecard = pd.DataFrame(rows)[columns]
scorecard = scorecard.sort_values(["vol_quartile", "model"]).round(3)

print(scorecard.to_string(index=False))
scorecard.to_csv("reports/scorecard_by_quartile.csv", index=False)