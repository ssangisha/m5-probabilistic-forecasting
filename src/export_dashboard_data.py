# export_dashboard_data.py
# Copies the small forecast files that the dashboard needs into reports/forecasts,
# so the dashboard works without the large data/processed folder.

import os
import pandas as pd

QUANTILES = ["q0.025", "q0.05", "q0.1", "q0.5", "q0.9", "q0.95", "q0.975"]
FILE_NAMES = [
    "baseline_quantiles.parquet",
    "lgbm_tuned_quantiles.parquet",
    "nhits_quantiles.parquet",
]

os.makedirs("reports/forecasts", exist_ok=True)

for name in FILE_NAMES:
    forecasts = pd.read_parquet(f"data/processed/{name}")
    keep = forecasts[["fold", "id", "date", "sales"] + QUANTILES]
    target = f"reports/forecasts/{name}"
    keep.to_parquet(target, index=False)
    size_mb = os.path.getsize(target) / 1_000_000
    print(f"{name}: {len(keep):,} rows, {size_mb:.1f} MB")