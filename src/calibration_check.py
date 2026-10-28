# calibration_check.py
# Checks whether each model's quantile forecasts match how often actual sales
# fall below them. Sales are whole numbers with many zeros, so for each quantile
# we report two rates:
#   share_below       = share of days with sales strictly below the forecast
#   share_at_or_below = share of days with sales at or below the forecast
# A calibrated forecast has share_below <= tau <= share_at_or_below.

import pandas as pd
import matplotlib.pyplot as plt
from scoring import TAUS

MODEL_FILES = {
    "quantile baseline": "data/processed/baseline_quantiles.parquet",
    "LightGBM tuned": "data/processed/lgbm_tuned_quantiles.parquet",
    "N-HiTS": "data/processed/nhits_quantiles.parquet",
}

# None means all series; 4 means only the fastest-selling quartile
SUBSETS = {"all series": None, "fast sellers (Q4)": 4}

series_quartiles = pd.read_csv("data/scope/scope_skus.csv")[["id", "vol_quartile"]]


def calibration_rows(forecasts, model_name, subset_name):
    rows = []
    actual = forecasts["sales"]
    for tau in TAUS:
        forecast = forecasts[f"q{tau}"]
        rows.append({
            "model": model_name,
            "subset": subset_name,
            "tau": tau,
            "share_below": (actual < forecast).mean(),
            "share_at_or_below": (actual <= forecast).mean(),
        })
    return rows


def judge(row, tolerance=0.01):
    """Small tolerance, because these are estimates from a finite sample."""
    if row["share_at_or_below"] < row["tau"] - tolerance:
        return "too low"
    if row["share_below"] > row["tau"] + tolerance:
        return "too high"
    return "ok"


rows = []
for model_name, path in MODEL_FILES.items():
    forecasts = pd.read_parquet(path).merge(series_quartiles, on="id")
    for subset_name, quartile in SUBSETS.items():
        if quartile is None:
            subset = forecasts
        else:
            subset = forecasts[forecasts["vol_quartile"] == quartile]
        rows += calibration_rows(subset, model_name, subset_name)

table = pd.DataFrame(rows)
table["verdict"] = table.apply(judge, axis=1)
table = table.round(3)
table.to_csv("reports/quantile_calibration.csv", index=False)

# Print the upper quantiles, which matter most for stocking decisions
upper = table[table["tau"] >= 0.9]
print(upper.to_string(index=False))

# One panel per model and subset. A bar that crosses the dashed line is calibrated.
figure, axes = plt.subplots(len(SUBSETS), len(MODEL_FILES), figsize=(13, 8),
                            sharex=True, sharey=True)
for row_number, subset_name in enumerate(SUBSETS):
    for column_number, model_name in enumerate(MODEL_FILES):
        axis = axes[row_number][column_number]
        part = table[(table["model"] == model_name) & (table["subset"] == subset_name)]
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.vlines(part["tau"], part["share_below"], part["share_at_or_below"], linewidth=5)
        axis.scatter(part["tau"], part["share_at_or_below"], s=12)
        axis.set_title(f"{model_name} - {subset_name}", fontsize=10)

for axis in axes[-1]:
    axis.set_xlabel("nominal quantile")
for axis in axes[:, 0]:
    axis.set_ylabel("observed share of actual sales")

plt.tight_layout()
plt.savefig("reports/figures/quantile_calibration.png", dpi=150)
print("Saved reports/figures/quantile_calibration.png")