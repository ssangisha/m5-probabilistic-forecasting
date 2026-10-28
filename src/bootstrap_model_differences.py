# bootstrap_model_differences.py
# Asks: are the differences between models bigger than the luck of which
# series ended up in the sample? We resample the 720 series many times.

import numpy as np
import pandas as pd

TAUS = [0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975]
N_RESAMPLES = 2000

files = {
    "quantile baseline": "data/processed/baseline_quantiles.parquet",
    "LightGBM tuned": "data/processed/lgbm_tuned_quantiles.parquet",
    "N-HiTS": "data/processed/nhits_quantiles.parquet",
}


def loss_per_series(forecasts):
    """Mean pinball loss of each series (average over the 7 quantiles and all days)."""
    actual = forecasts["sales"].to_numpy()
    row_loss = np.zeros(len(forecasts))
    for tau in TAUS:
        error = actual - forecasts[f"q{tau}"].to_numpy()
        row_loss += np.where(error >= 0, tau * error, (tau - 1) * error)
    row_loss = row_loss / len(TAUS)
    return pd.Series(row_loss, index=forecasts["id"].to_numpy()).groupby(level=0).mean().sort_index()


rng = np.random.default_rng(42)


def compare(loss_a, loss_b, label):
    """Bootstrap the difference in average loss (A minus B) over series."""
    n = len(loss_a)
    differences = []
    for _ in range(N_RESAMPLES):
        pick = rng.integers(0, n, n)
        differences.append(loss_a[pick].mean() - loss_b[pick].mean())
    low, high = np.percentile(differences, [2.5, 97.5])
    observed = loss_a.mean() - loss_b.mean()
    print(f"{label}: difference {observed:+.4f}, 95% interval {low:+.4f} to {high:+.4f}")


all_forecasts = {name: pd.read_parquet(path) for name, path in files.items()}

for title, fold_filter in [("Fold 4 only (not used for tuning)", [4]),
                           ("All four folds", [1, 2, 3, 4])]:
    print(title)
    losses = {}
    for name, forecasts in all_forecasts.items():
        part = forecasts[forecasts["fold"].isin(fold_filter)]
        losses[name] = loss_per_series(part).to_numpy()
    compare(losses["quantile baseline"], losses["LightGBM tuned"], "  baseline minus LightGBM")
    compare(losses["quantile baseline"], losses["N-HiTS"], "  baseline minus N-HiTS")
    compare(losses["LightGBM tuned"], losses["N-HiTS"], "  LightGBM minus N-HiTS")
    print()
print("If an interval does not include 0, the difference is unlikely to be sampling luck.")
print("It does not cover randomness from model training (seeds).")