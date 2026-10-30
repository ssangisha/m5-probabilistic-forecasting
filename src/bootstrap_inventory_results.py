# bootstrap_inventory_results.py
# Asks: is the simulated cost difference between policies bigger than the luck
# of which series ended up in the sample? We resample the 720 series.
# Simulation results under stated assumptions; costs are in relative units.

import numpy as np
import pandas as pd

HOLDING_COST = 1.0
SCENARIOS = {9: 0.9, 19: 0.95, 39: 0.975}      # shortage cost: target quantile
N_RESAMPLES = 2000
NORMAL_RULE = "mean + safety stock (normal rule)"
MODEL_NAMES = ["quantile baseline", "LightGBM tuned", "N-HiTS"]

results = pd.read_parquet("data/processed/inventory_series_level_results.parquet")
rng = np.random.default_rng(42)


def cost_per_series(data, model_name, tau, shortage_cost):
    """Total simulated cost of one policy for each series (folds added up)."""
    rows = data[(data["model"] == model_name) & (data["tau"] == tau)]
    per_series = rows.groupby("id").sum(numeric_only=True)
    cost = HOLDING_COST * per_series["inventory_units_nights"] \
        + shortage_cost * per_series["units_lost"]
    return cost.sort_index().to_numpy()


def percent_change(cost_a, cost_b):
    """Bootstrap the percent change in average cost of policy A against policy B."""
    n = len(cost_a)
    changes = []
    for _ in range(N_RESAMPLES):
        pick = rng.integers(0, n, n)
        changes.append((cost_a[pick].mean() / cost_b[pick].mean() - 1) * 100)
    low, high = np.percentile(changes, [2.5, 97.5])
    observed = (cost_a.mean() / cost_b.mean() - 1) * 100
    return f"{observed:+.1f}% (95% interval {low:+.1f}% to {high:+.1f}%)"


for title, folds in [("Fold 4 only (not used for tuning)", [4]),
                     ("All four folds", [1, 2, 3, 4])]:
    data = results[results["fold"].isin(folds)]
    print(title)
    for shortage_cost, tau in SCENARIOS.items():
        costs = {name: cost_per_series(data, name, tau, shortage_cost)
                 for name in [NORMAL_RULE] + MODEL_NAMES}
        print(f"  Shortage cost {shortage_cost} (target quantile {tau})")
        for name in MODEL_NAMES:
            print(f"    {name} vs normal rule: {percent_change(costs[name], costs[NORMAL_RULE])}")
        print(f"    N-HiTS vs LightGBM tuned: {percent_change(costs['N-HiTS'], costs['LightGBM tuned'])}")
    print()
print("Resampling covers which series were sampled, not randomness in model training.")