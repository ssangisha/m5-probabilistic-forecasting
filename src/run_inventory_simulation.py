# run_inventory_simulation.py
# Tests whether better forecasts lead to better stocking decisions.
# Results are simulation results under the stated assumptions below,
# not savings achieved in a real business.

import numpy as np
import pandas as pd
from scipy.stats import norm
from inventory_policy import simulate_base_stock
from splits import load_scope_sales, get_fold, HORIZON

# Assumptions (costs are in relative units, not money)
HOLDING_COST = 1.0                  # cost of one unit left on the shelf overnight
SHORTAGE_COSTS = [9, 19, 39]        # cost of one lost unit of demand
HISTORY_WEEKS = 8                   # weeks of history used by the safety-stock rule
ROUNDING = {"nearest": np.round, "up": np.ceil, "down": np.floor}

NAIVE = "seasonal-naive (no buffer)"
NORMAL_RULE = "mean + safety stock (normal rule)"

FORECAST_FILES = {
    NAIVE: "data/processed/baseline_forecasts.parquet",
    # The normal rule builds its own levels; this file only supplies the demand
    NORMAL_RULE: "data/processed/baseline_quantiles.parquet",
    "quantile baseline": "data/processed/baseline_quantiles.parquet",
    "LightGBM tuned": "data/processed/lgbm_tuned_quantiles.parquet",
    "N-HiTS": "data/processed/nhits_quantiles.parquet",
}
frames = {name: pd.read_parquet(path) for name, path in FORECAST_FILES.items()}
sales = load_scope_sales()


def normal_rule_levels(fold, tau):
    """Textbook safety stock: mean + z * standard deviation of the last 8 same weekdays."""
    train, _ = get_fold(sales, fold)
    train = train.sort_values(["id", "date"])
    recent = train.groupby("id").tail(HISTORY_WEEKS * 7)

    history = recent["sales"].to_numpy().reshape(-1, HISTORY_WEEKS, 7)
    mean = history.mean(axis=1)
    spread = history.std(axis=1, ddof=1)
    per_weekday = mean + norm.ppf(tau) * spread

    series_ids = recent["id"].to_numpy().reshape(-1, HISTORY_WEEKS * 7)[:, 0]
    return np.tile(per_weekday, HORIZON // 7), series_ids


def order_up_to_levels(model_name, fold_data, fold, tau):
    if model_name == NAIVE:
        return fold_data["forecast"].to_numpy().reshape(-1, HORIZON)
    if model_name == NORMAL_RULE:
        levels, series_ids = normal_rule_levels(fold, tau)
        demand_ids = fold_data["id"].to_numpy().reshape(-1, HORIZON)[:, 0]
        assert (series_ids == demand_ids).all(), "Series are in a different order"
        return levels
    return fold_data[f"q{round(tau, 3)}"].to_numpy().reshape(-1, HORIZON)


demand_by_fold = {}
rows = []
for rounding_name, rounding in ROUNDING.items():
    for shortage_cost in SHORTAGE_COSTS:
        # Best stocking quantile for these costs (the newsvendor critical fractile)
        tau = shortage_cost / (shortage_cost + HOLDING_COST)

        for model_name, frame in frames.items():
            for fold, fold_data in frame.groupby("fold"):
                fold_data = fold_data.sort_values(["id", "date"])
                demand = fold_data["sales"].to_numpy().reshape(-1, HORIZON)

                # Every policy must be simulated on exactly the same demand
                if fold not in demand_by_fold:
                    demand_by_fold[fold] = demand
                assert np.array_equal(demand_by_fold[fold], demand), "Demand differs"

                levels = order_up_to_levels(model_name, fold_data, fold, tau)
                levels = np.maximum(rounding(levels), 0)
                result = simulate_base_stock(demand, levels)

                inventory = result["inventory_units_nights"].sum()
                lost = result["units_lost"].sum()
                rows.append({
                    "rounding": rounding_name,
                    "shortage_cost": shortage_cost,
                    "target_quantile": round(tau, 3),
                    "model": model_name,
                    "fold": fold,
                    "series_days": demand.size,
                    "units_demanded": demand.sum(),
                    "units_sold": result["units_sold"].sum(),
                    "units_lost": lost,
                    "stockout_days": result["stockout_days"].sum(),
                    "inventory_units_nights": inventory,
                    "cost": HOLDING_COST * inventory + shortage_cost * lost,
                })

results = pd.DataFrame(rows)
results.to_csv("reports/inventory_simulation_by_fold.csv", index=False)

KEYS = ["rounding", "shortage_cost", "target_quantile", "model"]


def summarize(data):
    """Add up the folds in data and compute the headline measures."""
    totals = data.groupby(KEYS, sort=False).sum(numeric_only=True)
    summary = pd.DataFrame({
        "fill_rate": totals["units_sold"] / totals["units_demanded"],
        "stockout_day_rate": totals["stockout_days"] / totals["series_days"],
        "avg_inventory": totals["inventory_units_nights"] / totals["series_days"],
        "cost_per_series_day": totals["cost"] / totals["series_days"],
    })
    return summary.round(3)


main = results[results["rounding"] == "nearest"]
fold_4 = summarize(main[main["fold"] == 4]).droplevel("rounding")
pooled = summarize(main).droplevel("rounding")
rounding_check = summarize(results[results["shortage_cost"] == 19])["cost_per_series_day"]
rounding_check = rounding_check.unstack("rounding")

fold_4.to_csv("reports/inventory_simulation_fold4.csv")
pooled.to_csv("reports/inventory_simulation_pooled.csv")
rounding_check.to_csv("reports/inventory_simulation_rounding_check.csv")

print("Fold 4 (not used for tuning), rounding to nearest unit:")
print(fold_4.to_string())
print()
print("All four folds pooled, rounding to nearest unit:")
print(pooled.to_string())
print()
print("Cost per series-day for shortage cost 19, by rounding rule:")
print(rounding_check.to_string())
print()
print("Simulation results under stated assumptions; costs are in relative units.")