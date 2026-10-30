# inventory_frontier.py
# Efficiency frontier for the stocking policies: for each policy and stocking
# level, how many units sit on the shelf on average, and what share of demand
# is met? This comparison does not depend on any cost assumption.
# Simulation results under stated assumptions (see run_inventory_simulation.py).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from inventory_policy import simulate_base_stock
from splits import load_scope_sales, get_fold, HORIZON

HISTORY_WEEKS = 8
FOLDS = [1, 2, 3, 4]
MODEL_TAUS = [0.5, 0.9, 0.95, 0.975]                     # levels in the forecast files
NORMAL_TAUS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]    # the rule can use any level
COMPARE_TAUS = [0.9, 0.95, 0.975]
HOLDING_COST = 1.0
SHORTAGE_COST = 19          # base case: target quantile 0.95

NAIVE = "seasonal-naive (no buffer)"
NORMAL_RULE = "mean + safety stock (normal rule)"
MODEL_NAMES = ["quantile baseline", "LightGBM tuned", "N-HiTS"]

frames = {
    "quantile baseline": pd.read_parquet("data/processed/baseline_quantiles.parquet"),
    "LightGBM tuned": pd.read_parquet("data/processed/lgbm_tuned_quantiles.parquet"),
    "N-HiTS": pd.read_parquet("data/processed/nhits_quantiles.parquet"),
}
naive_frame = pd.read_parquet("data/processed/baseline_forecasts.parquet")
sales = load_scope_sales()
quartiles = pd.read_csv("data/scope/scope_skus.csv")[["id", "vol_quartile"]]


def fold_part(frame, fold):
    return frame[frame["fold"] == fold].sort_values(["id", "date"])


def run_policy(model_name, tau, fold, demand, levels, series_ids):
    """Simulate one policy and return one row per series."""
    levels = np.maximum(np.round(levels), 0)
    result = simulate_base_stock(demand, levels)
    return pd.DataFrame({
        "model": model_name,
        "tau": tau,
        "fold": fold,
        "id": series_ids,
        "units_demanded": demand.sum(axis=1),
        "units_sold": result["units_sold"],
        "units_lost": result["units_lost"],
        "inventory_units_nights": result["inventory_units_nights"],
        "series_days": demand.shape[1],
    })


def weekday_mean_and_spread(fold, series_ids):
    """Mean and standard deviation of the last 8 same weekdays, for each series."""
    train, _ = get_fold(sales, fold)
    train = train.sort_values(["id", "date"])
    recent = train.groupby("id").tail(HISTORY_WEEKS * 7)
    recent_ids = recent["id"].to_numpy().reshape(-1, HISTORY_WEEKS * 7)[:, 0]
    assert (recent_ids == series_ids).all(), "Series are in a different order"
    history = recent["sales"].to_numpy().reshape(-1, HISTORY_WEEKS, 7)
    return history.mean(axis=1), history.std(axis=1, ddof=1)


pieces = []
for fold in FOLDS:
    reference = fold_part(frames["quantile baseline"], fold)
    demand = reference["sales"].to_numpy().reshape(-1, HORIZON)
    series_ids = reference["id"].to_numpy().reshape(-1, HORIZON)[:, 0]

    # Point forecast with no buffer
    naive = fold_part(naive_frame, fold)
    assert np.array_equal(naive["sales"].to_numpy().reshape(-1, HORIZON), demand)
    levels = naive["forecast"].to_numpy().reshape(-1, HORIZON)
    pieces.append(run_policy(NAIVE, np.nan, fold, demand, levels, series_ids))

    # Textbook rule: mean + z * standard deviation
    mean, spread = weekday_mean_and_spread(fold, series_ids)
    for tau in NORMAL_TAUS:
        levels = np.tile(mean + norm.ppf(tau) * spread, HORIZON // 7)
        pieces.append(run_policy(NORMAL_RULE, tau, fold, demand, levels, series_ids))

    # The three quantile forecasts
    for model_name, frame in frames.items():
        part = fold_part(frame, fold)
        assert np.array_equal(part["sales"].to_numpy().reshape(-1, HORIZON), demand)
        for tau in MODEL_TAUS:
            levels = part[f"q{tau}"].to_numpy().reshape(-1, HORIZON)
            pieces.append(run_policy(model_name, tau, fold, demand, levels, series_ids))

results = pd.concat(pieces)
results.to_parquet("data/processed/inventory_series_level_results.parquet", index=False)


def summarize(data):
    totals = data.groupby(["model", "tau"], dropna=False).sum(numeric_only=True)
    return pd.DataFrame({
        "fill_rate": totals["units_sold"] / totals["units_demanded"],
        "avg_inventory": totals["inventory_units_nights"] / totals["series_days"],
    }).round(4)


frontier_all = summarize(results)
frontier_fold4 = summarize(results[results["fold"] == 4])

# Check against the earlier run (target 0.95, shortage cost 19, all folds pooled)
earlier = pd.read_csv("reports/inventory_simulation_pooled.csv")
earlier = earlier[earlier["shortage_cost"] == SHORTAGE_COST].set_index("model")
for model_name in [NORMAL_RULE] + MODEL_NAMES:
    now = frontier_all.loc[(model_name, 0.95)]
    assert abs(now["fill_rate"] - earlier.loc[model_name, "fill_rate"]) < 0.0015
    assert abs(now["avg_inventory"] - earlier.loc[model_name, "avg_inventory"]) < 0.0015
print("Check passed: matches the earlier simulation run")
print()


def gain_at_same_inventory(frontier):
    """Fill-rate gain over the normal rule, at the same average inventory."""
    normal = frontier.xs(NORMAL_RULE, level="model").sort_values("avg_inventory")
    rows = []
    for model_name in MODEL_NAMES:
        points = frontier.xs(model_name, level="model")
        for tau in COMPARE_TAUS:
            inventory = points.loc[tau, "avg_inventory"]
            fill = points.loc[tau, "fill_rate"]
            normal_fill = np.interp(inventory, normal["avg_inventory"],
                                    normal["fill_rate"], left=np.nan, right=np.nan)
            rows.append({
                "model": model_name,
                "target_quantile": tau,
                "avg_inventory": inventory,
                "fill_rate": fill,
                "normal_rule_fill_same_inventory": round(normal_fill, 4),
                "gain_in_points": round((fill - normal_fill) * 100, 2),
            })
    return pd.DataFrame(rows)


gain_fold4 = gain_at_same_inventory(frontier_fold4)
gain_all = gain_at_same_inventory(frontier_all)
print("Fill-rate gain over the normal rule at the same average inventory")
print("Fold 4 (not used for tuning):")
print(gain_fold4.to_string(index=False))
print()
print("All four folds pooled:")
print(gain_all.to_string(index=False))
print()

# Does the benefit come from fast or slow sellers? (target 0.95, shortage cost 19)
at_95 = results[results["tau"] == 0.95].merge(quartiles, on="id")
rows = []
for quartile, group in at_95.groupby("vol_quartile"):
    totals = group.groupby("model").sum(numeric_only=True)
    cost = (HOLDING_COST * totals["inventory_units_nights"]
            + SHORTAGE_COST * totals["units_lost"]) / totals["series_days"]
    for model_name in totals.index:
        rows.append({
            "vol_quartile": quartile,
            "model": model_name,
            "fill_rate": round(totals.loc[model_name, "units_sold"]
                               / totals.loc[model_name, "units_demanded"], 3),
            "avg_inventory": round(totals.loc[model_name, "inventory_units_nights"]
                                   / totals.loc[model_name, "series_days"], 3),
            "cost_per_series_day": round(cost[model_name], 3),
            "cost_vs_normal_rule_pct": round((cost[model_name] / cost[NORMAL_RULE] - 1) * 100, 1),
        })
by_quartile = pd.DataFrame(rows)
print("By volume quartile, all folds pooled (target 0.95, shortage cost 19):")
print(by_quartile.to_string(index=False))
print()

gain_fold4.to_csv("reports/inventory_frontier_gain_fold4.csv", index=False)
gain_all.to_csv("reports/inventory_frontier_gain_all_folds.csv", index=False)
by_quartile.to_csv("reports/inventory_by_quartile.csv", index=False)
frontier_all.to_csv("reports/inventory_frontier_points_all_folds.csv")
frontier_fold4.to_csv("reports/inventory_frontier_points_fold4.csv")

# Frontier chart: higher and further left is better
figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
panels = [("Fold 4 (not used for tuning)", frontier_fold4),
          ("All four folds pooled", frontier_all)]
for axis, (title, frontier) in zip(axes, panels):
    for model_name in [NORMAL_RULE] + MODEL_NAMES:
        points = frontier.xs(model_name, level="model").sort_index()
        axis.plot(points["avg_inventory"], points["fill_rate"], marker="o", label=model_name)
    naive_point = frontier.xs(NAIVE, level="model")
    axis.scatter(naive_point["avg_inventory"], naive_point["fill_rate"],
                 marker="x", color="black", s=60, label=NAIVE)
    axis.set_title(title)
    axis.set_xlabel("average units on the shelf, per series per night")
axes[0].set_ylabel("fill rate (share of demand met)")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("reports/figures/inventory_frontier.png", dpi=150)
print("Saved reports/figures/inventory_frontier.png")
print("Simulation results under stated assumptions.")