# dashboard.py
# Streamlit dashboard that displays the saved results of the project.
# Run with: streamlit run src/dashboard.py
# It only reads files in the reports folder, so nothing is retrained.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from inventory_policy import simulate_base_stock

REPORTS = "reports"

st.set_page_config(page_title="Probabilistic Demand Forecasting", layout="wide")


def load_csv(name):
    path = os.path.join(REPORTS, name)
    if not os.path.exists(path):
        st.warning(f"Missing file: {path}. Run the script that creates it.")
        return None
    return pd.read_csv(path)


def show_image(name, caption):
    path = os.path.join(REPORTS, "figures", name)
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.warning(f"Missing file: {path}")


st.title("Probabilistic Demand Forecasting for Retail Inventory Decisions (M5 Dataset)")
st.write(
    "Business question: if demand is uncertain, do forecasts that give a range of "
    "likely values lead to better stocking decisions than a single-number forecast "
    "with a standard safety-stock rule?"
)
st.caption(
    "Based on a stratified sample of 720 M5 product-store series. Inventory results "
    "are simulation results under stated assumptions, in relative cost units, "
    "not savings achieved in a real business."
)

tab_accuracy, tab_calibration, tab_inventory, tab_explorer, tab_about = st.tabs(
    ["Forecast accuracy", "Calibration", "Inventory decisions",
     "Explore one product", "About"]
)
# ---------------------------------------------------------------- explorer
FORECAST_FILES = {
    "quantile baseline": "baseline_quantiles.parquet",
    "LightGBM tuned": "lgbm_tuned_quantiles.parquet",
    "N-HiTS": "nhits_quantiles.parquet",
}
SCOPE_FILE = os.path.join("data", "scope", "scope_skus.csv")
FOLD_LABELS = {
    1: "Fold 1 (Jun-Jul 2015)",
    2: "Fold 2 (Sep-Oct 2015)",
    3: "Fold 3 (Dec 2015 - Jan 2016, includes Christmas)",
    4: "Fold 4 (Mar-Apr 2016, held out)",
}


@st.cache_data
def load_forecasts(file_name):
    forecasts = pd.read_parquet(os.path.join(REPORTS, "forecasts", file_name))
    forecasts["date"] = pd.to_datetime(forecasts["date"])
    return forecasts


@st.cache_data
def load_scope():
    return pd.read_csv(SCOPE_FILE)[["id", "cat_id", "store_id", "vol_quartile"]]


def get_window(file_name, series_id, fold):
    """Forecasts and actual sales of one product in one 28-day test window."""
    forecasts = load_forecasts(file_name)
    window = forecasts[(forecasts["id"] == series_id) & (forecasts["fold"] == fold)]
    return window.sort_values("date")


def stock_levels(window, target):
    """Order-up-to level for each day: the forecast quantile in whole units."""
    return np.maximum(np.round(window[f"q{target}"].to_numpy()), 0)


with tab_explorer:
    st.subheader("Explore one product")
    st.write(
        "Pick a product and a 28-day test window. Bars are actual daily sales. "
        "The shaded bands are the model's forecast range, and the red line is the "
        "stock level the policy would set each morning."
    )

    forecast_folder = os.path.join(REPORTS, "forecasts")
    missing = [name for name in FORECAST_FILES.values()
               if not os.path.exists(os.path.join(forecast_folder, name))]
    if missing or not os.path.exists(SCOPE_FILE):
        st.warning("Forecast files are missing. Run src/export_dashboard_data.py first.")
    else:
        scope = load_scope()

        left, middle, right = st.columns(3)
        category = left.selectbox("Category", sorted(scope["cat_id"].unique()))
        store = middle.selectbox("Store", sorted(scope["store_id"].unique()))
        group = right.selectbox(
            "Sales volume group (1 = slowest, 4 = fastest)", [1, 2, 3, 4], index=3)
        candidates = scope[(scope["cat_id"] == category) & (scope["store_id"] == store)
                           & (scope["vol_quartile"] == group)]
        series_id = st.selectbox("Product", sorted(candidates["id"]))

        first, second, third = st.columns(3)
        model_name = first.selectbox("Model", list(FORECAST_FILES))
        fold = second.selectbox("Test window", [4, 3, 2, 1],
                                format_func=lambda number: FOLD_LABELS[number])
        target = third.selectbox("Stocking quantile", [0.9, 0.95, 0.975], index=1)

        window = get_window(FORECAST_FILES[model_name], series_id, fold)
        if window.empty:
            st.warning("No forecasts found for this product and window.")
        else:
            levels = stock_levels(window, target)

            figure, axis = plt.subplots(figsize=(11, 4))
            axis.fill_between(window["date"], window["q0.05"], window["q0.95"],
                              alpha=0.2, label="5% to 95% range")
            axis.fill_between(window["date"], window["q0.1"], window["q0.9"],
                              alpha=0.35, label="10% to 90% range")
            axis.plot(window["date"], window["q0.5"], label="median forecast")
            axis.bar(window["date"], window["sales"], width=0.6, color="black",
                     alpha=0.6, label="actual sales")
            axis.step(window["date"], levels, where="mid", color="red",
                      label=f"stock level ({target} quantile)")
            axis.set_ylabel("units per day")
            axis.legend(loc="upper left", fontsize=8)
            figure.autofmt_xdate()
            st.pyplot(figure)
            plt.close(figure)

            result = simulate_base_stock(window["sales"].to_numpy()[None, :],
                                         levels[None, :])
            demanded = int(window["sales"].sum())
            lost = int(result["units_lost"][0])
            average_stock = result["inventory_units_nights"][0] / len(window)
            one, two, three, four = st.columns(4)
            one.metric("Units demanded", demanded)
            two.metric("Units lost", lost)
            three.metric("Stockout days", int(result["stockout_days"][0]))
            four.metric("Average stock at night", f"{average_stock:.1f}")

            st.write("Same product and window, all three models")
            rows = []
            for name, file_name in FORECAST_FILES.items():
                part = get_window(file_name, series_id, fold)
                part_levels = stock_levels(part, target)
                part_result = simulate_base_stock(part["sales"].to_numpy()[None, :],
                                                  part_levels[None, :])
                rows.append({
                    "model": name,
                    "units_lost": int(part_result["units_lost"][0]),
                    "stockout_days": int(part_result["stockout_days"][0]),
                    "average_stock": round(part_result["inventory_units_nights"][0] / len(part), 2),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)

            st.caption(
                "One product over 28 days is a single example, not evidence. Do not "
                "judge a model from one chart. The Inventory tab shows results over "
                "all 720 products."
            )
# ---------------------------------------------------------------- accuracy
with tab_accuracy:
    st.subheader("How accurate are the forecasts?")
    st.write(
        "Each model forecasts 28 days ahead and is tested on four periods it has "
        "never seen. Lower mean pinball loss is better. It scores the whole range "
        "of forecasts (seven quantile levels), not just one number."
    )

    comparison = load_csv("model_comparison.csv")
    if comparison is not None:
        period = st.radio(
            "Test period",
            ["Fold 4 (held out, not used for tuning)", "Average of the four folds"],
            horizontal=True,
            key="accuracy_period",
        )
        if period.startswith("Fold 4"):
            table = comparison[comparison["fold"] == 4].drop(columns="fold")
        else:
            table = comparison.groupby("model").mean(numeric_only=True)
            table = table.drop(columns="fold").reset_index()
        st.dataframe(table.sort_values("mean_pinball").round(3), hide_index=True)

    st.write("By sales volume group")
    scorecard = load_csv("scorecard_by_quartile.csv")
    if scorecard is not None:
        quartile = st.selectbox(
            "Volume group (1 = slowest sellers, 4 = fastest)", [1, 2, 3, 4], index=3)
        part = scorecard[scorecard["vol_quartile"] == quartile]
        part = part.drop(columns="vol_quartile").sort_values("mean_pinball")
        st.dataframe(part, hide_index=True)

    st.caption(
        "Interval coverage and width are shown for completeness. About 60% of daily "
        "sales are zero, which makes interval coverage look better than it is. "
        "The Calibration tab is the reliable check."
    )

# ------------------------------------------------------------- calibration
with tab_calibration:
    st.subheader("Can the forecast probabilities be trusted?")
    st.write(
        "If a model says the 95% quantile is 5 units, then sales should be at or "
        "below 5 on about 95% of days. Sales are whole numbers with many zeros, so "
        "for each quantile we show a range: the share of days strictly below the "
        "forecast and the share at or below it. A bar crossing the dashed line "
        "means the quantile is calibrated. A bar below the line means the forecast "
        "is too low, which in a store means stock-outs."
    )
    show_image("quantile_calibration.png", "Quantile calibration, all four folds pooled")

    calibration = load_csv("quantile_calibration.csv")
    if calibration is not None:
        subset = st.selectbox("Which series?", list(calibration["subset"].unique()))
        upper = calibration[(calibration["subset"] == subset) & (calibration["tau"] >= 0.9)]
        st.dataframe(upper.drop(columns="subset"), hide_index=True)

# --------------------------------------------------------------- inventory
with tab_inventory:
    st.subheader("Do better forecasts lead to better stocking decisions?")
    st.write(
        "A simple daily replenishment simulation: each morning stock is topped up "
        "to a level chosen by each policy, and demand that cannot be met is lost. "
        "The textbook rule is the usual safety-stock method: average demand plus a "
        "buffer based on its variation."
    )

    assumptions_path = os.path.join(REPORTS, "inventory_assumptions.md")
    if os.path.exists(assumptions_path):
        with st.expander("Assumptions and limits (please read first)"):
            with open(assumptions_path, encoding="utf-8") as file:
                st.markdown(file.read())

    period = st.radio(
        "Test period",
        ["Fold 4 (held out, not used for tuning)", "All four folds pooled"],
        horizontal=True,
        key="inventory_period",
    )
    use_fold_4 = period.startswith("Fold 4")

    st.write("Stock held against demand met")
    show_image("inventory_frontier.png",
               "Higher and further left is better: more demand met for the same stock")

    st.write("Fill-rate gain over the textbook rule at the same average stock")
    gain_file = ("inventory_frontier_gain_fold4.csv" if use_fold_4
                 else "inventory_frontier_gain_all_folds.csv")
    gains = load_csv(gain_file)
    if gains is not None:
        st.dataframe(gains, hide_index=True)

    st.write("Simulated cost by policy")
    cost_file = ("inventory_simulation_fold4.csv" if use_fold_4
                 else "inventory_simulation_pooled.csv")
    costs = load_csv(cost_file)
    if costs is not None:
        shortage = st.selectbox(
            "Cost of one lost unit, compared with holding one unit overnight",
            [9, 19, 39], index=1,
        )
        part = costs[costs["shortage_cost"] == shortage].drop(columns="shortage_cost")
        st.dataframe(part.sort_values("cost_per_series_day"), hide_index=True)

    st.write("By sales volume group (target quantile 0.95, all folds pooled)")
    by_quartile = load_csv("inventory_by_quartile.csv")
    if by_quartile is not None:
        st.dataframe(by_quartile, hide_index=True)

# ------------------------------------------------------------------- about
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Data.** M5 retail sales (store-level daily sales). A stratified sample of 720
series: 3 categories x 10 stores x 4 sales-volume groups x 6 series. About 60%
of daily values are zero, and most series sell rarely and unevenly.

**Models, from simple to complex.**
1. Seasonal-naive: same weekday last week.
2. Empirical quantiles of the last 8 same weekdays.
3. LightGBM quantile regression, tuned on folds 1-3 only.
4. N-HiTS neural network with a multi-quantile loss.

**Testing.** Four rolling-origin periods, 28 days each, identical for every
model. Fold 4 was never used for tuning.

**Main findings (as of this version).**
- LightGBM and N-HiTS lower the mean pinball loss by about 16% against the
  empirical-quantile baseline on fold 4, and are statistically indistinguishable
  from each other.
- At the upper quantiles that matter for stocking, the baseline is too low (its
  95% quantile covered demand on about 89% of days). LightGBM and N-HiTS are
  close to nominal.
- In the simulation, stocking from LightGBM or N-HiTS quantiles lowers simulated
  cost by roughly 8% to 20% against the textbook safety-stock rule, and meets
  about 2 to 4 percentage points more demand at the same average stock. The
  empirical-quantile baseline gives no gain over the textbook rule.

**Limits.** 720 series, not the whole dataset. Overnight replenishment, forecasts
fixed at the start of each window, all items valued equally, no order costs or
shelf life, no new products. Costs are relative units, not money.
        """
    )