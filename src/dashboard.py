# dashboard.py
# Streamlit dashboard that displays the saved results of the project.
# Run with: streamlit run src/dashboard.py
# It only reads files in the reports folder, so nothing is retrained.

import os
import pandas as pd
import streamlit as st

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

tab_accuracy, tab_calibration, tab_inventory, tab_about = st.tabs(
    ["Forecast accuracy", "Calibration", "Inventory decisions", "About"]
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