# Probabilistic Demand Forecasting for Retail Inventory Decisions (M5 Dataset)
Kavita SVL

Capstone project for the IIT Roorkee PG Certification in Data Science and AI.

## About the project

A store has to decide how much stock to keep before it knows what customers will
buy. If you forecast one number per day, you hide how uncertain the demand is. In
this project I forecast daily demand as a range instead of one number, using seven
quantiles from 2.5% to 97.5%. Then I run a simple stocking simulation to see if
those ranges lead to better decisions than the usual safety-stock rule.

The data is the M5 retail sales data from Kaggle. I used a sample of 720
product-store series: 3 categories, 10 stores, 4 sales-volume groups and 6 series
in each combination. About 60% of the daily values are zero, so most items sell
rarely and unevenly.

## What I compared

1. Seasonal-naive: the same weekday last week.
2. Empirical quantiles: quantiles of the last 8 same weekdays.
3. LightGBM with a quantile loss, tuned on folds 1 to 3 only.
4. N-HiTS, a neural network, trained with a multi-quantile loss.

All models are tested on the same four windows of 28 days. Training always ends
before the window starts. Fold 4 was never used for tuning, so I treat it as the
fairest test.

## Results

Forecast quality on fold 4, as mean pinball loss (lower is better):

- empirical quantile baseline: 0.231
- LightGBM: 0.195
- N-HiTS: 0.194 (0.195 with two other random seeds)

Both models are about 16% better than the baseline. When I resample the 720
series, the interval is roughly 12% to 20%. LightGBM and N-HiTS can't be told
apart on fold 4. N-HiTS was a bit better on folds 1 and 2, and LightGBM was
clearly better on fold 3, which has Christmas in it.

Calibration: if a model says the 95% quantile is X, sales should be at or below X
on about 95% of days. For the baseline this was true on 89.1% of days, so its
upper quantiles are too low. For LightGBM it was 95.0% and for N-HiTS 94.8%.
Sales are whole numbers with many zeros, so I check this using the share of days
strictly below the forecast and the share at or below it.

Inventory simulation: each morning the stock is topped up to a level chosen by
the policy, and demand that can't be met is lost. I compared the policies against
the textbook rule (average demand plus a safety buffer from the standard
deviation).

- Stocking from LightGBM or N-HiTS quantiles gave about 8% to 20% lower simulated
  cost than the textbook rule. All the bootstrap intervals stayed below zero.
- For the same average stock, they met about 2 to 4 percentage points more of the
  demand.
- The empirical quantile baseline was no better than the textbook rule, and at
  the highest service level it was about 10% worse.

This is a simulation under assumptions I wrote down in
`reports/inventory_assumptions.md`. It is not a claim about money saved in a real
business, and the costs are in relative units.

## Limits

- It is 720 series, not all 30,490, so I would not extend the results to the whole
  dataset.
- I only used series with at least two years of history, so new products are not
  covered.
- The LightGBM features are at least 28 days old. I did this on purpose so that
  every forecast could really be made at the start of the window.
- The simulation assumes overnight delivery, forecasts that are not updated during
  the window, lost sales, equal value for all items, and no order costs or shelf
  life.
- The bootstrap intervals only cover which series ended up in the sample. For
  random training effects I checked N-HiTS with three seeds.
- My numbers can't be compared with the M5 Kaggle leaderboard, which uses a
  different metric and different quantiles.

## Folder layout

```
data/
  raw/         M5 CSV files from Kaggle (not in the repository)
  processed/   Parquet files made by the code (not in the repository)
  scope/       scope_skus.csv, the list of the 720 series
notebooks/     notebooks (still in progress)
reports/
  figures/     charts
  forecasts/   small forecast files that the dashboard reads
  *.csv        result tables
  inventory_assumptions.md
src/           all the code
requirements.txt
```

## How to run it

I built and ran this on Windows with Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download `calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` from
the Kaggle competition "M5 Forecasting - Accuracy"
(https://www.kaggle.com/competitions/m5-forecasting-accuracy) and put them in
`data/raw/`. I don't include the data here. Run every command from the main
project folder, for example `python src/data_prep.py`.

The scripts in the order I ran them:

1. `data_prep.py` turns the raw files into one clean Parquet file and removes the
   days before a product was launched.
2. `make_scope.py` picks the 720 series, using only data before 2015-06-20.
3. `eda_intermittency.py`, `eda_seasonality.py` and `eda_hierarchy.py` are the
   exploratory analysis.
4. `features.py` builds the lag, rolling, price and calendar features.
5. `test_baseline.py` makes the seasonal-naive forecasts. It is a run script even
   though the name says test.
6. `baseline_quantiles.py` makes the empirical quantile forecasts.
7. `train_lgbm.py` trains LightGBM, and `tune_lightgbm.py` compares settings on
   folds 1 to 3.
8. `train_nhits.py` trains N-HiTS. You can pass a seed, for example
   `python src/train_nhits.py 43`.
9. `compare_models.py` and `score_by_quartile.py` score the models.
10. `calibration_check.py` checks the quantiles.
11. `bootstrap_model_differences.py` and `compare_nhits_seeds.py` check how
    reliable the differences are.
12. `run_inventory_simulation.py`, `inventory_frontier.py` and
    `bootstrap_inventory_results.py` run the inventory simulation.
13. `export_dashboard_data.py` copies the small forecast files for the dashboard.

Some scripts only check my work. `test_metrics.py`, `test_features.py` and
`test_inventory_policy.py` compare the code with small examples I worked out by
hand, and `check_data_integrity.py` checks the cleaned data, the scope list and
all the forecast files. `splits.py`, `metrics.py`, `scoring.py` and
`inventory_policy.py` are helper files that the other scripts import.

## Dashboard

```powershell
streamlit run src/dashboard.py
```

The dashboard only reads files in `reports/`, so it works without the large data
files.

## Notes on reproducing

- I fixed the random seeds: 42 for LightGBM and the bootstrap, and 42, 43 and 44
  for N-HiTS.
- `train_lgbm.py` now has the tuned setting. The "untuned" results I compare
  against came from an earlier run of the same script with `num_leaves=31`,
  saved as `lgbm_quantiles.parquet`.

## Tools

Python 3.12, DuckDB, pandas, LightGBM, neuralforecast, Streamlit.