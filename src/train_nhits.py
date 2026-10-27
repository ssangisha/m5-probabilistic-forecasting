# train_nhits.py
# Trains one N-HiTS model per fold with a multi-quantile loss and saves the forecasts
# in the same format as the other models.

import re
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss
from splits import load_scope_sales, get_fold, TRAIN_ENDS, HORIZON
from scoring import TAUS

INPUT_SIZE = 56          # look back 8 weeks
MAX_STEPS = 500          # fixed in advance, not tuned on any fold
LEVELS = [80, 90, 95]    # gives quantiles 0.1/0.9, 0.05/0.95, 0.025/0.975, plus the median


def column_to_quantile(column):
    """Turn a column name like 'NHITS-lo-90' into the quantile it holds (0.05)."""
    if column.endswith("-median"):
        return 0.5
    match = re.search(r"-(lo|hi)-([0-9.]+)$", column)
    if match is None:
        return None
    side = match.group(1)
    level = float(match.group(2))
    tail = (1 - level / 100) / 2
    return round(tail if side == "lo" else 1 - tail, 3)


data = load_scope_sales()
quantile_columns = [f"q{tau}" for tau in TAUS]

all_results = []
for fold in TRAIN_ENDS:
    train, test = get_fold(data, fold)

    neural_train = train[["id", "date", "sales"]].rename(
        columns={"id": "unique_id", "date": "ds", "sales": "y"})
    neural_train["y"] = neural_train["y"].astype(float)

    model = NHITS(h=HORIZON, input_size=INPUT_SIZE, loss=MQLoss(level=LEVELS),
                  max_steps=MAX_STEPS, scaler_type="standard",
                  random_seed=42, logger=False)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=neural_train)

    forecast = nf.predict()
    if "unique_id" not in forecast.columns:
        forecast = forecast.reset_index()

    # Rename the model's columns to q0.05, q0.5 and so on, to match the other models
    renamed = {}
    for column in forecast.columns:
        quantile = column_to_quantile(column)
        if quantile is not None:
            renamed[column] = f"q{quantile}"
    forecast = forecast.rename(columns=renamed)
    forecast = forecast.rename(columns={"unique_id": "id", "ds": "date"})

    # Attach the forecasts to the real sales of the test window
    result = test[["id", "date", "sales"]].merge(
        forecast[["id", "date"] + quantile_columns], on=["id", "date"], how="left")
    assert len(result) == len(test)
    assert result[quantile_columns].notna().all().all(), "Some forecasts are missing"

    # Sales cannot be negative, and quantiles must not cross (same fix as LightGBM)
    values = np.maximum(result[quantile_columns].values, 0)
    result[quantile_columns] = np.sort(values, axis=1)

    result["fold"] = fold
    all_results.append(result)
    print(f"fold {fold} done")

all_results = pd.concat(all_results)
all_results.to_parquet("data/processed/nhits_quantiles.parquet", index=False)
print(all_results.groupby("fold").size())