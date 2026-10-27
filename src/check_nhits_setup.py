# check_nhits_setup.py
# Quick test: trains a tiny N-HiTS on 20 series and shows what comes out.

import time
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss
from splits import load_scope_sales, get_fold

data = load_scope_sales()
train, test = get_fold(data, 1)

# Only 20 series, to keep the test fast
some_ids = train["id"].unique()[:20]
train = train[train["id"].isin(some_ids)]

# NeuralForecast expects the columns unique_id, ds and y
neural_train = train[["id", "date", "sales"]].rename(
    columns={"id": "unique_id", "date": "ds", "sales": "y"})
neural_train["y"] = neural_train["y"].astype(float)

model = NHITS(h=28, input_size=56, loss=MQLoss(level=[80, 90, 95]),
              max_steps=20, scaler_type="standard", random_seed=42, logger=False)
nf = NeuralForecast(models=[model], freq="D")

start = time.time()
nf.fit(df=neural_train)
forecast = nf.predict()

print(f"Took {time.time() - start:.0f} seconds")
print(forecast.columns.tolist())
print(forecast.head())
print("Rows:", len(forecast), "(expected 560)")