# compare_nhits_seeds.py
# Shows how much N-HiTS results move with the random seed, next to LightGBM.

import pandas as pd
from scoring import score

FILES = {
    "N-HiTS seed 42": "data/processed/nhits_quantiles.parquet",
    "N-HiTS seed 43": "data/processed/nhits_quantiles_seed43.parquet",
    "N-HiTS seed 44": "data/processed/nhits_quantiles_seed44.parquet",
    "LightGBM tuned": "data/processed/lgbm_tuned_quantiles.parquet",
}

rows = []
for name, path in FILES.items():
    forecasts = pd.read_parquet(path)
    for fold, group in forecasts.groupby("fold"):
        result = score(group)
        rows.append({"model": name, "fold": fold,
                     "mean_pinball": result["mean_pinball"],
                     "wape_median": result["wape_median"]})
table = pd.DataFrame(rows)

seed_names = [name for name in FILES if name.startswith("N-HiTS")]
for measure in ["mean_pinball", "wape_median"]:
    wide = table.pivot(index="fold", columns="model", values=measure)
    wide["range across N-HiTS seeds"] = wide[seed_names].max(axis=1) - wide[seed_names].min(axis=1)
    print(measure, "(lower is better)")
    print(wide.round(4).to_string())
    print()
    