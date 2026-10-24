import pandas as pd
from scoring import score

files = {
    "quantile baseline": "data/processed/baseline_quantiles.parquet",
    "LightGBM": "data/processed/lgbm_quantiles.parquet",
}

# Both models must be scored on exactly the same rows
keys = {}
rows = []
for name, path in files.items():
    df = pd.read_parquet(path)
    keys[name] = df[["fold", "id", "date"]].sort_values(["fold", "id", "date"]).reset_index(drop=True)
    for fold, group in df.groupby("fold"):
        row = score(group)
        row["model"] = name
        row["fold"] = fold
        rows.append(row)

names = list(files)
assert keys[names[0]].equals(keys[names[1]]), "Models were scored on different rows"

table = pd.DataFrame(rows)
table = table[["model", "fold", "mean_pinball", "wape_median", "cov_80", "cov_90", "cov_95", "width_90"]]
print(table.sort_values(["fold", "model"]).round(3).to_string(index=False))
table.sort_values(["fold", "model"]).round(3).to_csv("reports/model_comparison.csv", index=False)