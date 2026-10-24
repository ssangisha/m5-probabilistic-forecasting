import duckdb
import matplotlib.pyplot as plt
import pandas as pd

CUTOFF = "2015-06-20"
con = duckdb.connect()

# Get daily sales for our 720 series, before the cutoff only
query = f"""
SELECT id, cat_id, date, sales
FROM 'data/processed/sales_long.parquet'
WHERE date < DATE '{CUTOFF}'
  AND id IN (SELECT id FROM 'data/scope/scope_skus.csv')
"""
sales = con.execute(query).df()
print("Rows loaded:", len(sales))

# Work out ADI, CV2 and VMR for each series
results = []
for series_id, group in sales.groupby("id"):
    daily = group["sales"]
    nonzero = daily[daily > 0]

    adi = len(daily) / len(nonzero)
    cv2 = (nonzero.std() / nonzero.mean()) ** 2
    vmr = daily.var() / daily.mean()

    results.append({
        "id": series_id,
        "cat_id": group["cat_id"].iloc[0],
        "adi": adi,
        "cv2": cv2,
        "vmr": vmr,
    })

profile = pd.DataFrame(results)

# Label each series using the standard cutoffs
def label(row):
    if row["adi"] < 1.32 and row["cv2"] < 0.49:
        return "smooth"
    if row["adi"] < 1.32:
        return "erratic"
    if row["cv2"] < 0.49:
        return "intermittent"
    return "lumpy"

profile["demand_type"] = profile.apply(label, axis=1)
profile.to_csv("data/scope/series_profile.csv", index=False)

# Print the summary
print()
print(profile["demand_type"].value_counts())
print()
print("Median VMR:", round(profile["vmr"].median(), 2))
print("Share of series with VMR > 1:", round((profile["vmr"] > 1).mean(), 3))

# Draw the ADI / CV2 chart
plt.figure(figsize=(7, 5))
plt.scatter(profile["adi"], profile["cv2"], s=10, alpha=0.5)
plt.axvline(1.32, color="red", linestyle="--")
plt.axhline(0.49, color="red", linestyle="--")
plt.xscale("log")
plt.xlabel("ADI (average days between sales)")
plt.ylabel("CV2 of non-zero sales")
plt.title("Demand types for the 720 series")
plt.savefig("reports/figures/adi_cv2.png", dpi=150, bbox_inches="tight")
print("Saved reports/figures/adi_cv2.png")