import duckdb

con = duckdb.connect()

for f in ["calendar", "sell_prices", "sales_train_evaluation"]:
    n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('data/raw/{f}.csv')").fetchone()[0]
    print(f"{f}: {n:,} rows")