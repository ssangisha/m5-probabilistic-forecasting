import duckdb

con = duckdb.connect()
con.execute("SET preserve_insertion_order = false")  # uses less memory

con.execute(r"""
COPY (
    WITH sales_long AS (
        -- wide -> long: one row per series per day
        SELECT id, item_id, dept_id, cat_id, store_id, state_id, d, sales
        FROM (
            UNPIVOT read_csv_auto('data/raw/sales_train_evaluation.csv')
            ON COLUMNS('^d_\d+$')
            INTO NAME d VALUE sales
        )
    ),
    launch AS (
        -- first week each item-store pair appears in the price file = launch week
        SELECT store_id, item_id, MIN(wm_yr_wk) AS first_wk
        FROM read_csv_auto('data/raw/sell_prices.csv')
        GROUP BY store_id, item_id
    )
    SELECT
        s.id, s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id,
        c.date,
        s.sales,
        p.sell_price,
        (p.sell_price IS NULL) AS price_missing,   -- flagged, never silently filled
        c.* EXCLUDE (d, date)
    FROM sales_long s
    JOIN read_csv_auto('data/raw/calendar.csv') c ON s.d = c.d
    JOIN launch l ON s.store_id = l.store_id AND s.item_id = l.item_id
    LEFT JOIN read_csv_auto('data/raw/sell_prices.csv') p
           ON p.store_id = s.store_id AND p.item_id = s.item_id AND p.wm_yr_wk = c.wm_yr_wk
    WHERE c.wm_yr_wk >= l.first_wk                -- drop pre-launch days
) TO 'data/processed/sales_long.parquet' (FORMAT parquet)
""")

# quick sanity check
n, nulls, zeros = con.execute("""
    SELECT COUNT(*),
           SUM(price_missing::INT),
           AVG((sales = 0)::INT)
    FROM 'data/processed/sales_long.parquet'
""").fetchone()
print(f"rows after truncation: {n:,}")
print(f"rows with missing price: {nulls:,}")
print(f"zero rate: {zeros:.1%}")