import duckdb

CUTOFF = "2015-06-20"
con = duckdb.connect()

# Total sales and number of series, by category and department
query = f"""
SELECT cat_id, dept_id,
       COUNT(DISTINCT id) AS n_series,
       SUM(sales) AS total_sales
FROM 'data/processed/sales_long.parquet'
WHERE date < DATE '{CUTOFF}'
  AND id IN (SELECT id FROM 'data/scope/scope_skus.csv')
GROUP BY cat_id, dept_id
ORDER BY cat_id, dept_id
"""
by_dept = con.execute(query).df()
by_dept["share_pct"] = (by_dept["total_sales"] / by_dept["total_sales"].sum() * 100).round(1)
print(by_dept)

# Same idea by store
query = f"""
SELECT store_id,
       COUNT(DISTINCT id) AS n_series,
       SUM(sales) AS total_sales
FROM 'data/processed/sales_long.parquet'
WHERE date < DATE '{CUTOFF}'
  AND id IN (SELECT id FROM 'data/scope/scope_skus.csv')
GROUP BY store_id
ORDER BY store_id
"""
by_store = con.execute(query).df()
by_store["share_pct"] = (by_store["total_sales"] / by_store["total_sales"].sum() * 100).round(1)
print(by_store)