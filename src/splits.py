import duckdb
import pandas as pd

HORIZON = 28   # days to forecast in every fold

# Last day of training for each fold. The test window is the next 28 days.
TRAIN_ENDS = {
    1: "2015-06-19",
    2: "2015-09-18",
    3: "2015-12-18",
    4: "2016-03-18",
}


def load_scope_sales():
    """Daily sales for the 720 series in our frozen scope."""
    con = duckdb.connect()
    query = """
    SELECT id, cat_id, store_id, date, sales
    FROM 'data/processed/sales_long.parquet'
    WHERE id IN (SELECT id FROM 'data/scope/scope_skus.csv')
    ORDER BY id, date
    """
    df = con.execute(query).df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_fold(df, fold):
    """Split the data into train and test for one fold. No shuffling."""
    train_end = pd.Timestamp(TRAIN_ENDS[fold])
    test_start = train_end + pd.Timedelta(days=1)
    test_end = train_end + pd.Timedelta(days=HORIZON)

    train = df[df["date"] <= train_end]
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

    # Leakage check: training must end before testing starts
    assert train["date"].max() < test["date"].min()
    return train, test