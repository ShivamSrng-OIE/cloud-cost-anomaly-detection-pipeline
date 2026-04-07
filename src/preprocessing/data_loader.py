import os

import pandas as pd


def load_data(data_dir: str) -> tuple:
    """
    Loads the daily_aggregated.csv and anomaly_log.csv files from the
    specified data directory. Parses dates, casts account_id to string,
    sorts chronologically, and prints basic dataset statistics.

    data_dir (str): Path to the directory containing the generated CSVs.

    Returns a tuple of (daily_df, anomaly_log_df).
    """
    agg_path = os.path.join(data_dir, "daily_aggregated.csv")
    log_path = os.path.join(data_dir, "anomaly_log.csv")

    if not os.path.exists(agg_path):
        raise FileNotFoundError(
            f"Dataset not found in {data_dir}. "
            "Run main.py first to generate the synthetic dataset."
        )
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Anomaly log not found in {data_dir}. "
            "Run main.py first to generate the synthetic dataset."
        )

    daily_df = pd.read_csv(agg_path)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["account_id"] = daily_df["account_id"].astype(str)

    # anomaly_type and cascade_id may be NaN for non-anomaly rows
    daily_df["anomaly_type"] = daily_df["anomaly_type"].fillna("").astype(str)
    daily_df["cascade_id"] = daily_df["cascade_id"].fillna("").astype(str)

    daily_df = daily_df.sort_values(
        ["date", "account_id", "service", "region"],
    ).reset_index(drop=True)

    anomaly_log = pd.read_csv(log_path)
    anomaly_log["date_start"] = pd.to_datetime(anomaly_log["date_start"])
    anomaly_log["date_end"] = pd.to_datetime(anomaly_log["date_end"])
    anomaly_log["account_id"] = anomaly_log["account_id"].astype(str)

    print(f"  Rows loaded:       {len(daily_df):,}")
    print(f"  Date range:        {daily_df['date'].min().date()} → {daily_df['date'].max().date()}")
    print(f"  Unique accounts:   {daily_df['account_id'].nunique()}")
    print(f"  Unique services:   {daily_df['service'].nunique()}")
    print(f"  Unique regions:    {daily_df['region'].nunique()}")

    anom_counts = daily_df[daily_df["is_anomaly"] == 1]["anomaly_type"].value_counts()
    print(f"  Anomaly rows:      {daily_df['is_anomaly'].sum():,}")
    for atype, count in anom_counts.items():
        print(f"    {atype:<12s}     {count:,}")

    return daily_df, anomaly_log


def split_train_test(df: pd.DataFrame, train_ratio: float) -> tuple:
    """
    Splits the DataFrame chronologically so that approximately train_ratio
    of the total date span falls into the training set and the remainder
    into the test set. This prevents future data leakage.

    df (pd.DataFrame): Full dataset with a datetime 'date' column.
    train_ratio (float): Fraction of the date range for training (e.g. 0.7).

    Returns (train_df, test_df, split_date).
    """
    all_dates = sorted(df["date"].unique())
    n_dates = len(all_dates)
    split_idx = int(n_dates * train_ratio)
    split_date = all_dates[split_idx]

    train_df = df[df["date"] < split_date].copy().reset_index(drop=True)
    test_df = df[df["date"] >= split_date].copy().reset_index(drop=True)

    split_date_val = pd.Timestamp(split_date)

    print(f"  Train period:      {train_df['date'].min().date()} → {train_df['date'].max().date()}")
    print(f"  Test period:       {test_df['date'].min().date()} → {test_df['date'].max().date()}")
    print(f"  Split date:        {split_date_val.date()}")
    print(f"  Train rows:        {len(train_df):,}")
    print(f"  Test rows:         {len(test_df):,}")

    train_anom = train_df["is_anomaly"].sum()
    test_anom = test_df["is_anomaly"].sum()
    print(f"  Train anomalies:   {train_anom:,}")
    print(f"  Test anomalies:    {test_anom:,}")

    return train_df, test_df, split_date_val
