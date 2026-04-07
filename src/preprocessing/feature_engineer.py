import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the daily aggregated DataFrame with lag, rolling window,
    difference, ratio, z-score, and calendar features computed per
    (account_id, service, region) group. Rows where lag features are
    NaN (the first 28 days of each group) are dropped afterwards.

    df (pd.DataFrame): Daily aggregated data with date, daily_cost,
        daily_usage, num_resources, and calendar columns.

    Returns the enriched DataFrame with all new feature columns added.
    """
    df = df.sort_values(
        ["account_id", "service", "region", "date"],
    ).reset_index(drop=True)

    group_key = ["account_id", "service", "region"]

    grouped = df.groupby(group_key)["daily_cost"]

    # Lag features on daily_cost
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f"cost_lag_{lag}d"] = grouped.shift(lag)

    # Rolling window features on daily_cost
    for window in [3, 7, 14, 28]:
        rolling = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean(),
        )
        df[f"cost_rolling_{window}d_mean"] = rolling

        rolling_std = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std(),
        )
        df[f"cost_rolling_{window}d_std"] = rolling_std

    # Difference features
    df["cost_diff_1d"] = df["daily_cost"] - df["cost_lag_1d"]
    df["cost_diff_7d"] = df["daily_cost"] - df["cost_lag_7d"]

    # Ratio features (guard division by zero)
    df["cost_ratio_to_7d_mean"] = (
        df["daily_cost"] / df["cost_rolling_7d_mean"].replace(0, np.nan)
    ).fillna(1.0)

    df["cost_ratio_to_28d_mean"] = (
        df["daily_cost"] / df["cost_rolling_28d_mean"].replace(0, np.nan)
    ).fillna(1.0)

    # Z-score features
    df["cost_zscore_7d"] = (
        (df["daily_cost"] - df["cost_rolling_7d_mean"])
        / df["cost_rolling_7d_std"].replace(0, np.nan)
    ).fillna(0.0)

    df["cost_zscore_28d"] = (
        (df["daily_cost"] - df["cost_rolling_28d_mean"])
        / df["cost_rolling_28d_std"].replace(0, np.nan)
    ).fillna(0.0)

    # Usage lag and rolling features
    usage_grouped = df.groupby(group_key)["daily_usage"]
    df["usage_lag_1d"] = usage_grouped.shift(1)
    df["usage_lag_7d"] = usage_grouped.shift(7)
    df["usage_rolling_7d_mean"] = usage_grouped.transform(
        lambda x: x.rolling(7, min_periods=1).mean(),
    )

    # Per-unit cost features
    df["cost_per_resource"] = (
        df["daily_cost"] / df["num_resources"].replace(0, np.nan)
    ).fillna(0.0)

    df["cost_per_usage"] = (
        df["daily_cost"] / df["daily_usage"].replace(0, np.nan)
    ).fillna(0.0)

    # Calendar binary features
    df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)

    # Drop rows where the 28-day lag is still NaN (first 28 days per group)
    rows_before = len(df)
    df = df.dropna(subset=["cost_lag_28d"]).reset_index(drop=True)
    rows_dropped = rows_before - len(df)

    print(f"  Features engineered: {len(get_feature_columns()[0])} numerical, "
          f"{len(get_feature_columns()[1])} categorical")
    print(f"  Rows dropped (warm-up period): {rows_dropped:,}")
    print(f"  Rows remaining: {len(df):,}")

    return df


def get_feature_columns() -> tuple:
    """
    Returns the two lists of feature column names used by the ML models.
    Separated into numerical features (floats/ints that models consume
    directly) and categorical features (strings that need encoding).

    Returns (numerical_features, categorical_features).
    """
    numerical_features = [
        # Calendar / temporal
        "day_of_week", "is_weekend", "day_of_month", "week_of_year", "month",
        # Raw metrics
        "num_resources", "daily_usage",
        # Cost lags
        "cost_lag_1d", "cost_lag_2d", "cost_lag_3d",
        "cost_lag_7d", "cost_lag_14d", "cost_lag_28d",
        # Cost rolling means
        "cost_rolling_3d_mean", "cost_rolling_7d_mean",
        "cost_rolling_14d_mean", "cost_rolling_28d_mean",
        # Cost rolling stds
        "cost_rolling_3d_std", "cost_rolling_7d_std",
        "cost_rolling_14d_std", "cost_rolling_28d_std",
        # Cost diffs
        "cost_diff_1d", "cost_diff_7d",
        # Cost ratios
        "cost_ratio_to_7d_mean", "cost_ratio_to_28d_mean",
        # Cost z-scores
        "cost_zscore_7d", "cost_zscore_28d",
        # Usage features
        "usage_lag_1d", "usage_lag_7d", "usage_rolling_7d_mean",
        # Per-unit costs
        "cost_per_resource", "cost_per_usage",
        # Month boundary flags
        "is_month_start", "is_month_end",
    ]

    categorical_features = [
        "service", "account_name", "region", "day_name",
    ]

    return numerical_features, categorical_features
