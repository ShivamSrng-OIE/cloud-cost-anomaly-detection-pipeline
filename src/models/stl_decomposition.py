import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def run_stl_detection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    period: int,
    residual_threshold_sigma: float,
) -> pd.DataFrame:
    """
    Runs STL decomposition on each (account_id, service, region) group.
    The decomposition is fitted on the full available history (train + test
    concatenated per group) so seasonal and trend patterns are estimated
    across the entire span. Residual statistics (mean and std) are computed
    from the training portion only, and anomaly flags are applied to test
    rows where the absolute residual exceeds threshold_sigma standard
    deviations from the training mean.

    Groups with fewer than 2 * period data points are skipped because
    STL requires at least that much data for a reliable decomposition.

    train_df (pd.DataFrame): Training split with date, daily_cost, and
        group key columns.
    test_df (pd.DataFrame): Test split to score and flag.
    period (int): Seasonal period for STL (7 for weekly).
    residual_threshold_sigma (float): Number of standard deviations
        beyond which a residual is flagged anomalous.

    Returns the test_df with stl_residual, stl_seasonal, stl_trend,
    stl_anomaly_score, and stl_is_anomaly columns added.
    """
    test_df = test_df.copy()
    test_df["stl_residual"] = 0.0
    test_df["stl_seasonal"] = 0.0
    test_df["stl_trend"] = 0.0
    test_df["stl_anomaly_score"] = 0.0
    test_df["stl_is_anomaly"] = 0

    group_key = ["account_id", "service", "region"]
    split_date = test_df["date"].min()

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(group_key + ["date"]).reset_index(drop=True)

    groups = full_df.groupby(group_key)
    skipped = 0
    processed = 0

    for name, group in groups:
        if len(group) < 2 * period:
            skipped += 1
            continue

        series = group.set_index("date")["daily_cost"].asfreq("D")
        series = series.interpolate(method="linear").bfill().ffill()

        if series.isna().any() or len(series) < 2 * period:
            skipped += 1
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stl_result = STL(series, period=period, robust=True).fit()
            except Exception:
                skipped += 1
                continue

        residuals = stl_result.resid
        trend = stl_result.trend
        seasonal = stl_result.seasonal

        # Compute residual stats from training portion only
        train_residuals = residuals[residuals.index < split_date]
        if len(train_residuals) < 5:
            skipped += 1
            continue

        res_mean = train_residuals.mean()
        res_std = train_residuals.std()
        if res_std == 0:
            res_std = 1e-10

        # Apply to test rows in this group
        test_mask = (
            (test_df["account_id"] == name[0])
            & (test_df["service"] == name[1])
            & (test_df["region"] == name[2])
        )
        test_dates = test_df.loc[test_mask, "date"]

        for idx, d in test_dates.items():
            if d in residuals.index:
                r = residuals[d]
                test_df.at[idx, "stl_residual"] = r
                test_df.at[idx, "stl_seasonal"] = seasonal[d]
                test_df.at[idx, "stl_trend"] = trend[d]

                score = abs(r - res_mean) / res_std
                test_df.at[idx, "stl_anomaly_score"] = score
                test_df.at[idx, "stl_is_anomaly"] = (
                    1 if score > residual_threshold_sigma else 0
                )

        processed += 1

    print(f"    Groups processed: {processed:,}")
    print(f"    Groups skipped (too few data points): {skipped:,}")

    stl_flagged = int(test_df["stl_is_anomaly"].sum())
    print(f"    Anomalies flagged: {stl_flagged:,}")

    return test_df
