import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def detect_cascades(
    test_df: pd.DataFrame,
    eps_days: float,
    min_samples: int,
) -> pd.DataFrame:
    """
    Identifies cascade anomalies by clustering ensemble-flagged rows
    per account using DBSCAN. A cascade is a coordinated anomaly that
    spans multiple services within a short time window for the same
    account.

    For each account with at least 2 ensemble-flagged rows, DBSCAN
    clusters on a feature matrix of [date_ordinal] only. The date
    ordinal is scaled so that eps_days controls temporal proximity.
    Service diversity is checked post-clustering: clusters that
    contain anomalous rows from 2 or more distinct services are
    marked as predicted cascades.

    test_df (pd.DataFrame): Test data with ensemble_is_anomaly column.
    eps_days (float): Maximum number of days apart for DBSCAN
        neighborhood (maps to eps after scaling).
    min_samples (int): Minimum cluster size for DBSCAN.

    Returns the test_df with predicted_cascade_id and
    is_predicted_cascade columns added.
    """
    test_df = test_df.copy()
    test_df["predicted_cascade_id"] = ""
    test_df["is_predicted_cascade"] = 0

    anomalous = test_df[test_df["ensemble_is_anomaly"] == 1].copy()

    if len(anomalous) == 0:
        print("    No ensemble anomalies to cluster.")
        return test_df

    cascade_counter = 0
    total_cascade_rows = 0
    accounts = anomalous["account_id"].unique()

    for account_id in accounts:
        acct_mask = anomalous["account_id"] == account_id
        acct_df = anomalous[acct_mask]

        if len(acct_df) < min_samples:
            continue

        # Build feature matrix: date ordinal only.
        # Service diversity is checked post-clustering, so including
        # service one-hot in features would push different services apart
        # and prevent the cross-service clusters we're looking for.
        date_ordinal = acct_df["date"].map(
            lambda d: d.toordinal()
        ).values.reshape(-1, 1).astype(float)

        # Scale date ordinal so eps corresponds to eps_days
        scaler = StandardScaler()
        date_scaled = scaler.fit_transform(date_ordinal)

        # Compute the scale factor: what 1 day looks like in scaled space
        if scaler.scale_[0] > 0:
            one_day_scaled = 1.0 / scaler.scale_[0]
        else:
            one_day_scaled = 1.0

        eps_scaled = eps_days * one_day_scaled

        features = date_scaled

        dbscan = DBSCAN(eps=eps_scaled, min_samples=min_samples)
        labels = dbscan.fit_predict(features)

        # Check each cluster for multi-service presence
        acct_indices = acct_df.index.tolist()
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_mask = labels == cluster_id
            cluster_rows = acct_df.iloc[cluster_mask]
            unique_services = cluster_rows["service"].nunique()

            if unique_services >= 2:
                cascade_counter += 1
                cascade_label = f"pred_cascade_{cascade_counter}"

                for idx in cluster_rows.index:
                    test_df.at[idx, "predicted_cascade_id"] = cascade_label
                    test_df.at[idx, "is_predicted_cascade"] = 1
                    total_cascade_rows += 1

    # Compare with ground truth cascades
    actual_cascade_rows = int(
        (test_df["cascade_id"] != "").sum()
    )

    print(f"    Predicted cascades found: {cascade_counter}")
    print(f"    Predicted cascade rows:   {total_cascade_rows:,}")
    print(f"    Actual cascade rows:      {actual_cascade_rows:,}")
    print(f"    Accounts analyzed:        {len(accounts)}")

    return test_df
