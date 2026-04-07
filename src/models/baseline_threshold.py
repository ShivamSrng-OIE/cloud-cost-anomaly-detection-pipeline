import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def run_threshold_detection(df: pd.DataFrame, thresholds: list) -> dict:
    """
    Tests multiple percentage-change thresholds against the ground-truth
    is_anomaly labels. For each threshold, a row is flagged as anomalous
    if its cost_pct_change_vs_7d_avg OR cost_pct_change_vs_28d_avg
    exceeds the threshold value. This simulates how a basic alerting
    system like North.Cloud's current approach would work.

    df (pd.DataFrame): Dataset with cost_pct_change columns and is_anomaly.
    thresholds (list[float]): Threshold values to test, e.g. [0.30, 0.50].

    Returns a dict keyed by threshold value, each containing precision,
    recall, F1, counts, and per-type detection rates.
    """
    y_true = df["is_anomaly"].values
    results = {}

    for thresh in thresholds:
        y_pred = (
            (df["cost_pct_change_vs_7d_avg"].abs() > thresh)
            | (df["cost_pct_change_vs_28d_avg"].abs() > thresh)
        ).astype(int).values

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Per anomaly type detection rates
        type_rates = {}
        for atype in ["spike", "cascade", "drift"]:
            mask = df["anomaly_type"] == atype
            total = int(mask.sum())
            if total > 0:
                detected = int(((y_pred == 1) & mask).sum())
                type_rates[atype] = {
                    "total": total,
                    "detected": detected,
                    "rate": detected / total,
                }
            else:
                type_rates[atype] = {"total": 0, "detected": 0, "rate": 0.0}

        results[thresh] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "y_pred": y_pred,
            "type_rates": type_rates,
        }

        print(f"    Threshold {thresh:.0%}: "
              f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  "
              f"FP={fp:,}  FN={fn:,}")

    return results


def analyze_threshold_failures(df: pd.DataFrame, threshold: float) -> tuple:
    """
    Breaks down the false positives and false negatives for the given
    threshold to understand systematic failure patterns. False positives
    are analyzed by day-of-week and service to reveal seasonal bias.
    False negatives are analyzed by anomaly type to show which anomaly
    categories the threshold approach misses entirely.

    df (pd.DataFrame): Dataset with cost_pct_change columns, is_anomaly,
        anomaly_type, day_of_week, day_name, and service columns.
    threshold (float): The threshold value to analyze.

    Returns (false_positives_df, false_negatives_df).
    """
    y_pred = (
        (df["cost_pct_change_vs_7d_avg"].abs() > threshold)
        | (df["cost_pct_change_vs_28d_avg"].abs() > threshold)
    ).astype(int).values

    y_true = df["is_anomaly"].values

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    fp_df = df[fp_mask].copy()
    fn_df = df[fn_mask].copy()

    if len(fp_df) > 0:
        print(f"  False positives ({len(fp_df):,}) by day of week:")
        fp_by_day = fp_df.groupby("day_name").size().sort_values(ascending=False)
        for day, count in fp_by_day.items():
            print(f"    {day:<12s} {count:,}")

        print(f"  False positives by service:")
        fp_by_svc = fp_df.groupby("service").size().sort_values(ascending=False)
        for svc, count in fp_by_svc.head(5).items():
            print(f"    {svc:<20s} {count:,}")

    if len(fn_df) > 0:
        print(f"  False negatives ({len(fn_df):,}) by anomaly type:")
        fn_by_type = fn_df.groupby("anomaly_type").size().sort_values(ascending=False)
        for atype, count in fn_by_type.items():
            print(f"    {atype:<12s} {count:,}")

    return fp_df, fn_df
