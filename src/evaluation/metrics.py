import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
    model_name: str = "",
) -> dict:
    """
    Computes standard binary classification metrics for a single model's
    predictions against the ground-truth is_anomaly labels.

    y_true (np.ndarray): Ground truth binary labels.
    y_pred (np.ndarray): Predicted binary labels.
    y_score (np.ndarray, optional): Continuous anomaly scores for ROC AUC.
    model_name (str): Display name for print output.

    Returns a dict with precision, recall, f1, accuracy, confusion matrix,
    and optionally roc_auc.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    result = {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "confusion_matrix": cm,
    }

    if y_score is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_score)
            result["roc_auc"] = roc_auc
        except ValueError:
            result["roc_auc"] = np.nan
    else:
        result["roc_auc"] = np.nan

    if model_name:
        print(f"    {model_name}:")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall:    {recall:.4f}")
        print(f"      F1:        {f1:.4f}")
        print(f"      Accuracy:  {accuracy:.4f}")
        if not np.isnan(result["roc_auc"]):
            print(f"      ROC AUC:   {result['roc_auc']:.4f}")
        print(f"      TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    return result


def evaluate_by_anomaly_type(
    test_df: pd.DataFrame,
    pred_col: str,
    model_name: str = "",
) -> dict:
    """
    Breaks down detection performance by anomaly type (spike, cascade,
    drift). For each type, computes the detection rate as the fraction
    of actual anomalous rows of that type that were correctly flagged.

    test_df (pd.DataFrame): Test data with is_anomaly, anomaly_type,
        and the model's prediction column.
    pred_col (str): Name of the binary prediction column.
    model_name (str): Display name for print output.

    Returns a dict keyed by anomaly type with total, detected, and
    detection_rate.
    """
    type_results = {}

    for atype in ["spike", "cascade", "drift"]:
        mask = test_df["anomaly_type"] == atype
        total = int(mask.sum())

        if total > 0:
            detected = int(((test_df[pred_col] == 1) & mask).sum())
            rate = detected / total
        else:
            detected = 0
            rate = 0.0

        type_results[atype] = {
            "total": total,
            "detected": detected,
            "detection_rate": rate,
        }

    if model_name:
        print(f"    {model_name} by anomaly type:")
        for atype, stats in type_results.items():
            print(f"      {atype:<10s}: {stats['detected']:,}/{stats['total']:,} "
                  f"({stats['detection_rate']:.1%})")

    return type_results


def generate_comparison_report(
    test_df: pd.DataFrame,
    reports_dir: str,
) -> pd.DataFrame:
    """
    Builds a side-by-side comparison table of all models' performance
    metrics and saves it as a CSV. The models compared are: baseline
    threshold (best F1 from cached results if available), STL, LightGBM,
    Isolation Forest, and Ensemble.

    test_df (pd.DataFrame): Test data with all model prediction and
        score columns.
    reports_dir (str): Directory to save the comparison CSV.

    Returns a DataFrame with one row per model and columns for each metric.
    """
    os.makedirs(reports_dir, exist_ok=True)

    y_true = test_df["is_anomaly"].values

    models = [
        {
            "name": "STL Decomposition",
            "pred_col": "stl_is_anomaly",
            "score_col": "stl_anomaly_score",
        },
        {
            "name": "LightGBM",
            "pred_col": "lgbm_is_anomaly",
            "score_col": "lgbm_anomaly_score",
        },
        {
            "name": "Isolation Forest",
            "pred_col": "iforest_is_anomaly",
            "score_col": "iforest_anomaly_score",
        },
        {
            "name": "Ensemble",
            "pred_col": "ensemble_is_anomaly",
            "score_col": "ensemble_score",
        },
    ]

    rows = []
    for model_info in models:
        name = model_info["name"]
        pred_col = model_info["pred_col"]
        score_col = model_info["score_col"]

        if pred_col not in test_df.columns:
            continue

        y_pred = test_df[pred_col].values
        y_score = test_df[score_col].values if score_col in test_df.columns else None

        metrics = evaluate_model(y_true, y_pred, y_score, model_name=name)
        type_metrics = evaluate_by_anomaly_type(test_df, pred_col)

        row = {
            "Model": name,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Accuracy": metrics["accuracy"],
            "ROC_AUC": metrics.get("roc_auc", np.nan),
            "TP": metrics["tp"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "TN": metrics["tn"],
            "Spike_Detection": type_metrics["spike"]["detection_rate"],
            "Cascade_Detection": type_metrics["cascade"]["detection_rate"],
            "Drift_Detection": type_metrics["drift"]["detection_rate"],
        }
        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    csv_path = os.path.join(reports_dir, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n    Model comparison saved: {csv_path}")

    # Print formatted table
    print("\n    Model Comparison Summary:")
    print(f"    {'Model':<22s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} "
          f"{'AUC':>7s} {'Spike':>7s} {'Casc':>7s} {'Drift':>7s}")
    print(f"    {'-'*76}")

    for _, row in comparison_df.iterrows():
        auc_str = f"{row['ROC_AUC']:.4f}" if not np.isnan(row["ROC_AUC"]) else "  N/A "
        print(f"    {row['Model']:<22s} "
              f"{row['Precision']:7.4f} {row['Recall']:7.4f} {row['F1']:7.4f} "
              f"{auc_str:>7s} "
              f"{row['Spike_Detection']:7.1%} {row['Cascade_Detection']:7.1%} "
              f"{row['Drift_Detection']:7.1%}")

    return comparison_df
