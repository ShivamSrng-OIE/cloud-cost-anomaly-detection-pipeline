import os

import numpy as np
import pandas as pd
import shap


def compute_shap_values(
    model,
    test_df: pd.DataFrame,
    feature_cols: list,
    categorical_cols: list,
    max_samples: int,
) -> tuple:
    """
    Computes SHAP values for the LightGBM model using TreeExplainer.
    If the test set exceeds max_samples, a random subsample is used
    to keep computation time manageable.

    model: Trained LightGBM Booster.
    test_df (pd.DataFrame): Test data with feature columns.
    feature_cols (list[str]): Numerical feature column names.
    categorical_cols (list[str]): Categorical feature column names.
    max_samples (int): Maximum number of rows to explain.

    Returns (shap_values np.ndarray, sample_df pd.DataFrame of the
    rows that were explained, explainer shap.TreeExplainer).
    """
    all_features = feature_cols + categorical_cols

    sample_df = test_df.copy()
    for col in categorical_cols:
        sample_df[col] = sample_df[col].astype("category")

    if len(sample_df) > max_samples:
        sample_df = sample_df.sample(
            n=max_samples, random_state=42,
        ).reset_index(drop=True)

    X_sample = sample_df[all_features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print(f"    SHAP values computed for {len(sample_df):,} samples")
    print(f"    Features: {len(all_features)}")

    return shap_values, sample_df, explainer


def explain_single_anomaly(
    model,
    row: pd.Series,
    feature_cols: list,
    categorical_cols: list,
    top_features: int,
) -> str:
    """
    Generates a plain-English explanation for a single anomalous row
    by computing SHAP values and identifying the most impactful features.
    The explanation includes the feature name, its value, and the
    direction and magnitude of its SHAP contribution.

    model: Trained LightGBM Booster.
    row (pd.Series): Single row from the test DataFrame.
    feature_cols (list[str]): Numerical feature column names.
    categorical_cols (list[str]): Categorical feature column names.
    top_features (int): Number of top features to include.

    Returns a multi-line string explaining why this row was flagged.
    """
    all_features = feature_cols + categorical_cols

    row_df = pd.DataFrame([row])
    for col in categorical_cols:
        row_df[col] = row_df[col].astype("category")

    X = row_df[all_features]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)[0]

    # Pair features with their SHAP values, sorted by absolute impact
    feature_impacts = sorted(
        zip(all_features, sv, X.values[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    lines = []
    lines.append(
        f"Anomaly on {row['date'].strftime('%Y-%m-%d')} | "
        f"Account: {row.get('account_name', row.get('account_id', 'N/A'))} | "
        f"Service: {row.get('service', 'N/A')} | "
        f"Region: {row.get('region', 'N/A')}"
    )
    lines.append(
        f"Actual cost: ${row['daily_cost']:,.2f} | "
        f"Predicted: ${row.get('lgbm_predicted_cost', 0):,.2f} | "
        f"Ensemble score: {row.get('ensemble_score', 0):.3f}"
    )
    lines.append(f"Top {top_features} contributing features:")

    for feat, shap_val, feat_val in feature_impacts[:top_features]:
        direction = "↑ increased" if shap_val > 0 else "↓ decreased"
        lines.append(
            f"  {feat}: value={feat_val:.4f}, "
            f"SHAP={shap_val:+.4f} ({direction} predicted cost)"
        )

    return "\n".join(lines)


def generate_anomaly_report(
    model,
    test_df: pd.DataFrame,
    feature_cols: list,
    categorical_cols: list,
    top_features: int,
    reports_dir: str,
) -> pd.DataFrame:
    """
    Creates a CSV report of all ensemble-flagged anomalies with their
    SHAP-based top feature explanations, plus individual text explanations
    for each anomaly saved to a text file.

    model: Trained LightGBM Booster.
    test_df (pd.DataFrame): Test data with ensemble_is_anomaly and model
        score columns.
    feature_cols (list[str]): Numerical feature column names.
    categorical_cols (list[str]): Categorical feature column names.
    top_features (int): Number of top features per anomaly.
    reports_dir (str): Directory to save the report files.

    Returns a DataFrame summarizing each anomaly with key fields and
    the top contributing feature.
    """
    os.makedirs(reports_dir, exist_ok=True)

    anomalies = test_df[test_df["ensemble_is_anomaly"] == 1].copy()

    if len(anomalies) == 0:
        print("    No ensemble anomalies to report.")
        return pd.DataFrame()

    all_features = feature_cols + categorical_cols

    # Prepare data for batch SHAP computation
    anom_for_shap = anomalies.copy()
    for col in categorical_cols:
        anom_for_shap[col] = anom_for_shap[col].astype("category")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(anom_for_shap[all_features])

    # Build summary records
    records = []
    explanations = []

    for i, (idx, row) in enumerate(anomalies.iterrows()):
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[-1]
        top_feat = all_features[top_idx]
        top_shap = sv[top_idx]

        records.append({
            "date": row["date"],
            "account_id": row["account_id"],
            "account_name": row.get("account_name", ""),
            "service": row.get("service", ""),
            "region": row.get("region", ""),
            "daily_cost": row["daily_cost"],
            "lgbm_predicted_cost": row.get("lgbm_predicted_cost", np.nan),
            "ensemble_score": row["ensemble_score"],
            "stl_anomaly_score": row["stl_anomaly_score"],
            "lgbm_anomaly_score": row["lgbm_anomaly_score"],
            "iforest_anomaly_score": row["iforest_anomaly_score"],
            "is_actual_anomaly": row["is_anomaly"],
            "actual_anomaly_type": row["anomaly_type"],
            "top_feature": top_feat,
            "top_feature_shap": top_shap,
        })

        # Build text explanation from pre-computed batch SHAP values
        feature_impacts = sorted(
            zip(all_features, sv, anom_for_shap[all_features].iloc[i].values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        lines = []
        lines.append(
            f"Anomaly on {row['date'].strftime('%Y-%m-%d')} | "
            f"Account: {row.get('account_name', row.get('account_id', 'N/A'))} | "
            f"Service: {row.get('service', 'N/A')} | "
            f"Region: {row.get('region', 'N/A')}"
        )
        lines.append(
            f"Actual cost: ${row['daily_cost']:,.2f} | "
            f"Predicted: ${row.get('lgbm_predicted_cost', 0):,.2f} | "
            f"Ensemble score: {row.get('ensemble_score', 0):.3f}"
        )
        lines.append(f"Top {top_features} contributing features:")
        for feat, shap_val, feat_val in feature_impacts[:top_features]:
            direction = "↑ increased" if shap_val > 0 else "↓ decreased"
            lines.append(
                f"  {feat}: value={feat_val:.4f}, "
                f"SHAP={shap_val:+.4f} ({direction} predicted cost)"
            )
        explanations.append("\n".join(lines))

    report_df = pd.DataFrame(records)
    report_df = report_df.sort_values("ensemble_score", ascending=False)

    # Save CSV report
    csv_path = os.path.join(reports_dir, "anomaly_report.csv")
    report_df.to_csv(csv_path, index=False)
    print(f"    Anomaly report saved: {csv_path}")
    print(f"    Total anomalies explained: {len(report_df):,}")

    # Save text explanations
    txt_path = os.path.join(reports_dir, "anomaly_explanations.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, explanation in enumerate(explanations, 1):
            f.write(f"{'='*70}\n")
            f.write(f"Anomaly #{i}\n")
            f.write(f"{'='*70}\n")
            f.write(explanation + "\n\n")

    print(f"    Text explanations saved: {txt_path}")

    return report_df
