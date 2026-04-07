import numpy as np
import pandas as pd


def run_ensemble(
    test_df: pd.DataFrame,
    weights: dict,
    threshold: float,
) -> pd.DataFrame:
    """
    Combines anomaly scores from STL, LightGBM, and Isolation Forest
    into a single ensemble score via weighted average. Each model's
    raw score is first normalized to [0, 1] using min-max scaling
    across the test set, then the weighted sum produces the final
    ensemble_score.

    Rows where ensemble_score exceeds the threshold are flagged as
    anomalies in the ensemble_is_anomaly column.

    test_df (pd.DataFrame): Test data with stl_anomaly_score,
        lgbm_anomaly_score, and iforest_anomaly_score columns already
        populated by the individual model modules.
    weights (dict): Model weights with keys 'stl', 'lightgbm',
        'isolation_forest'. Values should sum to 1.0.
    threshold (float): Score threshold above which a row is flagged.

    Returns the test_df with ensemble_score and ensemble_is_anomaly
    columns added.
    """
    test_df = test_df.copy()

    score_cols = {
        "stl": "stl_anomaly_score",
        "lightgbm": "lgbm_anomaly_score",
        "isolation_forest": "iforest_anomaly_score",
    }

    # Min-max normalize each model's scores to [0, 1]
    normalized = {}
    for model_key, col in score_cols.items():
        raw = test_df[col].values.astype(float)
        col_min = np.nanmin(raw)
        col_max = np.nanmax(raw)

        if col_max - col_min > 0:
            normed = (raw - col_min) / (col_max - col_min)
        else:
            normed = np.zeros_like(raw)

        normalized[model_key] = normed

    # Weighted combination
    w_stl = weights.get("stl", 0.25)
    w_lgbm = weights.get("lightgbm", 0.45)
    w_iforest = weights.get("isolation_forest", 0.30)

    ensemble_score = (
        w_stl * normalized["stl"]
        + w_lgbm * normalized["lightgbm"]
        + w_iforest * normalized["isolation_forest"]
    )

    test_df["ensemble_score"] = ensemble_score
    test_df["ensemble_is_anomaly"] = (ensemble_score > threshold).astype(int)

    flagged = int(test_df["ensemble_is_anomaly"].sum())
    total = len(test_df)

    # Compare with individual model counts
    stl_count = int(test_df["stl_is_anomaly"].sum())
    lgbm_count = int(test_df["lgbm_is_anomaly"].sum())
    iforest_count = int(test_df["iforest_is_anomaly"].sum())

    print(f"    Weights: STL={w_stl:.2f}, LightGBM={w_lgbm:.2f}, "
          f"IsolationForest={w_iforest:.2f}")
    print(f"    Threshold: {threshold:.2f}")
    print(f"    Ensemble anomalies flagged: {flagged:,} / {total:,}")
    print(f"    Individual model counts: "
          f"STL={stl_count:,}, LightGBM={lgbm_count:,}, "
          f"IsolationForest={iforest_count:,}")

    # Agreement analysis
    all_agree = int((
        (test_df["stl_is_anomaly"] == 1)
        & (test_df["lgbm_is_anomaly"] == 1)
        & (test_df["iforest_is_anomaly"] == 1)
    ).sum())

    any_one = int((
        (test_df["stl_is_anomaly"] == 1)
        | (test_df["lgbm_is_anomaly"] == 1)
        | (test_df["iforest_is_anomaly"] == 1)
    ).sum())

    print(f"    All 3 models agree: {all_agree:,}")
    print(f"    At least 1 model flags: {any_one:,}")

    return test_df
