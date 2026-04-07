import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder


def run_isolation_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    categorical_cols: list,
    params: dict,
) -> tuple:
    """
    Fits an Isolation Forest on the training data and scores the test
    data. Categorical features are label-encoded (fitted on train,
    applied to both). The decision_function scores are negated so that
    higher values indicate more anomalous points, matching the
    convention used by the other models.

    train_df (pd.DataFrame): Training data with feature columns.
    test_df (pd.DataFrame): Test data to score.
    feature_cols (list[str]): Numerical feature column names.
    categorical_cols (list[str]): Categorical feature column names.
    params (dict): IsolationForest parameters from pipeline_config.yaml.

    Returns (modified test_df, label_encoders dict) where test_df has
    iforest_anomaly_score and iforest_is_anomaly columns added.
    """
    test_df = test_df.copy()

    all_features = feature_cols + categorical_cols

    # Label-encode categoricals: fit on train, transform both
    label_encoders = {}
    train_encoded = train_df[all_features].copy()
    test_encoded = test_df[all_features].copy()

    for col in categorical_cols:
        le = LabelEncoder()
        train_encoded[col] = le.fit_transform(train_encoded[col].astype(str))

        # Handle unseen categories in test by mapping to -1
        test_vals = test_encoded[col].astype(str)
        test_encoded[col] = test_vals.map(
            lambda x, _le=le: (
                _le.transform([x])[0] if x in _le.classes_ else -1
            )
        )
        label_encoders[col] = le

    # Replace any remaining NaN/inf with 0
    train_encoded = train_encoded.replace([np.inf, -np.inf], np.nan).fillna(0)
    test_encoded = test_encoded.replace([np.inf, -np.inf], np.nan).fillna(0)

    iso_forest = IsolationForest(
        n_estimators=params.get("n_estimators", 200),
        contamination=params.get("contamination", 0.03),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", -1),
    )

    print(f"    Fitting Isolation Forest on {len(train_encoded):,} rows...")
    iso_forest.fit(train_encoded)

    # Score test data: decision_function returns negative for anomalies
    raw_scores = iso_forest.decision_function(test_encoded)
    predictions = iso_forest.predict(test_encoded)

    # Negate scores so higher = more anomalous
    test_df["iforest_anomaly_score"] = -raw_scores
    test_df["iforest_is_anomaly"] = (predictions == -1).astype(int)

    flagged = int(test_df["iforest_is_anomaly"].sum())
    print(f"    Anomalies flagged: {flagged:,}")

    return test_df, label_encoders
