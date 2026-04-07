import numpy as np
import pandas as pd
import lightgbm as lgb


def train_lgbm_model(
    train_df: pd.DataFrame,
    feature_cols: list,
    categorical_cols: list,
    target_col: str,
    params: dict,
) -> tuple:
    """
    Trains a LightGBM regression model to predict daily_cost from the
    engineered feature set. The last 20% of the training data (by date)
    is held out as a validation set for early stopping, preventing
    overfitting to recent patterns.

    Categorical features are passed natively to LightGBM via the
    categorical_feature parameter so it can use optimal split-finding
    rather than one-hot encoding.

    train_df (pd.DataFrame): Training data with feature and target columns.
    feature_cols (list[str]): Numerical feature column names.
    categorical_cols (list[str]): Categorical feature column names.
    target_col (str): Name of the prediction target column ("daily_cost").
    params (dict): LightGBM parameters from pipeline_config.yaml.

    Returns (trained_model, val_predictions, residual_stats) where
    residual_stats is a dict with 'mean' and 'std' of training residuals.
    """
    all_features = feature_cols + categorical_cols

    # Convert categoricals to pandas category dtype for LightGBM
    train_df = train_df.copy()
    for col in categorical_cols:
        train_df[col] = train_df[col].astype("category")

    # Chronological val split: last 20% of training dates
    all_dates = sorted(train_df["date"].unique())
    val_split_idx = int(len(all_dates) * 0.8)
    val_split_date = all_dates[val_split_idx]

    fit_mask = train_df["date"] < val_split_date
    val_mask = train_df["date"] >= val_split_date

    X_fit = train_df.loc[fit_mask, all_features]
    y_fit = train_df.loc[fit_mask, target_col]
    X_val = train_df.loc[val_mask, all_features]
    y_val = train_df.loc[val_mask, target_col]

    lgb_params = {
        "objective": params.get("objective", "regression"),
        "metric": params.get("metric", "rmse"),
        "boosting_type": params.get("boosting_type", "gbdt"),
        "num_leaves": params.get("num_leaves", 31),
        "learning_rate": params.get("learning_rate", 0.05),
        "feature_fraction": params.get("feature_fraction", 0.8),
        "bagging_fraction": params.get("bagging_fraction", 0.8),
        "bagging_freq": params.get("bagging_freq", 5),
        "verbose": params.get("verbose", -1),
        "random_state": params.get("random_state", 42),
    }

    train_set = lgb.Dataset(
        X_fit, label=y_fit,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val, label=y_val,
        categorical_feature=categorical_cols,
        reference=train_set,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(
            stopping_rounds=params.get("early_stopping_rounds", 50),
        ),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=params.get("n_estimators", 500),
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Validation predictions and RMSE
    val_preds = model.predict(X_val)
    val_residuals = y_val.values - val_preds
    val_rmse = np.sqrt(np.mean(val_residuals ** 2))

    # Full training set residual stats (used for anomaly threshold)
    X_all_train = train_df[all_features]
    y_all_train = train_df[target_col]
    train_preds = model.predict(X_all_train)
    train_residuals = y_all_train.values - train_preds
    train_rmse = np.sqrt(np.mean(train_residuals ** 2))

    residual_stats = {
        "mean": float(np.mean(train_residuals)),
        "std": float(np.std(train_residuals)),
    }

    print(f"    Training RMSE:     {train_rmse:,.2f}")
    print(f"    Validation RMSE:   {val_rmse:,.2f}")
    print(f"    Best iteration:    {model.best_iteration}")
    print(f"    Residual mean:     {residual_stats['mean']:,.2f}")
    print(f"    Residual std:      {residual_stats['std']:,.2f}")

    return model, val_preds, residual_stats


def predict_and_detect(
    model,
    test_df: pd.DataFrame,
    feature_cols: list,
    categorical_cols: list,
    train_residual_stats: dict,
    residual_threshold_sigma: float,
) -> pd.DataFrame:
    """
    Runs the trained LightGBM model on the test set, computes residuals
    (actual - predicted), and flags rows where the absolute residual
    exceeds threshold_sigma standard deviations of the training residual
    distribution.

    model: Trained LightGBM Booster.
    test_df (pd.DataFrame): Test data with feature columns.
    feature_cols (list[str]): Numerical feature column names.
    categorical_cols (list[str]): Categorical feature column names.
    train_residual_stats (dict): Training residual 'mean' and 'std'.
    residual_threshold_sigma (float): Sigma multiplier for flagging.

    Returns the test_df with lgbm_predicted_cost, lgbm_residual,
    lgbm_anomaly_score, and lgbm_is_anomaly columns added.
    """
    test_df = test_df.copy()
    all_features = feature_cols + categorical_cols

    for col in categorical_cols:
        test_df[col] = test_df[col].astype("category")

    predictions = model.predict(test_df[all_features])

    test_df["lgbm_predicted_cost"] = predictions
    test_df["lgbm_residual"] = test_df["daily_cost"] - predictions

    res_std = train_residual_stats["std"]
    res_mean = train_residual_stats["mean"]
    if res_std == 0:
        res_std = 1e-10

    test_df["lgbm_anomaly_score"] = (
        (test_df["lgbm_residual"] - res_mean).abs() / res_std
    )
    test_df["lgbm_is_anomaly"] = (
        test_df["lgbm_anomaly_score"] > residual_threshold_sigma
    ).astype(int)

    flagged = int(test_df["lgbm_is_anomaly"].sum())
    test_rmse = np.sqrt(np.mean(test_df["lgbm_residual"] ** 2))
    print(f"    Test RMSE:         {test_rmse:,.2f}")
    print(f"    Anomalies flagged: {flagged:,}")

    return test_df
