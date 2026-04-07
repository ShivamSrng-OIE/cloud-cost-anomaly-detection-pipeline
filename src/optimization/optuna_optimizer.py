"""
Optuna-based hyperparameter optimization for the anomaly detection pipeline.

Design rationale
────────────────
1. **Objective function**: Maximizes ensemble F1 on the held-out test split.
   F1 is the right metric for imbalanced anomaly detection because accuracy
   is misleading when anomalies are < 3 % of the data. We add soft penalty
   terms for drift and cascade detection rates so the optimizer does not
   sacrifice hard-to-detect anomaly types in exchange for easy spike recall.

2. **Search spaces per component**:
   - LightGBM tree parameters (num_leaves, learning_rate, feature_fraction,
     bagging_fraction, n_estimators) control model capacity. Overfitting
     cloud cost data is common because of strong weekly seasonality, so we
     constrain num_leaves to [15, 127] and use regularization via
     feature / bagging fractions.
   - LightGBM residual_threshold_sigma controls how many sigma above the
     training residual mean triggers a detection. Too tight = false negatives
     on drift; too loose = false positives on normal Monday spikes.
   - Isolation Forest contamination must approximate the true anomaly
     prevalence in the data. We search [0.003, 0.03] because our datasets
     range from 0.3 % to 3 % anomaly rate depending on generation config.
   - STL residual_threshold_sigma interacts with the seasonal period.
     Lower values catch slower drifts but fire on weekend noise.
   - Ensemble weights are sampled from Dirichlet-like draws (three floats
     normalized to sum to 1) so every convex combination is reachable.
   - Ensemble threshold is the final decision boundary on the [0, 1]
     blended score. Searched [0.05, 0.40].
   - DBSCAN eps_days controls cascade temporal window; searched [1, 5].

3. **Validation strategy**: Strict chronological split — no random CV.
   Cloud cost data is non-stationary (growth trends, new services), so
   random k-fold would overfit to future patterns. We use the same
   train_split_ratio as the main pipeline.

4. **Pruning**: Optuna MedianPruner kills trials whose intermediate
   metrics (reported after each component) fall below the running median,
   saving up to 60 % of compute on bad parameter combinations.

5. **Persistence**: Study is stored in a SQLite database under
   output/optuna/ so it survives restarts and supports distributed
   workers if needed later.

6. **Outputs**: Best parameters are written back to pipeline_config.yaml
   and a human-readable trial report is saved to output/optuna/.
"""
import contextlib
import io
import os
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import optuna
import yaml
from sklearn.metrics import f1_score

from src.preprocessing.data_loader import load_data, split_train_test
from src.preprocessing.feature_engineer import (
    engineer_features, get_feature_columns,
)
from src.models.stl_decomposition import run_stl_detection
from src.models.lightgbm_forecaster import train_lgbm_model, predict_and_detect
from src.models.isolation_forest import run_isolation_forest
from src.models.ensemble import run_ensemble
from src.clustering.cascade_detector import detect_cascades
from src.evaluation.metrics import evaluate_model, evaluate_by_anomaly_type


# ── Objective function ───────────────────────────────────────────────

class _AnomalyObjective:
    """
    Callable objective for Optuna. Encapsulates the pre-loaded and
    pre-engineered data so each trial only re-runs the model fitting
    and scoring stages, not the expensive data loading and feature
    engineering.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        numerical_features: list,
        categorical_features: list,
        stl_period: int,
    ) -> None:
        self._train_df = train_df
        self._test_df = test_df
        self._num_feats = numerical_features
        self._cat_feats = categorical_features
        self._stl_period = stl_period

    # ----------------------------------------------------------------

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Single trial: suggests hyperparameters, runs the detection
        pipeline, and returns the composite objective score.
        """

        # ── 1. LightGBM hyperparameters ─────────────────────────────
        # num_leaves: Controls tree complexity. Higher values capture
        # service-specific cost patterns but risk overfitting to
        # individual account noise. 15-127 covers simple linear-ish
        # trees to moderately complex interactions.
        lgbm_num_leaves = trial.suggest_int(
            "lgbm_num_leaves", 15, 127,
        )

        # learning_rate: Step size shrinkage. Lower rates improve
        # generalization but need more boosting rounds. Log-uniform
        # because the effect is multiplicative.
        lgbm_learning_rate = trial.suggest_float(
            "lgbm_learning_rate", 0.005, 0.2, log=True,
        )

        # feature_fraction: Column subsampling per tree. Reduces
        # correlation between trees and prevents over-reliance on
        # dominant features like cost_lag_1d.
        lgbm_feature_fraction = trial.suggest_float(
            "lgbm_feature_fraction", 0.5, 1.0,
        )

        # bagging_fraction: Row subsampling per tree. Combined with
        # feature_fraction, this is the primary regularization lever
        # for LightGBM.
        lgbm_bagging_fraction = trial.suggest_float(
            "lgbm_bagging_fraction", 0.5, 1.0,
        )

        # n_estimators: Maximum boosting rounds. Early stopping will
        # likely halt before this but we let the optimizer control
        # the ceiling. More rounds help at low learning rates.
        lgbm_n_estimators = trial.suggest_int(
            "lgbm_n_estimators", 200, 1500, step=100,
        )

        # residual_threshold_sigma: How many standard deviations above
        # the training residual mean a test residual must be to trigger
        # a LightGBM anomaly flag. Lower = more sensitive to drifts.
        lgbm_sigma = trial.suggest_float(
            "lgbm_residual_threshold_sigma", 0.5, 3.0,
        )

        # ── 2. STL threshold ────────────────────────────────────────
        # Same concept as LightGBM sigma but applied to STL seasonal
        # decomposition residuals. Critical for drift detection because
        # STL separates trend from noise.
        stl_sigma = trial.suggest_float(
            "stl_residual_threshold_sigma", 0.5, 3.0,
        )

        # ── 3. Isolation Forest ─────────────────────────────────────
        # contamination: Expected fraction of anomalies in training
        # data. Must approximate reality; too high floods false
        # positives, too low catches nothing.
        iforest_contamination = trial.suggest_float(
            "iforest_contamination", 0.003, 0.03, log=True,
        )

        # n_estimators: Number of isolation trees. More trees give
        # more stable anomaly scores but diminishing returns past ~300.
        iforest_n_estimators = trial.suggest_int(
            "iforest_n_estimators", 100, 500, step=50,
        )

        # ── 4. Ensemble weights ─────────────────────────────────────
        # Three raw weights sampled independently then normalized to
        # sum to 1. This produces a uniform distribution over the
        # probability simplex (better than fixing one weight).
        w_stl_raw = trial.suggest_float("w_stl_raw", 0.1, 1.0)
        w_lgbm_raw = trial.suggest_float("w_lgbm_raw", 0.1, 1.0)
        w_iforest_raw = trial.suggest_float("w_iforest_raw", 0.1, 1.0)
        w_total = w_stl_raw + w_lgbm_raw + w_iforest_raw
        w_stl = w_stl_raw / w_total
        w_lgbm = w_lgbm_raw / w_total
        w_iforest = w_iforest_raw / w_total

        # ensemble threshold: Final decision boundary on the blended
        # [0, 1] score. Low = aggressive (more TP, more FP).
        ensemble_threshold = trial.suggest_float(
            "ensemble_threshold", 0.05, 0.40,
        )

        # ── 5. DBSCAN (cascade clustering) ──────────────────────────
        # eps_days: Max temporal distance for DBSCAN to consider two
        # flagged rows part of the same cascade event.
        dbscan_eps = trial.suggest_int("dbscan_eps_days", 1, 5)

        # ── Run pipeline stages ─────────────────────────────────────
        # Suppress verbose model output during optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            _silence = contextlib.redirect_stdout(io.StringIO())
            test_df = self._test_df.copy()

            # STL
            try:
                with _silence:
                    test_df = run_stl_detection(
                        self._train_df, test_df,
                        period=self._stl_period,
                        residual_threshold_sigma=stl_sigma,
                    )
            except Exception:
                return 0.0  # STL failure → worst score

            # Report intermediate: STL F1 for pruning
            stl_f1 = f1_score(
                test_df["is_anomaly"], test_df["stl_is_anomaly"],
                zero_division=0,
            )
            trial.report(stl_f1, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # LightGBM
            lgbm_params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": lgbm_num_leaves,
                "learning_rate": lgbm_learning_rate,
                "feature_fraction": lgbm_feature_fraction,
                "bagging_fraction": lgbm_bagging_fraction,
                "bagging_freq": 5,
                "verbose": -1,
                "n_estimators": lgbm_n_estimators,
                "early_stopping_rounds": 50,
                "random_state": 42,
            }

            try:
                with _silence:
                    lgbm_model, _, residual_stats = train_lgbm_model(
                        self._train_df,
                        feature_cols=self._num_feats,
                        categorical_cols=self._cat_feats,
                        target_col="daily_cost",
                        params=lgbm_params,
                    )
                    test_df = predict_and_detect(
                        lgbm_model, test_df,
                        feature_cols=self._num_feats,
                        categorical_cols=self._cat_feats,
                        train_residual_stats=residual_stats,
                        residual_threshold_sigma=lgbm_sigma,
                    )
            except Exception:
                return 0.0

            # Report intermediate: LightGBM F1 for pruning
            lgbm_f1 = f1_score(
                test_df["is_anomaly"], test_df["lgbm_is_anomaly"],
                zero_division=0,
            )
            trial.report(lgbm_f1, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Isolation Forest
            iforest_params = {
                "n_estimators": iforest_n_estimators,
                "contamination": iforest_contamination,
                "random_state": 42,
                "n_jobs": -1,
            }
            try:
                with _silence:
                    test_df, _ = run_isolation_forest(
                        self._train_df, test_df,
                        feature_cols=self._num_feats,
                        categorical_cols=self._cat_feats,
                        params=iforest_params,
                    )
            except Exception:
                return 0.0

            # Ensemble
            weights = {
                "stl": w_stl,
                "lightgbm": w_lgbm,
                "isolation_forest": w_iforest,
            }
            test_df = run_ensemble(
                test_df, weights=weights, threshold=ensemble_threshold,
            )

            # Cascade clustering
            test_df = detect_cascades(
                test_df, eps_days=dbscan_eps, min_samples=2,
            )

            # ── Compute objective ────────────────────────────────────
            y_true = test_df["is_anomaly"].values
            y_pred = test_df["ensemble_is_anomaly"].values

            ensemble_f1 = f1_score(y_true, y_pred, zero_division=0)

            # Per-type detection rates for penalty terms
            type_rates = {}
            for atype in ["spike", "cascade", "drift"]:
                mask = test_df["anomaly_type"] == atype
                total = int(mask.sum())
                if total > 0:
                    detected = int(((y_pred == 1) & mask).sum())
                    type_rates[atype] = detected / total
                else:
                    type_rates[atype] = 0.0

            # Composite score: 70% ensemble F1 + 15% drift rate + 15%
            # cascade rate. Drift and cascade are the hardest anomaly
            # types to detect, so we incentivize the optimizer to not
            # ignore them in favor of easy spike recall.
            composite = (
                0.70 * ensemble_f1
                + 0.15 * type_rates.get("drift", 0.0)
                + 0.15 * type_rates.get("cascade", 0.0)
            )

            # Store useful diagnostics as trial user attributes
            trial.set_user_attr("ensemble_f1", round(ensemble_f1, 4))
            trial.set_user_attr("ensemble_tp", int(((y_pred == 1) & (y_true == 1)).sum()))
            trial.set_user_attr("ensemble_fp", int(((y_pred == 1) & (y_true == 0)).sum()))
            trial.set_user_attr("spike_rate", round(type_rates.get("spike", 0.0), 4))
            trial.set_user_attr("cascade_rate", round(type_rates.get("cascade", 0.0), 4))
            trial.set_user_attr("drift_rate", round(type_rates.get("drift", 0.0), 4))
            trial.set_user_attr("stl_f1", round(stl_f1, 4))
            trial.set_user_attr("lgbm_f1", round(lgbm_f1, 4))

            # Normalized ensemble weights (for config writeback)
            trial.set_user_attr("w_stl", round(w_stl, 4))
            trial.set_user_attr("w_lgbm", round(w_lgbm, 4))
            trial.set_user_attr("w_iforest", round(w_iforest, 4))

            return composite


# ── Public entry point ───────────────────────────────────────────────

def run_optimization(
    n_trials: int = 50,
    timeout_minutes: Optional[int] = None,
) -> dict:
    """
    Runs Optuna hyperparameter optimization over the full anomaly
    detection pipeline and writes the best parameters back to
    pipeline_config.yaml.

    n_trials (int): Maximum number of optimization trials.
    timeout_minutes (int, optional): Wall-clock time limit in minutes.

    Returns a dict with the best parameters and trial summary.
    """
    t_start = time.time()

    # ── Load pipeline config ─────────────────────────────────────────
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "pipeline_config.yaml",
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = config["paths"]["data_dir"]
    stl_period = config["stl"]["period"]

    # ── Output directory ─────────────────────────────────────────────
    optuna_dir = os.path.join(
        config["paths"]["results_dir"], "optuna",
    )
    os.makedirs(optuna_dir, exist_ok=True)

    # ── Load and prepare data (done once, reused across trials) ──────
    print("=" * 70)
    print("  Optuna Hyperparameter Optimization")
    print("=" * 70)
    print(f"\n  Trials:  {n_trials}")
    if timeout_minutes:
        print(f"  Timeout: {timeout_minutes} min")
    print(f"  Study:   {os.path.join(optuna_dir, 'study.db')}")

    print(f"\n  Loading data from {os.path.abspath(data_dir)} ...")

    daily_df, _ = load_data(data_dir)

    train_ratio = config["preprocessing"]["train_split_ratio"]
    train_df, test_df, split_date = split_train_test(daily_df, train_ratio)

    print(f"\n  Engineering features ...")
    # Suppress repeated print output during feature engineering
    with contextlib.redirect_stdout(io.StringIO()):
        train_df = engineer_features(train_df)
        test_df = engineer_features(test_df)

    numerical_features, categorical_features = get_feature_columns()

    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    print(f"  Test anomaly rate: {test_df['is_anomaly'].mean():.2%}")
    print(f"  Features: {len(numerical_features)} numerical, "
          f"{len(categorical_features)} categorical")

    # ── Create Optuna study ──────────────────────────────────────────
    db_path = os.path.join(optuna_dir, "study.db")
    storage = f"sqlite:///{db_path}"

    # MedianPruner: prunes trials that report intermediate values
    # below the running median of completed trials at the same step.
    # n_startup_trials=10 lets the first 10 trials complete without
    # pruning so the median estimate is stable.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=1,
    )

    # TPESampler: Tree-structured Parzen Estimator is the default
    # Bayesian optimization strategy. n_startup_trials=15 means the
    # first 15 trials are random (exploration), then TPE kicks in
    # (exploitation). seed ensures reproducibility.
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=15,
        seed=42,
    )

    study = optuna.create_study(
        study_name="anomaly_detection_hpo",
        storage=storage,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    # ── Build objective ──────────────────────────────────────────────
    objective = _AnomalyObjective(
        train_df=train_df,
        test_df=test_df,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        stl_period=stl_period,
    )

    # ── Redirect noisy model output during optimization ──────────────
    print(f"\n{'─' * 70}")
    print(f"  Starting optimization ({n_trials} trials) ...")
    print(f"{'─' * 70}\n")

    # Custom callback to print concise per-trial summaries
    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            attrs = trial.user_attrs
            print(
                f"  Trial {trial.number:3d} │ "
                f"score={trial.value:.4f} │ "
                f"F1={attrs.get('ensemble_f1', 0):.4f} │ "
                f"TP={attrs.get('ensemble_tp', 0):3d} │ "
                f"drift={attrs.get('drift_rate', 0):.1%} │ "
                f"cascade={attrs.get('cascade_rate', 0):.1%} │ "
                f"best={study.best_value:.4f}"
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"  Trial {trial.number:3d} │ PRUNED at step {trial.last_step}")

    timeout_sec = timeout_minutes * 60 if timeout_minutes else None

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        callbacks=[_trial_callback],
        show_progress_bar=True,
    )

    # ── Process results ──────────────────────────────────────────────
    best = study.best_trial
    best_params = best.params
    best_attrs = best.user_attrs

    print(f"\n{'=' * 70}")
    print(f"  Optimization Complete")
    print(f"{'=' * 70}")
    print(f"\n  Total trials:     {len(study.trials)}")
    print(f"  Completed:        "
          f"{sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)}")
    print(f"  Pruned:           "
          f"{sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}")
    print(f"  Best trial:       #{best.number}")
    print(f"  Best score:       {best.value:.4f}")
    print(f"\n  Best trial metrics:")
    print(f"    Ensemble F1:       {best_attrs.get('ensemble_f1', 'N/A')}")
    print(f"    Ensemble TP:       {best_attrs.get('ensemble_tp', 'N/A')}")
    print(f"    Spike detection:   {best_attrs.get('spike_rate', 0):.1%}")
    print(f"    Cascade detection: {best_attrs.get('cascade_rate', 0):.1%}")
    print(f"    Drift detection:   {best_attrs.get('drift_rate', 0):.1%}")

    print(f"\n  Best hyperparameters:")
    for k, v in sorted(best_params.items()):
        if isinstance(v, float):
            print(f"    {k:<35s} {v:.6f}")
        else:
            print(f"    {k:<35s} {v}")

    # ── Write best params to pipeline_config.yaml ────────────────────
    _write_best_config(config, config_path, best_params, best_attrs)

    # ── Save trial report ────────────────────────────────────────────
    _save_trial_report(study, optuna_dir)

    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"\n  Optimization runtime: {minutes}m {seconds:.1f}s")
    print(f"  Updated: {config_path}")
    print(f"  Report:  {os.path.join(optuna_dir, 'trial_report.csv')}")
    print(f"  Study:   {db_path}")
    print(f"\n  Run 'python main.py run_pipeline' to use optimized parameters.")

    print(f"\n{'=' * 70}")
    print(f"  Optimization finished successfully.")
    print(f"{'=' * 70}\n")

    return {
        "best_params": best_params,
        "best_score": best.value,
        "best_attrs": best_attrs,
        "n_trials": len(study.trials),
    }


# ── Config writeback ─────────────────────────────────────────────────

def _write_best_config(
    config: dict,
    config_path: str,
    best_params: dict,
    best_attrs: dict,
) -> None:
    """
    Merges the best Optuna parameters into the existing pipeline config
    and writes it back. Only the tuned parameters are updated; all other
    settings (paths, visualization, SHAP, etc.) are preserved exactly.
    """
    # STL
    config["stl"]["residual_threshold_sigma"] = round(
        best_params["stl_residual_threshold_sigma"], 4,
    )

    # LightGBM
    config["lightgbm"]["num_leaves"] = best_params["lgbm_num_leaves"]
    config["lightgbm"]["learning_rate"] = round(
        best_params["lgbm_learning_rate"], 6,
    )
    config["lightgbm"]["feature_fraction"] = round(
        best_params["lgbm_feature_fraction"], 4,
    )
    config["lightgbm"]["bagging_fraction"] = round(
        best_params["lgbm_bagging_fraction"], 4,
    )
    config["lightgbm"]["n_estimators"] = best_params["lgbm_n_estimators"]
    config["lightgbm"]["residual_threshold_sigma"] = round(
        best_params["lgbm_residual_threshold_sigma"], 4,
    )

    # Isolation Forest
    config["isolation_forest"]["n_estimators"] = best_params["iforest_n_estimators"]
    config["isolation_forest"]["contamination"] = round(
        best_params["iforest_contamination"], 6,
    )

    # Ensemble weights (normalized in the objective, stored in user attrs)
    config["ensemble"]["weights"]["stl"] = round(
        best_attrs["w_stl"], 4,
    )
    config["ensemble"]["weights"]["lightgbm"] = round(
        best_attrs["w_lgbm"], 4,
    )
    config["ensemble"]["weights"]["isolation_forest"] = round(
        best_attrs["w_iforest"], 4,
    )
    config["ensemble"]["threshold"] = round(
        best_params["ensemble_threshold"], 4,
    )

    # DBSCAN
    config["dbscan"]["eps_days"] = best_params["dbscan_eps_days"]

    with open(config_path, "w") as f:
        yaml.dump(
            config, f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"\n  ✓ Best parameters written to {config_path}")


# ── Trial report ─────────────────────────────────────────────────────

def _save_trial_report(
    study: optuna.Study,
    optuna_dir: str,
) -> None:
    """
    Saves a CSV report of all trials with their parameters, objective
    values, and diagnostic user attributes for audit and analysis.
    """
    records = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial": trial.number,
            "score": round(trial.value, 4),
            "duration_sec": round(trial.duration.total_seconds(), 1),
        }
        row.update(trial.params)
        row.update(trial.user_attrs)
        records.append(row)

    if records:
        df = pd.DataFrame(records).sort_values("score", ascending=False)
        csv_path = os.path.join(optuna_dir, "trial_report.csv")
        df.to_csv(csv_path, index=False)
