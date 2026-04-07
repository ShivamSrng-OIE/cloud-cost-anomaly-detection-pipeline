import os
import time
from typing import Optional, List

import numpy as np
import pandas as pd
import yaml

from src.utilities.general_utils import console_and_logger
from src.utilities.log_handler import LogHandler
from src.utilities.consts_handler import (
    GenerationConsts,
    AnomalyConsts,
    OutputConsts,
)
from src.synthetic_data_generation.account_builder import AccountBuilder
from src.synthetic_data_generation.anomaly_planner import AnomalyPlanner
from src.synthetic_data_generation.raw_cur_generator import RawCurGenerator
from src.synthetic_data_generation.daily_aggregator import DailyAggregator
from src.synthetic_data_generation.summary_builder import SummaryBuilder


class Engine:

    def __init__(self) -> None:
        self.__logger = LogHandler().get_logger()

    def __merge_config(
            self,
            generation_consts: dict,
            anomaly_consts: dict,
            output_consts: dict,
            num_accounts: Optional[int] = None,
            num_months: Optional[int] = None,
            services: Optional[str] = None,
            regions: Optional[str] = None,
            num_spike_anomalies: Optional[int] = None,
            num_cascade_anomalies: Optional[int] = None,
            num_drift_anomalies: Optional[int] = None,
            seasonal_strength: Optional[int] = None,
            spike_magnitude: Optional[float] = None,
            output_dir: Optional[str] = None,
            seed: Optional[int] = None,
            target_rows: Optional[int] = None,
        ) -> dict:
        """
        Merges the three config.yaml sections with any CLI overrides to produce
        a single flat configuration dict used by all downstream pipeline stages.

        Each parameter that is not None takes precedence over the corresponding
        value from configuration file. Services and regions are parsed from
        comma-separated strings when passed via CLI, otherwise taken as-is from
        the YAML list.

        generation_consts (dict): Parsed 'generation' section from config.yaml.
        anomaly_consts (dict): Parsed 'anomalies' section from config.yaml.
        output_consts (dict): Parsed 'output' section from config.yaml.
        num_accounts (int, optional): Override for the number of AWS accounts.
        num_months (int, optional): Override for how many months of data to generate.
        services (str, optional): Comma-separated service codes, e.g. "AmazonEC2,AmazonS3".
        regions (str, optional): Comma-separated AWS regions.
        num_spike_anomalies (int, optional): Override for spike anomaly count.
        num_cascade_anomalies (int, optional): Override for cascade anomaly count.
        num_drift_anomalies (int, optional): Override for drift anomaly count.
        seasonal_strength (int, optional): Seasonal effect intensity on a 1-5 scale.
        spike_magnitude (float, optional): Cost multiplier applied to spike anomalies.
        output_dir (str, optional): Filesystem directory for all output files.
        seed (int, optional): Random seed for full reproducibility.
        target_rows (int, optional): Approximate row count target for the raw CUR.

        Returns a flat dict with every pipeline setting resolved to a concrete value.
        """

        def _first(*vals):
            for v in vals:
                if v is not None:
                    return v
            return None

        if services:
            svc_list = [s.strip() for s in services.split(",")]
        else:
            svc_list = generation_consts.get("services", [])

        if regions:
            reg_list = [r.strip() for r in regions.split(",")]
        else:
            reg_list = generation_consts.get("regions", [])

        return {
            "num_accounts":          _first(num_accounts, generation_consts.get("num_accounts"), 5),
            "num_months":            _first(num_months, generation_consts.get("num_months"), 12),
            "target_rows":           _first(target_rows, generation_consts.get("target_rows"), 1_000_000),
            "seed":                  _first(seed, generation_consts.get("seed"), 42),
            "services":              svc_list,
            "regions":               reg_list,
            "num_spike_anomalies":   _first(num_spike_anomalies, anomaly_consts.get("num_spike_anomalies"), 15),
            "num_cascade_anomalies": _first(num_cascade_anomalies, anomaly_consts.get("num_cascade_anomalies"), 8),
            "num_drift_anomalies":   _first(num_drift_anomalies, anomaly_consts.get("num_drift_anomalies"), 5),
            "seasonal_strength":     _first(seasonal_strength, anomaly_consts.get("seasonal_strength"), 3),
            "spike_magnitude":       _first(spike_magnitude, anomaly_consts.get("spike_magnitude"), 3.0),
            "cascade_min_services":  anomaly_consts.get("cascade_min_services", 3),
            "output_dir":            _first(output_dir, output_consts.get("output_dir"), "./output/synthetic-data"),
        }

    def generate_synthetic_cur_data(
            self,
            num_accounts: Optional[int] = None,
            num_months: Optional[int] = None,
            services: Optional[str] = None,
            regions: Optional[str] = None,
            num_spike_anomalies: Optional[int] = None,
            num_cascade_anomalies: Optional[int] = None,
            num_drift_anomalies: Optional[int] = None,
            seasonal_strength: Optional[int] = None,
            spike_magnitude: Optional[float] = None,
            output_dir: Optional[str] = None,
            seed: Optional[int] = None,
            target_rows: Optional[int] = None,
        ) -> None:
        """
        Runs the full 6-stage synthetic CUR generation pipeline end to end.

        The stages execute in order: (1) merge configuration, (2) build account
        metadata and resource ARN pools, (3) plan anomaly injection schedule,
        (4) generate raw CUR line items month by month, (5) aggregate to daily
        time-series with rolling stats and anomaly labels, (6) write the anomaly
        log CSV, and (7) produce a human-readable dataset summary.

        All keyword arguments are optional CLI overrides that take precedence
        over values in config.yaml. See __merge_config for full descriptions.
        Outputs are written to the resolved output_dir.
        """
        start_time = time.time()

        # Configuration
        console_and_logger(self.__logger, "-" * 60)
        console_and_logger(self.__logger, "Cloud - Synthetic AWS CUR Dataset Generator")
        console_and_logger(self.__logger, "-" * 60)
        console_and_logger(self.__logger, "")

        generation_consts = GenerationConsts(logger=self.__logger).get_config()
        anomaly_consts = AnomalyConsts(logger=self.__logger).get_config()
        output_consts = OutputConsts(logger=self.__logger).get_config()

        config = self.__merge_config(
            generation_consts=generation_consts,
            anomaly_consts=anomaly_consts,
            output_consts=output_consts,
            num_accounts=num_accounts,
            num_months=num_months,
            services=services,
            regions=regions,
            num_spike_anomalies=num_spike_anomalies,
            num_cascade_anomalies=num_cascade_anomalies,
            num_drift_anomalies=num_drift_anomalies,
            seasonal_strength=seasonal_strength,
            spike_magnitude=spike_magnitude,
            output_dir=output_dir,
            seed=seed,
            target_rows=target_rows,
        )

        console_and_logger(self.__logger, f"Accounts:   {config['num_accounts']}")
        console_and_logger(self.__logger, f"Months:     {config['num_months']}")
        console_and_logger(self.__logger, f"Services:   {', '.join(config['services'])}")
        console_and_logger(self.__logger, f"Regions:    {', '.join(config['regions'])}")
        console_and_logger(self.__logger, f"Target:     ~{config['target_rows']:,} rows")
        console_and_logger(self.__logger, f"Seed:       {config['seed']}")
        console_and_logger(self.__logger, f"Output:     {config['output_dir']}")
        console_and_logger(self.__logger, f"Anomalies:  {config['num_spike_anomalies']} spikes, "
                           f"{config['num_cascade_anomalies']} cascades, "
                           f"{config['num_drift_anomalies']} drifts")
        console_and_logger(self.__logger, "")

        out_dir = config["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        np.random.seed(config["seed"])

        # Accounts & resource pool
        console_and_logger(self.__logger, "-" * 60)
        console_and_logger(self.__logger, "[1/6] Setting up accounts and resource pools")
        console_and_logger(self.__logger, "-" * 60)

        account_builder = AccountBuilder(
            logger=self.__logger,
            generation_config=config,
        )
        accounts = account_builder.build_accounts()
        resource_pool = account_builder.build_resource_pool(accounts=accounts)

        console_and_logger(self.__logger, "")

        # Anomaly planning
        console_and_logger(self.__logger, "-" * 60)
        console_and_logger(self.__logger, "[2/6] Planning anomaly injection schedule")
        console_and_logger(self.__logger, "-" * 60)

        anomaly_planner = AnomalyPlanner(
            logger=self.__logger,
            generation_config=config,
        )
        anomaly_plan = anomaly_planner.plan_anomalies(accounts=accounts)

        console_and_logger(self.__logger, "")

        # Raw CUR generation
        console_and_logger(self.__logger, "-" * 60)
        console_and_logger(self.__logger, "[3/6] Generating raw CUR data (chunked by month)")
        console_and_logger(self.__logger, "-" * 60)

        raw_path = os.path.join(out_dir, "raw_cur_data.csv")
        raw_cur_generator = RawCurGenerator(
            logger=self.__logger,
            generation_config=config,
        )
        raw_cur_generator.generate(
            output_path=raw_path,
            accounts=accounts,
            resource_pool=resource_pool,
            anomaly_lookup=anomaly_plan["lookup"],
        )

        console_and_logger(self.__logger, "")

        # Daily aggregation
        console_and_logger(self.__logger, "-" * 60)
        console_and_logger(self.__logger, "[4/6] Building daily aggregated time-series")
        console_and_logger(self.__logger, "-" * 60)

        agg_path = os.path.join(out_dir, "daily_aggregated.csv")
        daily_aggregator = DailyAggregator(
            logger=self.__logger,
        )
        daily_aggregator.aggregate(
            raw_path=raw_path,
            output_path=agg_path,
            accounts=accounts,
            anomaly_lookup=anomaly_plan["lookup"],
        )

        console_and_logger(self.__logger, "")

        # Anomaly log
        console_and_logger(self.__logger, "-" * 60)
        console_and_logger(self.__logger, "[5/6] Writing anomaly log")
        console_and_logger(self.__logger, "-" * 60)

        anomaly_planner.write_anomaly_log(
            anomaly_plan=anomaly_plan,
            output_path=os.path.join(out_dir, "anomaly_log.csv"),
        )

        console_and_logger(self.__logger, "")

        # Summary
        console_and_logger(self.__logger, "-" * 60)
        console_and_logger(self.__logger, "[6/6] Generating dataset summary")
        console_and_logger(self.__logger, "-" * 60)

        summary_builder = SummaryBuilder(
            logger=self.__logger,
        )
        summary_builder.generate(
            raw_path=raw_path,
            agg_path=agg_path,
            anomaly_plan=anomaly_plan,
            accounts=accounts,
            config=config,
            output_path=os.path.join(out_dir, "dataset_summary.txt"),
        )

        elapsed = time.time() - start_time
        console_and_logger(self.__logger, "")
        console_and_logger(self.__logger, "-" * 60)
        console_and_logger(
            self.__logger,
            f"Pipeline completed in {elapsed:.1f}s.  Output: {out_dir}/",
        )
        console_and_logger(self.__logger, "-" * 60)

    @staticmethod
    def __pipeline_header(step_num: int, title: str):
        print(f"\n{'='*70}")
        print(f"  Step {step_num}: {title}")
        print(f"{'='*70}\n")

    def run_pipeline(self) -> None:
        """
        Runs the full ML anomaly detection pipeline end to end: loads
        the synthetic CUR data, engineers features, runs five detection
        models (baseline threshold, STL decomposition, LightGBM, Isolation
        Forest, weighted ensemble), performs cascade clustering, generates
        SHAP explanations, evaluates all models, and produces 15 diagnostic
        plots.

        Configuration is read from pipeline_config.yaml at the project root.
        """
        from src.preprocessing.data_loader import load_data, split_train_test
        from src.preprocessing.feature_engineer import (
            engineer_features, get_feature_columns,
        )
        from src.models.baseline_threshold import (
            run_threshold_detection, analyze_threshold_failures,
        )
        from src.models.stl_decomposition import run_stl_detection
        from src.models.lightgbm_forecaster import (
            train_lgbm_model, predict_and_detect,
        )
        from src.models.isolation_forest import run_isolation_forest
        from src.models.ensemble import run_ensemble
        from src.clustering.cascade_detector import detect_cascades
        from src.explainability.shap_explainer import (
            compute_shap_values, generate_anomaly_report,
        )
        from src.evaluation.metrics import generate_comparison_report
        from src.visualizations.plot_generator import generate_all_plots

        _h = self.__pipeline_header
        t_start = time.time()

        # ── Step 1: Load configuration ───────────────────────────────
        _h(1, "Load Configuration")

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "pipeline_config.yaml",
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        paths = config["paths"]
        data_dir = paths["data_dir"]
        results_dir = paths["results_dir"]
        plots_dir = paths["plots_dir"]
        reports_dir = paths["reports_dir"]

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        print(f"  Data dir:    {os.path.abspath(data_dir)}")
        print(f"  Results dir: {os.path.abspath(results_dir)}")
        print(f"  Plots dir:   {os.path.abspath(plots_dir)}")
        print(f"  Reports dir: {os.path.abspath(reports_dir)}")

        # ── Step 2: Load data ────────────────────────────────────────
        _h(2, "Load Data")

        daily_df, anomaly_log = load_data(data_dir)

        # ── Step 3: Train/test split ─────────────────────────────────
        _h(3, "Train/Test Split")

        train_ratio = config["preprocessing"]["train_split_ratio"]
        train_df, test_df, split_date = split_train_test(daily_df, train_ratio)

        print(f"  Split date:  {split_date}")
        print(f"  Train rows:  {len(train_df):,}")
        print(f"  Test rows:   {len(test_df):,}")
        print(f"  Train anomaly rate: "
              f"{train_df['is_anomaly'].mean():.4%}")
        print(f"  Test anomaly rate:  "
              f"{test_df['is_anomaly'].mean():.4%}")

        # ── Step 4: Feature engineering ──────────────────────────────
        _h(4, "Feature Engineering")

        train_df = engineer_features(train_df)
        test_df = engineer_features(test_df)

        numerical_features, categorical_features = get_feature_columns()

        print(f"\n  Train rows after warm-up drop: {len(train_df):,}")
        print(f"  Test rows after warm-up drop:  {len(test_df):,}")

        # ── Step 5: Baseline threshold detection ─────────────────────
        _h(5, "Baseline Threshold Detection")

        thresholds = config["baseline_threshold"]["thresholds"]
        threshold_results = run_threshold_detection(test_df, thresholds)

        best_thresh = max(
            threshold_results, key=lambda t: threshold_results[t]["f1"],
        )
        print(f"\n  Best baseline threshold: {best_thresh:.0%} "
              f"(F1={threshold_results[best_thresh]['f1']:.4f})")

        print(f"\n  Failure analysis for best threshold ({best_thresh:.0%}):")
        fp_df, fn_df = analyze_threshold_failures(test_df, best_thresh)

        # ── Step 6: STL decomposition ────────────────────────────────
        _h(6, "STL Decomposition")

        stl_config = config["stl"]
        test_df = run_stl_detection(
            train_df, test_df,
            period=stl_config["period"],
            residual_threshold_sigma=stl_config["residual_threshold_sigma"],
        )

        # ── Step 7: LightGBM training ────────────────────────────────
        _h(7, "LightGBM Training")

        lgbm_config = config["lightgbm"]
        lgbm_model, val_preds, residual_stats = train_lgbm_model(
            train_df,
            feature_cols=numerical_features,
            categorical_cols=categorical_features,
            target_col="daily_cost",
            params=lgbm_config,
        )

        # ── Step 8: LightGBM prediction & detection ─────────────────
        _h(8, "LightGBM Prediction & Anomaly Detection")

        test_df = predict_and_detect(
            lgbm_model, test_df,
            feature_cols=numerical_features,
            categorical_cols=categorical_features,
            train_residual_stats=residual_stats,
            residual_threshold_sigma=lgbm_config["residual_threshold_sigma"],
        )

        # ── Step 9: Isolation Forest ─────────────────────────────────
        _h(9, "Isolation Forest")

        iforest_config = config["isolation_forest"]
        test_df, label_encoders = run_isolation_forest(
            train_df, test_df,
            feature_cols=numerical_features,
            categorical_cols=categorical_features,
            params=iforest_config,
        )

        # ── Step 10: Ensemble ────────────────────────────────────────
        _h(10, "Ensemble Scoring")

        ensemble_config = config["ensemble"]
        test_df = run_ensemble(
            test_df,
            weights=ensemble_config["weights"],
            threshold=ensemble_config["threshold"],
        )

        # ── Step 11: Cascade clustering ──────────────────────────────
        _h(11, "Cascade Clustering (DBSCAN)")

        dbscan_config = config["dbscan"]
        test_df = detect_cascades(
            test_df,
            eps_days=dbscan_config["eps_days"],
            min_samples=dbscan_config["min_samples"],
        )

        # ── Step 12: SHAP values ─────────────────────────────────────
        _h(12, "SHAP Explainability")

        shap_config = config["shap"]
        shap_values, shap_sample_df, shap_explainer = compute_shap_values(
            lgbm_model, test_df,
            feature_cols=numerical_features,
            categorical_cols=categorical_features,
            max_samples=shap_config["max_samples"],
        )

        # ── Step 13: SHAP anomaly report ─────────────────────────────
        _h(13, "SHAP Anomaly Report")

        anomaly_report_df = generate_anomaly_report(
            lgbm_model, test_df,
            feature_cols=numerical_features,
            categorical_cols=categorical_features,
            top_features=shap_config["top_features"],
            reports_dir=reports_dir,
        )

        # ── Step 14: Model evaluation ────────────────────────────────
        _h(14, "Model Evaluation")

        comparison_df = generate_comparison_report(test_df, reports_dir)

        # ── Step 15: Save scored test set ────────────────────────────
        _h(15, "Save Scored Test Set")

        scored_path = os.path.join(results_dir, "scored_test_set.csv")
        test_df.to_csv(scored_path, index=False)
        print(f"  Scored test set saved: {scored_path}")
        print(f"  Rows: {len(test_df):,}")
        print(f"  Columns: {len(test_df.columns)}")

        # ── Step 16: Generate visualizations ─────────────────────────
        _h(16, "Generating Visualizations")

        viz_config = config["visualization"]
        plot_config = {
            "plots_dir": plots_dir,
            "dpi": viz_config["dpi"],
            "figure_size_wide": viz_config["figure_size_wide"],
            "figure_size_square": viz_config["figure_size_square"],
            "figure_size_tall": viz_config["figure_size_tall"],
            "colors": viz_config["colors"],
        }

        generate_all_plots(
            daily_df=daily_df,
            test_df=test_df,
            split_date=split_date,
            comparison_df=comparison_df,
            shap_values=shap_values,
            shap_sample_df=shap_sample_df,
            lgbm_model=lgbm_model,
            feature_cols=numerical_features,
            categorical_cols=categorical_features,
            config=plot_config,
        )

        # ── Step 17: Save baseline threshold results ─────────────────
        _h(17, "Save Baseline Threshold Results")

        threshold_rows = []
        for thresh, res in threshold_results.items():
            row = {
                "threshold": thresh,
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
                "tp": res["tp"], "fp": res["fp"],
                "fn": res["fn"], "tn": res["tn"],
            }
            for atype in ["spike", "cascade", "drift"]:
                row[f"{atype}_detection_rate"] = res["type_rates"][atype]["rate"]
            threshold_rows.append(row)

        thresh_df = pd.DataFrame(threshold_rows)
        thresh_path = os.path.join(reports_dir, "baseline_thresholds.csv")
        thresh_df.to_csv(thresh_path, index=False)
        print(f"  Baseline threshold results saved: {thresh_path}")

        # ── Step 18: Final summary ───────────────────────────────────
        _h(18, "Pipeline Complete")

        elapsed = time.time() - t_start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60

        print(f"  Total runtime: {minutes}m {seconds:.1f}s")
        print(f"  Test set size: {len(test_df):,} rows")
        print(f"  Ground truth anomalies in test: "
              f"{int(test_df['is_anomaly'].sum()):,}")
        print(f"  Ensemble anomalies flagged: "
              f"{int(test_df['ensemble_is_anomaly'].sum()):,}")

        if len(comparison_df) > 0:
            best_model = comparison_df.loc[comparison_df["F1"].idxmax()]
            print(f"\n  Best model by F1: {best_model['Model']} "
                  f"(F1={best_model['F1']:.4f})")

        print(f"\n  Output files:")
        print(f"    {os.path.abspath(scored_path)}")
        print(f"    {os.path.abspath(reports_dir)}/")

        report_files = [f for f in os.listdir(reports_dir) if f.endswith(".csv")]
        for f in sorted(report_files):
            print(f"      {f}")

        plot_files = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
        print(f"    {os.path.abspath(plots_dir)}/")
        for f in sorted(plot_files):
            print(f"      {f}")

        print(f"\n{'='*70}")
        print(f"  Pipeline finished successfully.")
        print(f"{'='*70}\n")

    def optimize_pipeline(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Runs Optuna hyperparameter optimization over the anomaly detection
        pipeline. Best parameters are written back to pipeline_config.yaml.

        n_trials (int, optional): Max optimization trials (default from config).
        timeout (int, optional): Wall-clock limit in minutes.
        """
        from src.optimization.optuna_optimizer import run_optimization

        # Load defaults from pipeline_config.yaml if not overridden via CLI
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "pipeline_config.yaml",
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        optuna_config = config.get("optuna", {})
        trials = n_trials if n_trials is not None else optuna_config.get("n_trials", 50)
        timeout_min = timeout if timeout is not None else optuna_config.get("timeout_minutes")

        run_optimization(n_trials=trials, timeout_minutes=timeout_min)

