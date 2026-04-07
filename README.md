# Cloud Anomaly Detection

An end-to-end cloud cost anomaly detection system that generates realistic synthetic AWS Cost and Usage Report (CUR) data with injected anomalies, then runs a multi-model machine learning pipeline to detect, classify, explain, and visualize those anomalies. Includes Optuna-based Bayesian hyperparameter optimization to automatically find the best model parameters. The entire workflow is driven from a single entry point with three subcommands.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Command Reference](#command-reference)
6. [Configuration](#configuration)
7. [Repository Structure](#repository-structure)
8. [Pipeline Architecture](#pipeline-architecture)
9. [Hyperparameter Optimization](#hyperparameter-optimization)
10. [Output Structure](#output-structure)
11. [Detection Models](#detection-models)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Visualizations](#visualizations)

---

## Overview

Cloud infrastructure costs fluctuate daily. Some fluctuations are normal (weekly traffic patterns, month-end batch jobs), while others signal real problems: a misconfigured auto-scaler doubling your EC2 spend overnight, a runaway query spiking database costs, or a slow resource leak gradually inflating bills over weeks.

This project tackles the problem in two phases:

**Phase 1 -- Synthetic Data Generation.** Produces a realistic multi-account, multi-service, multi-region AWS CUR dataset spanning configurable time periods. Three types of anomalies are injected at controlled rates: cost spikes (sudden single-day jumps), cascades (coordinated failures across multiple services within the same account), and drifts (gradual cost increases over days or weeks). Every anomalous row is labelled with ground-truth metadata so models can be evaluated against known answers.

**Phase 2 -- Anomaly Detection Pipeline.** Loads the generated dataset, engineers 35 time-series features, runs four detection models (STL decomposition, LightGBM regression, Isolation Forest, and a baseline percentage-change threshold), combines them into a weighted ensemble, clusters flagged anomalies into cascade events using DBSCAN, generates SHAP-based explanations for every flagged row, evaluates all models with standard classification metrics, and produces 15 diagnostic visualizations.

**Phase 3 -- Hyperparameter Optimization.** Uses Optuna's Tree-structured Parzen Estimator (TPE) to automatically search for the best model parameters across all pipeline components. Optimizes a composite objective (70% ensemble F1, 15% drift detection rate, 15% cascade detection rate) to prevent the optimizer from ignoring hard-to-detect anomaly types. Trials are pruned early via a median pruner, the study persists in SQLite for resumability, and the best parameters are written back to `pipeline_config.yaml`.

---

## Prerequisites

- Python 3.10 or later
- pip (included with Python)
- Operating system: Windows, macOS, or Linux

---

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone <repository-url>
cd north-anomaly-detection
```

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv-north-anomaly-detection
venv-north-anomaly-detection\Scripts\activate

# macOS / Linux
python3 -m venv venv-north-anomaly-detection
source venv-north-anomaly-detection/bin/activate
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

The requirements file installs the following packages: numpy, pandas, tqdm, pyyaml, rich, lightgbm, shap, statsmodels, matplotlib, seaborn, scikit-learn, and optuna. Exact version constraints are specified in `requirements.txt`.

---

## Usage

Everything runs through `main.py`. There are three subcommands.

### Step 1: Generate Synthetic Data

```bash
python main.py generate_data
```

This reads `config.yaml`, builds simulated AWS accounts with resource pools, plans an anomaly injection schedule, generates raw CUR line items month by month, aggregates them into a daily time-series with rolling statistics and anomaly labels, and writes all output files to `output/synthetic-data/`.

### Step 2: Run the Anomaly Detection Pipeline

```bash
python main.py run_pipeline
```

This reads `pipeline_config.yaml`, loads the synthetic data from `output/synthetic-data/`, runs all 18 pipeline steps (feature engineering through visualization), and writes results to `output/anomaly-detection-results/`.

### Step 3: Optimize Hyperparameters (Optional)

```bash
python main.py optimize
```

This runs Optuna Bayesian hyperparameter optimization over all tunable pipeline parameters (LightGBM tree structure, STL thresholds, Isolation Forest contamination, ensemble weights and threshold, DBSCAN eps). The best parameters are automatically written back to `pipeline_config.yaml`. After optimization, re-run `python main.py run_pipeline` to use the optimized parameters.

Steps 1 and 2 must be run in order. The pipeline requires the synthetic dataset to exist before it can operate. Step 3 is optional and can be run after step 1 to find the best parameters before the final pipeline run.

### Quick Start (Small Dataset)

To run a fast end-to-end test with minimal data:

```bash
python main.py generate_data --num_accounts 2 --num_months 6 --target_rows 10000
python main.py optimize --n_trials 10
python main.py run_pipeline
```

---

## Command Reference

### generate_synthetic_cur_data

Aliases: `generate_cur`, `generate_data`

Generates the synthetic AWS CUR dataset. All arguments are optional. When omitted, values are read from `config.yaml`. When provided, CLI arguments override the configuration file.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_accounts` | int | 5 | Number of AWS accounts to simulate. Each account receives a unique name, a 12-digit account ID, and its own resource pool. |
| `--num_months` | int | 12 | Number of months of daily data to generate, counting backwards from the current date. |
| `--target_rows` | int | 1000000 | Approximate total row count in the raw CUR file. The generator distributes line items across accounts, services, regions, and days to reach this target. |
| `--services` | str | (from config) | Comma-separated list of AWS service codes to include, for example `AmazonEC2,AmazonS3,AmazonRDS`. Overrides the list in `config.yaml`. |
| `--regions` | str | (from config) | Comma-separated list of AWS regions, for example `us-east-1,eu-west-1`. Overrides the list in `config.yaml`. |
| `--num_spike_anomalies` | int | 15 | Number of single-day cost spike anomalies to inject. Each spike multiplies the normal cost of a randomly selected (account, service, region, date) combination. |
| `--num_cascade_anomalies` | int | 8 | Number of cascade anomalies to inject. Each cascade selects an account and date, then spikes costs across 2 or more services in that account simultaneously. |
| `--num_drift_anomalies` | int | 5 | Number of drift anomalies to inject. Each drift selects an (account, service, region) and applies a gradual cost multiplier that increases linearly over a multi-day window. |
| `--seasonal_strength` | int | 3 | Intensity of the weekly seasonal cost pattern on a scale of 1 (minimal) to 5 (pronounced). Higher values create larger weekday-to-weekend cost differences. |
| `--spike_magnitude` | float | 3.0 | Multiplier applied to normal cost during spike anomalies. A value of 3.0 means the anomalous cost is approximately 3 times the normal amount. |
| `--output_dir` | str | ./output/synthetic-data | Filesystem directory where all generated CSV and summary files are saved. Created automatically if it does not exist. |
| `--seed` | int | 42 | Random seed for full reproducibility. Setting the same seed with the same parameters produces identical output. |

### run_pipeline

Aliases: `pipeline`, `detect`

Runs the ML anomaly detection pipeline. This subcommand takes no arguments. All configuration is read from `pipeline_config.yaml`.

### optimize_pipeline

Aliases: `optimize`, `tune`

Runs Optuna Bayesian hyperparameter optimization over all tunable pipeline parameters. The best parameters are written back to `pipeline_config.yaml` when the study completes. The study is persisted in a SQLite database under `output/anomaly-detection-results/optuna/` so it can be resumed across multiple invocations.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n_trials` | int | 50 | Maximum number of optimization trials. Each trial runs the full detection pipeline with a different parameter combination. |
| `--timeout` | int | (none) | Wall-clock time limit in minutes. The study stops when either `n_trials` or `timeout` is reached, whichever comes first. |

---

## Configuration

### config.yaml

Controls the synthetic data generation phase. Organized into three sections:

**generation** -- Defines the shape of the dataset: number of accounts, months, target row count, random seed, and the lists of AWS services and regions to simulate.

**anomalies** -- Controls the anomaly injection: counts for each anomaly type (spike, cascade, drift), seasonal effect strength, and the spike cost multiplier.

**output** -- Specifies the output directory for generated files.

### pipeline_config.yaml

Controls the anomaly detection pipeline. Organized into the following sections:

**paths** -- Input data directory and output directories for results, plots, and reports.

**preprocessing** -- Minimum history requirement (28 days) and the train/test split ratio (0.7, meaning 70% of the date range is used for training).

**baseline_threshold** -- List of percentage-change thresholds to evaluate: 0.30 (30%), 0.50 (50%), 0.75 (75%), 1.00 (100%).

**stl** -- Seasonal period (7 for weekly) and the residual z-score threshold for flagging anomalies. Tuned by Optuna.

**lightgbm** -- Full set of LightGBM hyperparameters: objective, metric, boosting type, tree structure, learning rate, regularization (feature and bagging fractions), training limits (with early stopping), and the residual z-score threshold. All tunable parameters are optimized by Optuna.

**isolation_forest** -- Number of estimators, expected contamination rate, and parallelism setting. Tuned by Optuna.

**ensemble** -- Per-model weights for the weighted average and the ensemble score threshold above which a row is flagged. Tuned by Optuna.

**dbscan** -- Parameters for cascade clustering: temporal proximity in days (eps_days) and minimum cluster size (min_samples). eps_days is tuned by Optuna.

**optuna** -- Hyperparameter optimization settings: maximum number of trials (`n_trials`, default 50) and optional wall-clock timeout in minutes (`timeout_minutes`, default null for no limit). These defaults are used when `--n_trials` or `--timeout` are not provided on the command line.

**shap** -- Maximum number of samples for SHAP computation (500) and number of top features to include in explanations (10).

**visualization** -- Plot resolution (DPI), figure dimensions for three layout types (wide, square, tall), and hex color codes for each anomaly type.

---

## Repository Structure

```
north-anomaly-detection/
    main.py                      Entry point for all operations
    config.yaml                  Synthetic data generation configuration
    pipeline_config.yaml         Anomaly detection pipeline configuration
    requirements.txt             Python package dependencies
    README.md                    This file
    .gitignore                   Git ignore rules

    src/
        engine.py                Orchestrator class with both pipeline methods

        synthetic_data_generation/
            account_builder.py   Builds simulated AWS accounts and resource ARN pools
            anomaly_planner.py   Plans the anomaly injection schedule and writes the anomaly log
            raw_cur_generator.py Generates raw CUR line items month by month with anomaly injection
            daily_aggregator.py  Aggregates raw CUR to daily time-series with rolling statistics
            summary_builder.py   Produces a human-readable dataset summary

        preprocessing/
            data_loader.py       Loads CSV files, parses dates, handles missing values
            feature_engineer.py  Computes 35 time-series features per group

        models/
            baseline_threshold.py  Percentage-change threshold detector
            stl_decomposition.py   STL seasonal-trend decomposition detector
            lightgbm_forecaster.py LightGBM cost prediction and residual-based detection
            isolation_forest.py    Isolation Forest unsupervised anomaly detector
            ensemble.py            Weighted ensemble combiner

        clustering/
            cascade_detector.py  DBSCAN-based cascade identification

        optimization/
            optuna_optimizer.py  Optuna Bayesian hyperparameter optimization module

        explainability/
            shap_explainer.py    SHAP value computation and anomaly explanation generation

        evaluation/
            metrics.py           Precision, recall, F1, ROC AUC, per-type detection rates

        visualizations/
            plot_generator.py    15 diagnostic plot functions and orchestrator

        utilities/
            consts_handler.py    YAML configuration section loaders
            general_utils.py     Shared logging and console output helper
            log_handler.py       Application logger setup
```

---

## Pipeline Architecture

The anomaly detection pipeline executes 18 sequential steps:

| Step | Operation | Description |
|---|---|---|
| 1 | Load configuration | Reads `pipeline_config.yaml` and creates output directories. |
| 2 | Load data | Reads `daily_aggregated.csv` and `anomaly_log.csv` from the synthetic data directory. Parses dates, casts types, fills missing anomaly labels. |
| 3 | Train/test split | Splits the dataset chronologically at the 70th percentile date. All data before the split is training; everything after is testing. Chronological splitting prevents future data leakage. |
| 4 | Feature engineering | Computes 31 numerical and 4 categorical features per (account, service, region) group. Features include cost lags (1, 2, 3, 7, 14, 28 days), rolling means and standard deviations (3, 7, 14, 28 day windows), day-over-day and week-over-week differences, cost-to-moving-average ratios, z-scores against 7-day and 28-day baselines, usage lags and rolling averages, per-unit costs, and calendar flags. The first 28 days of each group are dropped as a warm-up period because lag features are undefined. |
| 5 | Baseline threshold | Tests four percentage-change thresholds (30%, 50%, 75%, 100%) against the 7-day and 28-day rolling average. Evaluates precision, recall, and F1 for each. Analyzes false positive and false negative patterns by day-of-week, service, and anomaly type. |
| 6 | STL decomposition | Decomposes each group's cost time-series into trend, seasonal, and residual components using Seasonal-Trend decomposition by Loess with a 7-day period and robust fitting. Residual statistics are computed from the training portion only. Test rows where the residual z-score exceeds 2.5 sigma are flagged. |
| 7 | LightGBM training | Trains a gradient-boosted regression model to predict daily cost from the engineered features. Uses the last 20% of training dates as a validation set for early stopping. Computes training residual mean and standard deviation for anomaly thresholding. |
| 8 | LightGBM detection | Predicts costs on the test set, computes residuals (actual minus predicted), and flags rows where the residual z-score exceeds 2.5 sigma. |
| 9 | Isolation Forest | Fits an Isolation Forest with 200 trees on the training features. Scores the test set based on isolation path depth. Rows that are easy to isolate (short paths) receive high anomaly scores. The contamination parameter (0.03) sets the decision boundary. |
| 10 | Ensemble scoring | Normalizes each model's anomaly scores to the 0-to-1 range using min-max scaling, then computes a weighted average: 0.25 STL + 0.45 LightGBM + 0.30 Isolation Forest. Rows where the ensemble score exceeds 0.40 are flagged. |
| 11 | Cascade clustering | For each account, clusters ensemble-flagged anomalies using DBSCAN on scaled date ordinals only. Service diversity is checked post-clustering: clusters containing anomalous rows from two or more distinct services are labelled as predicted cascades. Using date-only features avoids pushing different services apart, which would prevent the cross-service clusters we are looking for. |
| 12 | SHAP values | Computes SHAP values for the LightGBM model using TreeExplainer on a subsample of up to 500 test rows. Produces per-feature contribution values for each explained row. |
| 13 | SHAP anomaly report | Generates a CSV report and a plain-text explanation file for every ensemble-flagged anomaly, identifying the top contributing features and their SHAP impact values. |
| 14 | Model evaluation | Computes precision, recall, F1, accuracy, ROC AUC, and per-anomaly-type detection rates for each model and the ensemble. Saves a side-by-side comparison table. |
| 15 | Save scored test set | Writes the full test DataFrame with all model scores, predictions, and ground-truth labels to CSV. |
| 16 | Generate visualizations | Produces 15 diagnostic PNG plots covering time-series overviews, seasonal patterns, model decompositions, predictions, comparisons, confusion matrices, false positive analysis, cascade clustering, SHAP summaries, SHAP waterfalls per anomaly type, ROC curves, and drift detection. |
| 17 | Save baseline results | Writes the threshold comparison table with per-threshold metrics and per-type detection rates to CSV. |
| 18 | Final summary | Prints runtime, dataset statistics, flagged anomaly counts, best model by F1, and lists all output files. |

---

## Hyperparameter Optimization

The optimization module uses Optuna to search for the best parameters across all pipeline components via Bayesian optimization.

### Search Spaces

| Component | Parameter | Range | Scale | Rationale |
|---|---|---|---|---|
| LightGBM | num_leaves | 15 -- 127 | linear | Controls tree complexity. Higher values capture service-specific patterns but risk overfitting to individual account noise. |
| LightGBM | learning_rate | 0.005 -- 0.2 | log | Step size shrinkage. Log-uniform because the effect is multiplicative. |
| LightGBM | feature_fraction | 0.5 -- 1.0 | linear | Column subsampling. Reduces correlation between trees and prevents over-reliance on dominant features like cost\_lag\_1d. |
| LightGBM | bagging_fraction | 0.5 -- 1.0 | linear | Row subsampling. Combined with feature_fraction, this is the primary regularization lever. |
| LightGBM | n_estimators | 200 -- 1500 | step=100 | Maximum boosting rounds. Early stopping typically halts before this ceiling. |
| LightGBM | residual_sigma | 0.5 -- 3.0 | linear | Residual z-score threshold. Lower values catch more drifts but increase false positives on normal Monday spikes. |
| STL | residual_sigma | 0.5 -- 3.0 | linear | Same concept applied to STL decomposition residuals. Critical for drift detection. |
| Isolation Forest | contamination | 0.003 -- 0.03 | log | Expected anomaly fraction. Must approximate reality; too high floods false positives. |
| Isolation Forest | n_estimators | 100 -- 500 | step=50 | Number of isolation trees. More trees give more stable scores. |
| Ensemble | weights (×3) | 0.1 -- 1.0 each | linear | Three raw weights normalized to sum to 1 (Dirichlet-like). Every convex combination is reachable. |
| Ensemble | threshold | 0.05 -- 0.40 | linear | Final decision boundary on the blended [0, 1] score. |
| DBSCAN | eps_days | 1 -- 5 | integer | Max temporal distance for cascade grouping. |

### Optimization Strategy

- **Sampler**: TPE (Tree-structured Parzen Estimator) with 15 random startup trials for exploration before Bayesian exploitation begins.
- **Pruner**: Median pruner with 10 startup trials. Reports intermediate F1 after STL (step 0) and LightGBM (step 1). Trials below the running median are killed early, saving up to 60% of compute.
- **Objective**: Composite score = 0.70 × ensemble F1 + 0.15 × drift detection rate + 0.15 × cascade detection rate. The penalty terms prevent the optimizer from ignoring hard-to-detect anomaly types in favor of easy spike recall.
- **Validation**: Strict chronological train/test split (same ratio as the main pipeline). No random cross-validation to avoid future data leakage.
- **Persistence**: Study is stored in SQLite (`output/anomaly-detection-results/optuna/study.db`). Supports resuming interrupted runs and incremental trial additions via `load_if_exists`.
- **Outputs**: Best parameters are merged back into `pipeline_config.yaml` (preserving all non-tuned settings). A trial report CSV is saved for audit.

---

## Output Structure

After running both commands, the output directory contains:

```
output/
    synthetic-data/
        raw_cur_data.csv              Raw CUR line items (one row per line item per day)
        daily_aggregated.csv          Daily aggregated time-series with anomaly labels
        anomaly_log.csv               Injected anomaly schedule with types, dates, and magnitudes
        dataset_summary.txt           Human-readable summary of the generated dataset

    anomaly-detection-results/
        scored_test_set.csv           Full test set with all model scores and predictions

        optuna/
            study.db                  SQLite-backed Optuna study (survives restarts)
            trial_report.csv          Per-trial parameters, scores, and diagnostics

        reports/
            model_comparison.csv      Side-by-side model performance metrics
            anomaly_report.csv        SHAP-explained anomaly details
            anomaly_explanations.txt  Plain-text explanations for each flagged anomaly
            baseline_thresholds.csv   Per-threshold precision, recall, F1, and detection rates

        plots/
            01_time_series_overview.png
            02_seasonal_patterns.png
            03_stl_decomposition.png
            04_lgbm_predictions.png
            05_model_comparison.png
            06_detection_by_type.png
            07_confusion_matrices.png
            08_false_positive_analysis.png
            09_cascade_clustering.png
            10_shap_summary.png
            11_shap_waterfall_spike.png
            12_shap_waterfall_cascade.png
            13_shap_waterfall_drift.png
            14_roc_curves.png
            15_drift_showcase.png
```

---

## Detection Models

### Baseline Threshold

A non-ML approach that flags any row where the absolute percentage change (relative to the 7-day or 28-day rolling average) exceeds a fixed threshold. Four thresholds are tested: 30%, 50%, 75%, and 100%. This establishes a performance floor and reveals the limitations of simple rule-based alerting.

### STL Decomposition

Decomposes each (account, service, region) time-series into three additive components: a long-term trend, a repeating weekly seasonal pattern, and a residual. The decomposition uses Loess smoothing with robust fitting to prevent outliers from contaminating the trend and seasonal estimates. Anomalies are identified as test rows where the residual deviates more than a configurable number of standard deviations from the training residual distribution. The residual sigma threshold is tuned by Optuna. This approach captures anomalies that persist after removing expected seasonal and trend behavior.

### LightGBM Regression

A gradient-boosted decision tree model trained to predict the expected daily cost from 35 engineered features. Trees are built sequentially, each correcting the errors of the previous ensemble. Early stopping on a chronological validation set prevents overfitting. Anomalies are identified as rows where the prediction residual (actual cost minus predicted cost) exceeds a configurable sigma threshold of the training residual distribution. All LightGBM hyperparameters (num_leaves, learning_rate, feature/bagging fractions, n_estimators, residual sigma) are tuned by Optuna. This approach captures complex multi-feature interactions that simpler models miss.

### Isolation Forest

An unsupervised algorithm that isolates data points by randomly partitioning the feature space. Anomalous points, having extreme or unusual feature values, require fewer random splits to isolate than normal points. The number of trees and the contamination parameter are tuned by Optuna. This approach provides a complementary perspective because it does not rely on prediction accuracy or time-series structure.

### Weighted Ensemble

Combines the three model scores (STL, LightGBM, Isolation Forest) into a single score. Each model's raw scores are first normalized to the 0-to-1 range using min-max scaling across the test set, then combined as a weighted average. The per-model weights and the ensemble score threshold are tuned by Optuna using a Dirichlet-like sampling strategy that ensures every convex weight combination is reachable. The ensemble reduces individual model blind spots: different models tend to produce different false positives, so combining them improves overall reliability.

---

## Evaluation Metrics

The pipeline evaluates each model using the following metrics:

**Precision** -- The fraction of flagged rows that are actual anomalies. High precision means few false alarms.

**Recall** -- The fraction of actual anomalies that were detected. High recall means few missed anomalies.

**F1 Score** -- The harmonic mean of precision and recall. Balances both metrics into a single number.

**Accuracy** -- The fraction of all rows classified correctly. Less informative for imbalanced datasets where anomalies are rare.

**ROC AUC** -- Area Under the Receiver Operating Characteristic curve. Measures the model's ability to rank anomalous rows higher than normal rows across all possible thresholds. A value of 1.0 indicates perfect ranking; 0.5 indicates random chance.

**Per-type detection rates** -- For each anomaly type (spike, cascade, drift), the fraction of actual anomalous rows of that type that the model correctly flagged. Reveals which models specialize in which anomaly categories.

---

## Visualizations

The pipeline produces the following 15 plots:

| Plot | Description |
|---|---|
| 01 Time Series Overview | Total daily cost across all accounts with ground-truth anomalies color-coded by type and the train/test split marked. |
| 02 Seasonal Patterns | Box plots of cost distribution by day-of-week and by month, revealing weekly and monthly seasonality. |
| 03 STL Decomposition | Four-panel decomposition (original, trend, seasonal, residual) for one representative group with anomalies highlighted in the residual panel. |
| 04 LightGBM Predictions | Actual versus predicted cost for one group, with residual bars and flagged anomalies highlighted. |
| 05 Model Comparison | Grouped bar chart comparing precision, recall, and F1 across all models. |
| 06 Detection by Type | Grouped bar chart showing each model's detection rate broken down by spike, cascade, and drift. |
| 07 Confusion Matrices | 2x2 heatmaps for STL, LightGBM, Isolation Forest, and Ensemble showing true/false positive/negative counts. |
| 08 False Positive Analysis | Ensemble false positives broken down by service and by day-of-week to identify systematic patterns. |
| 09 Cascade Clustering | Scatter plot of predicted cascades with date on the x-axis and service on the y-axis, colored by cluster ID. |
| 10 SHAP Summary | Beeswarm plot showing global feature importance and impact direction across all explained samples. |
| 11 SHAP Waterfall (Spike) | Feature contribution waterfall for one example spike anomaly. |
| 12 SHAP Waterfall (Cascade) | Feature contribution waterfall for one example cascade anomaly. |
| 13 SHAP Waterfall (Drift) | Feature contribution waterfall for one example drift anomaly. |
| 14 ROC Curves | ROC curves for all models with AUC values in the legend. |
| 15 Drift Showcase | Cost time-series for a drift-affected group with the 28-day rolling average overlaid to show the gradual cost shift. |