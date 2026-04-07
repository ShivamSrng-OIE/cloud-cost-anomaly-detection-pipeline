import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def _setup_style(colors: dict):
    """Sets consistent plotting style for all figures."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def _save_fig(fig, plots_dir: str, filename: str, dpi: int):
    """Saves figure and closes it to free memory."""
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      Saved: {filename}")


# ── Plot 1: Time Series Overview ─────────────────────────────────────

def plot_time_series_overview(
    daily_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_date,
    config: dict,
):
    """
    Shows total daily cost across all accounts with ground-truth
    anomalies highlighted and the train/test split marked. Anomalies
    are color-coded by type (spike, cascade, drift).

    daily_df (pd.DataFrame): Full dataset (train + test) before feature
        engineering.
    test_df (pd.DataFrame): Unused here but keeps signature consistent.
    split_date: The date where training ends and testing begins.
    config (dict): Visualization config with plots_dir, dpi, colors, etc.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    # Aggregate total daily cost
    daily_total = daily_df.groupby("date").agg(
        total_cost=("daily_cost", "sum"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=config.get("figure_size_wide", [16, 6]))

    ax.plot(daily_total["date"], daily_total["total_cost"],
            color="#2c3e50", linewidth=0.8, alpha=0.9, label="Total daily cost")

    # Highlight anomalies by type
    anomaly_rows = daily_df[daily_df["is_anomaly"] == 1]
    for atype, color in [("spike", colors["spike"]),
                          ("cascade", colors["cascade"]),
                          ("drift", colors["drift"])]:
        type_rows = anomaly_rows[anomaly_rows["anomaly_type"] == atype]
        if len(type_rows) > 0:
            type_daily = type_rows.groupby("date")["daily_cost"].sum()
            ax.scatter(type_daily.index, type_daily.values,
                       color=color, s=20, alpha=0.7, label=f"{atype} anomaly",
                       zorder=3)

    # Train/test split line
    ax.axvline(x=pd.Timestamp(split_date), color=colors["threshold"],
               linestyle="--", linewidth=1.5, label="Train/Test split")

    ax.set_title("Daily Cloud Cost with Ground-Truth Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Daily Cost ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    _save_fig(fig, plots_dir, "01_time_series_overview.png", dpi)


# ── Plot 2: Seasonal Patterns ────────────────────────────────────────

def plot_seasonal_patterns(
    daily_df: pd.DataFrame,
    config: dict,
):
    """
    Shows cost distribution by day-of-week and by month using box plots
    to reveal weekly and monthly seasonality in cloud spending.

    daily_df (pd.DataFrame): Full dataset.
    config (dict): Visualization config.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    fig, axes = plt.subplots(1, 2, figsize=config.get("figure_size_wide", [16, 6]))

    # Day of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    sns.boxplot(data=daily_df, x="day_name", y="daily_cost",
                order=day_order, ax=axes[0], fliersize=1,
                palette="Blues_d")
    axes[0].set_title("Cost by Day of Week")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Daily Cost ($)")
    axes[0].tick_params(axis="x", rotation=45)

    # Month
    sns.boxplot(data=daily_df, x="month", y="daily_cost",
                ax=axes[1], fliersize=1, palette="Greens_d")
    axes[1].set_title("Cost by Month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Daily Cost ($)")

    fig.suptitle("Seasonal Patterns in Cloud Cost", fontsize=14, y=1.02)
    fig.tight_layout()

    _save_fig(fig, plots_dir, "02_seasonal_patterns.png", dpi)


# ── Plot 3: STL Decomposition ────────────────────────────────────────

def plot_stl_decomposition(
    daily_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
):
    """
    Visualizes STL decomposition results for one representative
    account-service-region group: original series, trend, seasonal,
    and residual components stacked vertically, with anomalies
    highlighted in the residual panel.

    daily_df (pd.DataFrame): Full dataset (to pick a group with anomalies).
    test_df (pd.DataFrame): Test data with stl_* columns.
    config (dict): Visualization config.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    # Pick a group that has STL anomalies
    anom_test = test_df[test_df["stl_is_anomaly"] == 1]
    if len(anom_test) == 0:
        print("      Skipping STL plot: no STL anomalies found.")
        return

    sample_row = anom_test.iloc[0]
    acct = sample_row["account_id"]
    svc = sample_row["service"]
    rgn = sample_row["region"]

    # Get this group's test data
    mask = (
        (test_df["account_id"] == acct)
        & (test_df["service"] == svc)
        & (test_df["region"] == rgn)
    )
    group = test_df[mask].sort_values("date").copy()

    if len(group) < 10:
        print("      Skipping STL plot: selected group too small.")
        return

    fig, axes = plt.subplots(4, 1, figsize=config.get("figure_size_tall", [12, 16]),
                              sharex=True)

    # Original
    axes[0].plot(group["date"], group["daily_cost"], color="#2c3e50", linewidth=1)
    axes[0].set_title(f"STL Decomposition: {svc} / {rgn} / Account {acct}")
    axes[0].set_ylabel("Cost ($)")

    # Trend
    axes[1].plot(group["date"], group["stl_trend"], color="#3498db", linewidth=1)
    axes[1].set_ylabel("Trend")

    # Seasonal
    axes[2].plot(group["date"], group["stl_seasonal"], color="#27ae60", linewidth=1)
    axes[2].set_ylabel("Seasonal")

    # Residual with anomaly highlights
    axes[3].plot(group["date"], group["stl_residual"], color="#7f8c8d", linewidth=1)
    anom_mask = group["stl_is_anomaly"] == 1
    if anom_mask.any():
        axes[3].scatter(group.loc[anom_mask, "date"],
                        group.loc[anom_mask, "stl_residual"],
                        color=colors["spike"], s=30, zorder=3,
                        label="STL anomaly")
        axes[3].legend(fontsize=9)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")

    fig.tight_layout()
    _save_fig(fig, plots_dir, "03_stl_decomposition.png", dpi)


# ── Plot 4: LightGBM Predictions ─────────────────────────────────────

def plot_lgbm_predictions(
    test_df: pd.DataFrame,
    config: dict,
):
    """
    Shows actual vs. LightGBM-predicted cost for one representative
    group, with residual anomalies highlighted. Helps visualize how
    well the model captures normal cost patterns and where it flags
    deviations.

    test_df (pd.DataFrame): Test data with lgbm_* columns.
    config (dict): Visualization config.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    if "lgbm_predicted_cost" not in test_df.columns:
        print("      Skipping LightGBM plot: no predictions found.")
        return

    # Pick a group with LightGBM anomalies
    anom_test = test_df[test_df["lgbm_is_anomaly"] == 1]
    if len(anom_test) == 0:
        sample_row = test_df.iloc[0]
    else:
        sample_row = anom_test.iloc[0]

    acct = sample_row["account_id"]
    svc = sample_row["service"]
    rgn = sample_row["region"]

    mask = (
        (test_df["account_id"] == acct)
        & (test_df["service"] == svc)
        & (test_df["region"] == rgn)
    )
    group = test_df[mask].sort_values("date").copy()

    fig, axes = plt.subplots(2, 1, figsize=config.get("figure_size_wide", [16, 6]),
                              sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Actual vs predicted
    axes[0].plot(group["date"], group["daily_cost"],
                 color="#2c3e50", linewidth=1, label="Actual cost")
    axes[0].plot(group["date"], group["lgbm_predicted_cost"],
                 color="#3498db", linewidth=1, linestyle="--",
                 label="LightGBM predicted")

    anom_mask = group["lgbm_is_anomaly"] == 1
    if anom_mask.any():
        axes[0].scatter(group.loc[anom_mask, "date"],
                        group.loc[anom_mask, "daily_cost"],
                        color=colors["spike"], s=30, zorder=3,
                        label="Flagged anomaly")
    axes[0].set_title(f"LightGBM: Actual vs Predicted — {svc} / {rgn} / Account {acct}")
    axes[0].set_ylabel("Cost ($)")
    axes[0].legend(fontsize=9)

    # Residual
    axes[1].bar(group["date"], group["lgbm_residual"],
                color="#95a5a6", alpha=0.7, width=1)
    if anom_mask.any():
        axes[1].bar(group.loc[anom_mask, "date"],
                    group.loc[anom_mask, "lgbm_residual"],
                    color=colors["spike"], alpha=0.9, width=1)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Date")
    axes[1].axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    _save_fig(fig, plots_dir, "04_lgbm_predictions.png", dpi)


# ── Plot 5: Model Comparison Bars ─────────────────────────────────────

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    config: dict,
):
    """
    Grouped bar chart comparing precision, recall, and F1 across all
    models side by side for quick visual comparison.

    comparison_df (pd.DataFrame): Output of generate_comparison_report
        with Model, Precision, Recall, F1 columns.
    config (dict): Visualization config.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    fig, ax = plt.subplots(figsize=config.get("figure_size_square", [10, 8]))

    models = comparison_df["Model"].values
    x = np.arange(len(models))
    width = 0.25

    bars_p = ax.bar(x - width, comparison_df["Precision"], width,
                     label="Precision", color="#3498db", alpha=0.85)
    bars_r = ax.bar(x, comparison_df["Recall"], width,
                     label="Recall", color="#e74c3c", alpha=0.85)
    bars_f = ax.bar(x + width, comparison_df["F1"], width,
                     label="F1", color="#2ecc71", alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Precision, Recall, F1")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, plots_dir, "05_model_comparison.png", dpi)


# ── Plot 6: Detection by Anomaly Type ────────────────────────────────

def plot_detection_by_type(
    comparison_df: pd.DataFrame,
    config: dict,
):
    """
    Grouped bar chart showing each model's detection rate broken down
    by anomaly type (spike, cascade, drift).

    comparison_df (pd.DataFrame): Comparison report with Spike_Detection,
        Cascade_Detection, Drift_Detection columns.
    config (dict): Visualization config.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    fig, ax = plt.subplots(figsize=config.get("figure_size_square", [10, 8]))

    models = comparison_df["Model"].values
    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, comparison_df["Spike_Detection"], width,
           label="Spike", color=colors["spike"], alpha=0.85)
    ax.bar(x, comparison_df["Cascade_Detection"], width,
           label="Cascade", color=colors["cascade"], alpha=0.85)
    ax.bar(x + width, comparison_df["Drift_Detection"], width,
           label="Drift", color=colors["drift"], alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate by Anomaly Type")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save_fig(fig, plots_dir, "06_detection_by_type.png", dpi)


# ── Plot 7: Confusion Matrices ───────────────────────────────────────

def plot_confusion_matrices(
    test_df: pd.DataFrame,
    config: dict,
):
    """
    Displays 2x2 confusion matrix heatmaps for each model (STL, LightGBM,
    Isolation Forest, Ensemble) in a 2x2 subplot grid.

    test_df (pd.DataFrame): Test data with all model prediction columns.
    config (dict): Visualization config.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    y_true = test_df["is_anomaly"].values

    model_preds = [
        ("STL", "stl_is_anomaly"),
        ("LightGBM", "lgbm_is_anomaly"),
        ("Isolation Forest", "iforest_is_anomaly"),
        ("Ensemble", "ensemble_is_anomaly"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=config.get("figure_size_square", [10, 8]))
    axes = axes.flatten()

    for i, (name, pred_col) in enumerate(model_preds):
        if pred_col not in test_df.columns:
            axes[i].set_visible(False)
            continue

        y_pred = test_df[pred_col].values
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                    ax=axes[i], cbar=False,
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"])
        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "07_confusion_matrices.png", dpi)


# ── Plot 8: False Positive Analysis ──────────────────────────────────

def plot_false_positive_analysis(
    test_df: pd.DataFrame,
    config: dict,
):
    """
    Analyzes false positives from the ensemble model by service and
    day-of-week. Two subplots: bar chart of FPs by service (top 10)
    and bar chart of FPs by day of week.

    test_df (pd.DataFrame): Test data with ensemble_is_anomaly column.
    config (dict): Visualization config.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    y_true = test_df["is_anomaly"].values
    y_pred = test_df["ensemble_is_anomaly"].values

    fp_mask = (y_pred == 1) & (y_true == 0)
    fp_df = test_df[fp_mask]

    if len(fp_df) == 0:
        print("      Skipping FP analysis: no false positives found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=config.get("figure_size_wide", [16, 6]))

    # FPs by service
    fp_by_svc = fp_df["service"].value_counts().head(10)
    fp_by_svc.plot(kind="barh", ax=axes[0], color="#e74c3c", alpha=0.8)
    axes[0].set_title("False Positives by Service (Top 10)")
    axes[0].set_xlabel("Count")
    axes[0].invert_yaxis()

    # FPs by day of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    fp_by_day = fp_df["day_name"].value_counts().reindex(day_order, fill_value=0)
    fp_by_day.plot(kind="bar", ax=axes[1], color="#e67e22", alpha=0.8)
    axes[1].set_title("False Positives by Day of Week")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("False Positive Analysis (Ensemble)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, plots_dir, "08_false_positive_analysis.png", dpi)


# ── Plot 9: Cascade Clustering ───────────────────────────────────────

def plot_cascade_clustering(
    test_df: pd.DataFrame,
    config: dict,
):
    """
    Scatter plot of predicted cascade anomalies with x=date, y=service,
    colored by predicted_cascade_id. Non-cascade anomalies shown in gray.

    test_df (pd.DataFrame): Test data with predicted_cascade_id and
        is_predicted_cascade columns.
    config (dict): Visualization config.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    anom = test_df[test_df["ensemble_is_anomaly"] == 1].copy()

    if len(anom) == 0:
        print("      Skipping cascade plot: no anomalies.")
        return

    fig, ax = plt.subplots(figsize=config.get("figure_size_wide", [16, 6]))

    # Non-cascade anomalies in gray
    non_cascade = anom[anom["is_predicted_cascade"] == 0]
    if len(non_cascade) > 0:
        services = sorted(test_df["service"].unique())
        svc_map = {s: i for i, s in enumerate(services)}
        y_nc = non_cascade["service"].map(svc_map)
        ax.scatter(non_cascade["date"], y_nc, color="#bdc3c7",
                   s=20, alpha=0.5, label="Non-cascade anomaly")

    # Cascade anomalies colored by cluster
    cascade = anom[anom["is_predicted_cascade"] == 1]
    if len(cascade) > 0:
        services = sorted(test_df["service"].unique())
        svc_map = {s: i for i, s in enumerate(services)}

        unique_cascades = cascade["predicted_cascade_id"].unique()
        cmap = plt.cm.get_cmap("tab10", max(len(unique_cascades), 1))

        for i, cid in enumerate(unique_cascades):
            c_mask = cascade["predicted_cascade_id"] == cid
            c_rows = cascade[c_mask]
            y_c = c_rows["service"].map(svc_map)
            ax.scatter(c_rows["date"], y_c, color=cmap(i),
                       s=40, alpha=0.8, label=f"Cascade {cid}", zorder=3)

        ax.set_yticks(range(len(services)))
        ax.set_yticklabels(services)
    else:
        services = sorted(test_df["service"].unique())
        svc_map = {s: i for i, s in enumerate(services)}
        ax.set_yticks(range(len(services)))
        ax.set_yticklabels(services)

    ax.set_title("Predicted Cascade Clusters")
    ax.set_xlabel("Date")
    ax.set_ylabel("Service")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    _save_fig(fig, plots_dir, "09_cascade_clustering.png", dpi)


# ── Plot 10: SHAP Summary ────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    sample_df: pd.DataFrame,
    feature_cols: list,
    categorical_cols: list,
    config: dict,
):
    """
    SHAP beeswarm summary plot showing feature importance and impact
    direction across the explained sample.

    shap_values (np.ndarray): SHAP values array from compute_shap_values.
    sample_df (pd.DataFrame): The sample DataFrame that was explained.
    feature_cols (list[str]): Numerical feature column names.
    categorical_cols (list[str]): Categorical feature column names.
    config (dict): Visualization config.
    """
    import shap as shap_lib

    plots_dir = config["plots_dir"]
    dpi = config["dpi"]

    all_features = feature_cols + categorical_cols

    # For categorical features, convert to numeric codes for the plot
    plot_df = sample_df[all_features].copy()
    for col in categorical_cols:
        if plot_df[col].dtype.name == "category":
            plot_df[col] = plot_df[col].cat.codes
        else:
            plot_df[col] = pd.Categorical(plot_df[col]).codes

    fig, ax = plt.subplots(figsize=config.get("figure_size_square", [10, 8]))
    plt.sca(ax)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_lib.summary_plot(
            shap_values, plot_df,
            feature_names=all_features,
            show=False, max_display=20,
        )

    _save_fig(plt.gcf(), plots_dir, "10_shap_summary.png", dpi)


# ── Plots 11-13: SHAP Waterfall (one per anomaly type) ───────────────

def plot_shap_waterfall(
    model,
    test_df: pd.DataFrame,
    feature_cols: list,
    categorical_cols: list,
    anomaly_type: str,
    plot_number: int,
    config: dict,
):
    """
    SHAP waterfall plot for one example anomaly of the specified type,
    showing individual feature contributions for that specific prediction.

    model: Trained LightGBM Booster.
    test_df (pd.DataFrame): Test data with anomaly_type column.
    feature_cols (list[str]): Numerical feature column names.
    categorical_cols (list[str]): Categorical feature column names.
    anomaly_type (str): One of 'spike', 'cascade', 'drift'.
    plot_number (int): Plot number for filename (11, 12, or 13).
    config (dict): Visualization config.
    """
    import shap as shap_lib

    plots_dir = config["plots_dir"]
    dpi = config["dpi"]

    all_features = feature_cols + categorical_cols

    # Find an example of this anomaly type
    type_mask = (
        (test_df["anomaly_type"] == anomaly_type)
        & (test_df["ensemble_is_anomaly"] == 1)
    )
    candidates = test_df[type_mask]

    if len(candidates) == 0:
        # Fall back to any row with this anomaly type
        candidates = test_df[test_df["anomaly_type"] == anomaly_type]

    if len(candidates) == 0:
        print(f"      Skipping SHAP waterfall for '{anomaly_type}': "
              "no examples found.")
        return

    # Pick the row with highest ensemble score
    if "ensemble_score" in candidates.columns:
        row = candidates.loc[candidates["ensemble_score"].idxmax()]
    else:
        row = candidates.iloc[0]

    row_df = pd.DataFrame([row])
    for col in categorical_cols:
        row_df[col] = row_df[col].astype("category")

    X = row_df[all_features]

    explainer = shap_lib.TreeExplainer(model)
    sv = explainer.shap_values(X)
    expected = explainer.expected_value

    explanation = shap_lib.Explanation(
        values=sv[0],
        base_values=expected,
        data=X.values[0],
        feature_names=all_features,
    )

    fig, ax = plt.subplots(figsize=config.get("figure_size_square", [10, 8]))
    plt.sca(ax)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_lib.plots.waterfall(explanation, max_display=15, show=False)

    plt.title(f"SHAP Waterfall: {anomaly_type.title()} Anomaly "
              f"({row['date'].strftime('%Y-%m-%d')})")

    _save_fig(plt.gcf(), plots_dir,
              f"{plot_number}_shap_waterfall_{anomaly_type}.png", dpi)


# ── Plot 14: Ensemble ROC Curve ──────────────────────────────────────

def plot_ensemble_roc(
    test_df: pd.DataFrame,
    config: dict,
):
    """
    ROC curve for each model's anomaly scores against the ground-truth
    is_anomaly labels, with AUC values in the legend.

    test_df (pd.DataFrame): Test data with all model score columns.
    config (dict): Visualization config.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    y_true = test_df["is_anomaly"].values

    model_scores = [
        ("STL", "stl_anomaly_score", "#e74c3c"),
        ("LightGBM", "lgbm_anomaly_score", "#3498db"),
        ("Isolation Forest", "iforest_anomaly_score", "#e67e22"),
        ("Ensemble", "ensemble_score", "#2ecc71"),
    ]

    fig, ax = plt.subplots(figsize=config.get("figure_size_square", [10, 8]))

    for name, score_col, color in model_scores:
        if score_col not in test_df.columns:
            continue

        y_score = test_df[score_col].values

        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name} (AUC = {roc_auc:.4f})")
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], color="#bdc3c7", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: All Models")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    _save_fig(fig, plots_dir, "14_roc_curves.png", dpi)


# ── Plot 15: Drift Detection Showcase ────────────────────────────────

def plot_drift_showcase(
    daily_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
):
    """
    Highlights drift anomalies specifically by showing cost time series
    for an account/service/region group that contains drift events.
    Overlays the 28-day rolling average to show the gradual cost shift.

    daily_df (pd.DataFrame): Full dataset.
    test_df (pd.DataFrame): Test data with ensemble_is_anomaly.
    config (dict): Visualization config.
    """
    colors = config["colors"]
    plots_dir = config["plots_dir"]
    dpi = config["dpi"]
    _setup_style(colors)

    drift_rows = daily_df[daily_df["anomaly_type"] == "drift"]
    if len(drift_rows) == 0:
        print("      Skipping drift plot: no drift anomalies in dataset.")
        return

    # Pick a group with drift
    sample = drift_rows.iloc[0]
    acct = sample["account_id"]
    svc = sample["service"]
    rgn = sample["region"]

    mask = (
        (daily_df["account_id"] == acct)
        & (daily_df["service"] == svc)
        & (daily_df["region"] == rgn)
    )
    group = daily_df[mask].sort_values("date").copy()

    fig, ax = plt.subplots(figsize=config.get("figure_size_wide", [16, 6]))

    ax.plot(group["date"], group["daily_cost"],
            color="#2c3e50", linewidth=0.8, alpha=0.7, label="Daily cost")

    if "cost_28d_rolling_avg" in group.columns:
        ax.plot(group["date"], group["cost_28d_rolling_avg"],
                color="#3498db", linewidth=2, label="28-day rolling avg")

    # Highlight drift periods
    drift_mask = group["anomaly_type"] == "drift"
    if drift_mask.any():
        ax.scatter(group.loc[drift_mask, "date"],
                   group.loc[drift_mask, "daily_cost"],
                   color=colors["drift"], s=30, zorder=3,
                   label="Drift anomaly")

    ax.set_title(f"Drift Detection: {svc} / {rgn} / Account {acct}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Cost ($)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    _save_fig(fig, plots_dir, "15_drift_showcase.png", dpi)


# ── Master Plot Generator ────────────────────────────────────────────

def generate_all_plots(
    daily_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_date,
    comparison_df: pd.DataFrame,
    shap_values,
    shap_sample_df: pd.DataFrame,
    lgbm_model,
    feature_cols: list,
    categorical_cols: list,
    config: dict,
):
    """
    Orchestrates generation of all 15 visualization plots. Each plot
    is wrapped in try-except so a failure in one does not prevent the
    rest from being created.

    daily_df (pd.DataFrame): Full dataset before feature engineering.
    test_df (pd.DataFrame): Test data with all model outputs.
    split_date: Train/test split date.
    comparison_df (pd.DataFrame): Model comparison metrics table.
    shap_values: SHAP values array from compute_shap_values.
    shap_sample_df (pd.DataFrame): Sample used for SHAP computation.
    lgbm_model: Trained LightGBM Booster.
    feature_cols (list[str]): Numerical feature column names.
    categorical_cols (list[str]): Categorical feature column names.
    config (dict): Visualization section of pipeline_config.yaml
        merged with paths (plots_dir, dpi, colors, figure sizes).
    """
    plots_dir = config["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    plots = [
        ("01 Time Series Overview",
         lambda: plot_time_series_overview(daily_df, test_df, split_date, config)),
        ("02 Seasonal Patterns",
         lambda: plot_seasonal_patterns(daily_df, config)),
        ("03 STL Decomposition",
         lambda: plot_stl_decomposition(daily_df, test_df, config)),
        ("04 LightGBM Predictions",
         lambda: plot_lgbm_predictions(test_df, config)),
        ("05 Model Comparison",
         lambda: plot_model_comparison(comparison_df, config)),
        ("06 Detection by Anomaly Type",
         lambda: plot_detection_by_type(comparison_df, config)),
        ("07 Confusion Matrices",
         lambda: plot_confusion_matrices(test_df, config)),
        ("08 False Positive Analysis",
         lambda: plot_false_positive_analysis(test_df, config)),
        ("09 Cascade Clustering",
         lambda: plot_cascade_clustering(test_df, config)),
        ("10 SHAP Summary",
         lambda: plot_shap_summary(
             shap_values, shap_sample_df,
             feature_cols, categorical_cols, config)),
        ("11 SHAP Waterfall: Spike",
         lambda: plot_shap_waterfall(
             lgbm_model, test_df, feature_cols, categorical_cols,
             "spike", 11, config)),
        ("12 SHAP Waterfall: Cascade",
         lambda: plot_shap_waterfall(
             lgbm_model, test_df, feature_cols, categorical_cols,
             "cascade", 12, config)),
        ("13 SHAP Waterfall: Drift",
         lambda: plot_shap_waterfall(
             lgbm_model, test_df, feature_cols, categorical_cols,
             "drift", 13, config)),
        ("14 ROC Curves",
         lambda: plot_ensemble_roc(test_df, config)),
        ("15 Drift Detection Showcase",
         lambda: plot_drift_showcase(daily_df, test_df, config)),
    ]

    successful = 0
    failed = 0

    for name, plot_fn in plots:
        try:
            print(f"    Generating: {name}")
            plot_fn()
            successful += 1
        except Exception as e:
            print(f"      FAILED: {name} — {e}")
            failed += 1

    print(f"\n    Plots generated: {successful}/{len(plots)} "
          f"({failed} failed)")
