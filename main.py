from argparse import ArgumentParser


def _add_subparser_to_generate_synthetic_cur_data(
    subparsers: ArgumentParser,
) -> None:
    """Register the generate_synthetic_cur_data subcommand."""
    parser = subparsers.add_parser(
        name="generate_synthetic_cur_data",
        help=(
            "Generate a synthetic AWS CUR dataset with configurable "
            "accounts, services, regions, and anomaly injection."
        ),
        description=(
            "Produce a realistic synthetic AWS Cost and Usage Report with "
            "spike, cascade, and drift anomalies for anomaly-detection "
            "model development."
        ),
        aliases=[
            "generate_cur",
            "generate_data",
        ],
    )
    parser.add_argument(
        "--num_accounts",
        type=int,
        default=None,
        help="Number of AWS accounts to simulate (default: 5).",
    )
    parser.add_argument(
        "--num_months",
        type=int,
        default=None,
        help="Number of months of data to generate (default: 12).",
    )
    parser.add_argument(
        "--services",
        type=str,
        default=None,
        help="Comma-separated list of AWS services (overrides config.yaml).",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        help="Comma-separated list of AWS regions (overrides config.yaml).",
    )
    parser.add_argument(
        "--num_spike_anomalies",
        type=int,
        default=None,
        help="Number of single-day cost spike anomalies (default: 15).",
    )
    parser.add_argument(
        "--num_cascade_anomalies",
        type=int,
        default=None,
        help="Number of cross-service cascade anomalies (default: 8).",
    )
    parser.add_argument(
        "--num_drift_anomalies",
        type=int,
        default=None,
        help="Number of slow-burn cost drift anomalies (default: 5).",
    )
    parser.add_argument(
        "--seasonal_strength",
        type=int,
        default=None,
        help="Seasonal effect strength on a scale of 1-5 (default: 3).",
    )
    parser.add_argument(
        "--spike_magnitude",
        type=float,
        default=None,
        help="Spike anomaly multiplier of normal cost (default: 3.0).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output files (default: ./output/synthetic-data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--target_rows",
        type=int,
        default=None,
        help="Approximate target row count in the raw CUR file (default: 1000000).",
    )


def _add_subparser_to_run_pipeline(
    subparsers: ArgumentParser,
) -> None:
    subparsers.add_parser(
        name="run_pipeline",
        help=(
            "Run the ML anomaly detection pipeline on the generated "
            "synthetic dataset."
        ),
        description=(
            "End-to-end orchestrator that loads synthetic AWS CUR data, "
            "engineers features, runs five detection models (baseline "
            "threshold, STL decomposition, LightGBM, Isolation Forest, "
            "weighted ensemble), performs cascade clustering, generates "
            "SHAP explanations, evaluates all models, and produces 15 "
            "diagnostic plots."
        ),
        aliases=[
            "pipeline",
            "detect",
        ],
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="North.Cloud - Anomaly Detection Pipeline",
    )
    subparsers = parser.add_subparsers(dest="subcommands")

    _add_subparser_to_generate_synthetic_cur_data(subparsers)
    _add_subparser_to_run_pipeline(subparsers)

    kwargs = parser.parse_args().__dict__
    if subcommands := kwargs.pop("subcommands"):
        from src.engine import Engine
        method = {
            "generate_cur": "generate_synthetic_cur_data",
            "generate_data": "generate_synthetic_cur_data",
            "pipeline": "run_pipeline",
            "detect": "run_pipeline",
        }.get(subcommands, subcommands)
        getattr(Engine(), method)(**kwargs)
    else:
        parser.print_help()

