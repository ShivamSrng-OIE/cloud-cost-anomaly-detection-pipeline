import os
import time
from typing import Optional, List

import numpy as np

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
            "output_dir":            _first(output_dir, output_consts.get("output_dir"), "./output"),
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
        console_and_logger(self.__logger, "North.Cloud - Synthetic AWS CUR Dataset Generator")
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

