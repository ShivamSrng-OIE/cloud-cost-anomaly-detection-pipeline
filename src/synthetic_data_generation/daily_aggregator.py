from typing import Dict, List, Tuple

import pandas as pd

from src.utilities.general_utils import console_and_logger
from src.utilities.consts_handler import AGG_COLUMNS


class DailyAggregator:

    def __init__(
            self,
            logger,
        ) -> None:
        """
        Stores the logger used for progress reporting during the
        aggregation, rolling-stats, and anomaly-labelling stages.

        logger: Pipeline logger for progress messages.
        """
        self.__logger = logger

    def aggregate(
            self,
            raw_path: str,
            output_path: str,
            accounts: List[dict],
            anomaly_lookup: dict,
        ) -> None:
        """
        Reads the raw CUR CSV in 500k-row chunks, aggregates each chunk
        to daily granularity per (account, service, region), then merges
        and re-aggregates to handle days split across chunk boundaries.
        After that it enriches the DataFrame with temporal features
        (day_of_week, is_weekend, etc.), computes 7-day and 28-day
        rolling cost averages per group, and labels each row with
        anomaly flags from the pre-planned lookup.

        raw_path (str): Path to the raw CUR CSV from RawCurGenerator.
        output_path (str): Destination path for the aggregated CSV.
        accounts (List[dict]): Account metadata for name mapping.
        anomaly_lookup (dict): Anomaly lookup from AnomalyPlanner.
        """
        account_map = {a["account_id"]: a["account_name"] for a in accounts}

        console_and_logger(self.__logger, "Reading raw CUR data in chunks...")

        chunks: List[pd.DataFrame] = []
        for chunk in pd.read_csv(raw_path, chunksize=500_000):
            agg = self.__aggregate_chunk(chunk)
            chunks.append(agg)

        daily = pd.concat(chunks, ignore_index=True)

        # Re-aggregate in case a day was split across chunks
        group_cols = ["date", "account_id", "service", "region"]
        daily = daily.groupby(group_cols, as_index=False).agg({
            "daily_cost": "sum",
            "daily_usage": "sum",
            "num_resources": "sum",
        })

        # Account name
        daily["account_name"] = daily["account_id"].astype(str).map(account_map)

        # Temporal features
        daily["date"] = pd.to_datetime(daily["date"])
        daily = self.__add_temporal_features(daily)

        # Rolling averages per group
        daily = self.__add_rolling_stats(daily)

        # Anomaly labels
        daily = self.__label_anomalies(daily, anomaly_lookup)

        # Final column order
        daily = daily[[c for c in AGG_COLUMNS if c in daily.columns]]
        daily.to_csv(output_path, index=False)

        console_and_logger(
            self.__logger,
            f"Daily aggregated saved: {output_path} ({len(daily):,} rows)",
        )

    def __aggregate_chunk(
            self,
            chunk: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Takes a single chunk from pd.read_csv and collapses it to daily
        totals per (date, account_id, service, region). Extracts the date
        from lineItem/UsageStartDate, sums unblended cost and usage amount,
        and counts unique resource IDs.

        chunk (pd.DataFrame): Raw CUR rows from one read_csv chunk.

        Returns a DataFrame with columns: date, account_id, service,
        region, daily_cost, daily_usage, num_resources.
        """
        chunk["date"] = pd.to_datetime(
            chunk["lineItem/UsageStartDate"],
        ).dt.date

        chunk["account_id"] = chunk["lineItem/UsageAccountId"].astype(str)

        agg = chunk.groupby(
            ["date", "account_id", "lineItem/ProductCode", "product/region"],
            as_index=False,
        ).agg(
            daily_cost=("lineItem/UnblendedCost", "sum"),
            daily_usage=("lineItem/UsageAmount", "sum"),
            num_resources=("lineItem/ResourceId", "nunique"),
        )

        agg = agg.rename(columns={
            "lineItem/ProductCode": "service",
            "product/region": "region",
        })

        return agg

    def __add_temporal_features(
            self,
            df: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Adds calendar-based columns derived from the date column:
        day_of_week (0=Mon), day_name, is_weekend flag, day_of_month,
        week_of_year, and month. These serve as potential features for
        downstream anomaly-detection models.

        df (pd.DataFrame): DataFrame with a datetime 'date' column.

        Returns the same DataFrame with the new columns appended.
        """
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_name"] = df["date"].dt.day_name()
        df["is_weekend"] = df["day_of_week"].ge(5).astype(int)
        df["day_of_month"] = df["date"].dt.day
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month
        return df

    def __add_rolling_stats(
            self,
            df: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Computes 7-day and 28-day rolling average costs for each
        (account_id, service, region) group, plus the percentage change
        of each day's cost relative to that rolling average. These
        rolling signals help anomaly-detection models distinguish
        genuine spikes from normal variance.

        df (pd.DataFrame): Daily-aggregated DataFrame sorted by date.

        Returns the same DataFrame with four new columns:
        cost_7d_rolling_avg, cost_pct_change_vs_7d_avg,
        cost_28d_rolling_avg, cost_pct_change_vs_28d_avg.
        """
        df = df.sort_values(
            ["account_id", "service", "region", "date"],
        ).reset_index(drop=True)

        group_key = ["account_id", "service", "region"]

        df["cost_7d_rolling_avg"] = (
            df.groupby(group_key)["daily_cost"]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )
        df["cost_pct_change_vs_7d_avg"] = (
            (df["daily_cost"] - df["cost_7d_rolling_avg"])
            / df["cost_7d_rolling_avg"].replace(0, float("nan"))
        ).fillna(0.0).round(4)

        df["cost_28d_rolling_avg"] = (
            df.groupby(group_key)["daily_cost"]
            .transform(lambda x: x.rolling(28, min_periods=1).mean())
        )
        df["cost_pct_change_vs_28d_avg"] = (
            (df["daily_cost"] - df["cost_28d_rolling_avg"])
            / df["cost_28d_rolling_avg"].replace(0, float("nan"))
        ).fillna(0.0).round(4)

        return df

    def __label_anomalies(
            self,
            df: pd.DataFrame,
            anomaly_lookup: dict,
        ) -> pd.DataFrame:
        """
        Iterates through every row and checks the anomaly lookup for a
        matching (date, account_id, service) key. Rows that match get
        is_anomaly=1 with the anomaly_type and cascade_id filled in;
        non-matching rows get is_anomaly=0 with empty strings. These
        labels are the ground-truth targets for model training.

        df (pd.DataFrame): Daily-aggregated DataFrame.
        anomaly_lookup (dict): Keyed by (date, account_id, service).

        Returns the DataFrame with is_anomaly, anomaly_type, and
        cascade_id columns appended.
        """
        is_anomaly: list = []
        anomaly_type: list = []
        cascade_id: list = []

        for _, row in df.iterrows():
            d = row["date"]
            if hasattr(d, "date"):
                d = d.date()
            key = (d, str(row["account_id"]), row["service"])
            info = anomaly_lookup.get(key)

            if info:
                is_anomaly.append(1)
                anomaly_type.append(info["type"])
                cascade_id.append(info.get("cascade_id", ""))
            else:
                is_anomaly.append(0)
                anomaly_type.append("")
                cascade_id.append("")

        df["is_anomaly"] = is_anomaly
        df["anomaly_type"] = anomaly_type
        df["cascade_id"] = cascade_id

        anom_count = sum(is_anomaly)
        console_and_logger(
            self.__logger,
            f"Anomaly-labelled rows: {anom_count:,} / {len(df):,}",
        )

        return df
