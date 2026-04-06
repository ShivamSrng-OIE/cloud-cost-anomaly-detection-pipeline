import os
from typing import List

import pandas as pd

from src.utilities.general_utils import console_and_logger


class SummaryBuilder:

    def __init__(
            self,
            logger,
        ) -> None:
        """
        Stores the logger used for printing summary lines to the
        console and recording them in the log file.

        logger: Pipeline logger for progress messages.
        """
        self.__logger = logger

    def generate(
            self,
            raw_path: str,
            agg_path: str,
            anomaly_plan: dict,
            accounts: List[dict],
            config: dict,
            output_path: str,
        ) -> None:
        """
        Reads both the raw CUR and daily aggregated CSVs, computes key
        statistics (row counts, file sizes, cost breakdowns by service
        and account, date range, anomaly counts), formats everything
        into a human-readable text block, prints it to the console,
        and saves it to a text file.

        raw_path (str): Path to the raw CUR CSV.
        agg_path (str): Path to the daily aggregated CSV.
        anomaly_plan (dict): Full anomaly plan from AnomalyPlanner.
        accounts (List[dict]): Account metadata for name mapping.
        config (dict): Merged generation configuration.
        output_path (str): Destination path for the summary text file.
        """
        raw_rows = self.__count_csv_rows(raw_path)
        agg_rows = self.__count_csv_rows(agg_path)
        agg_df = pd.read_csv(agg_path)

        lines = self.__build_summary_lines(
            raw_rows=raw_rows,
            agg_rows=agg_rows,
            agg_df=agg_df,
            anomaly_plan=anomaly_plan,
            accounts=accounts,
            config=config,
            raw_path=raw_path,
            agg_path=agg_path,
        )

        summary_text = "\n".join(lines)

        # Print to console
        for line in lines:
            console_and_logger(self.__logger, line)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")

        console_and_logger(
            self.__logger,
            f"Summary saved: {output_path}",
        )

    def __count_csv_rows(
            self,
            path: str,
        ) -> int:
        """
        Counts the number of data rows in a CSV file by streaming
        through it line by line (skipping the header). This avoids
        loading the entire file into memory just to get a row count.

        path (str): Filesystem path to the CSV.

        Returns the integer row count (excludes the header line).
        """
        count = 0
        with open(path, "r") as f:
            next(f)  # skip header
            for _ in f:
                count += 1
        return count

    def __build_summary_lines(
            self,
            raw_rows: int,
            agg_rows: int,
            agg_df: pd.DataFrame,
            anomaly_plan: dict,
            accounts: List[dict],
            config: dict,
            raw_path: str,
            agg_path: str,
        ) -> List[str]:
        """
        Assembles the formatted summary text as a list of strings.
        Reads the raw CUR one more time (just the cost and service
        columns) to compute the top-5-services and top-5-accounts
        cost breakdowns. Also extracts date range from the aggregated
        DataFrame and tallies anomaly counts by type.

        raw_rows (int): Row count from the raw CUR CSV.
        agg_rows (int): Row count from the aggregated CSV.
        agg_df (pd.DataFrame): Full aggregated DataFrame.
        anomaly_plan (dict): Full anomaly plan for counting events.
        accounts (List[dict]): Account metadata for name mapping.
        config (dict): Merged pipeline configuration.
        raw_path (str): Path to the raw CUR CSV (re-read for cost stats).
        agg_path (str): Path to the aggregated CSV (for file size).

        Returns an ordered list of summary text lines.
        """
        # Compute cost breakdown by service
        raw_chunk = pd.read_csv(raw_path, usecols=[
            "lineItem/ProductCode",
            "lineItem/UnblendedCost",
        ])
        total_cost = raw_chunk["lineItem/UnblendedCost"].sum()
        svc_cost = (
            raw_chunk.groupby("lineItem/ProductCode")["lineItem/UnblendedCost"]
            .sum()
            .sort_values(ascending=False)
        )
        top_services = svc_cost.head(5)

        # Account cost breakdown
        acct_chunk = pd.read_csv(raw_path, usecols=[
            "lineItem/UsageAccountId",
            "lineItem/UnblendedCost",
        ])
        acct_chunk["lineItem/UsageAccountId"] = (
            acct_chunk["lineItem/UsageAccountId"].astype(str)
        )
        acct_cost = (
            acct_chunk.groupby("lineItem/UsageAccountId")["lineItem/UnblendedCost"]
            .sum()
            .sort_values(ascending=False)
        )
        acct_map = {a["account_id"]: a["account_name"] for a in accounts}
        top_accounts = acct_cost.head(5)

        # Date range from aggregated
        if "date" in agg_df.columns:
            dates = pd.to_datetime(agg_df["date"])
            date_min = dates.min().strftime("%Y-%m-%d")
            date_max = dates.max().strftime("%Y-%m-%d")
        else:
            date_min = "N/A"
            date_max = "N/A"

        # Anomaly stats
        num_spikes = len(anomaly_plan.get("spikes", []))
        num_cascades = len(anomaly_plan.get("cascades", []))
        num_drifts = len(anomaly_plan.get("drifts", []))
        anom_rows = int(agg_df.get("is_anomaly", pd.Series([0])).sum())

        # File sizes
        raw_size = os.path.getsize(raw_path) / (1024 * 1024)
        agg_size = os.path.getsize(agg_path) / (1024 * 1024)

        lines = [
            "=" * 60,
            "DATASET SUMMARY",
            "=" * 60,
            "",
            f"  Date range:          {date_min} → {date_max}",
            f"  Accounts:            {config['num_accounts']}",
            f"  Services:            {len(config['services'])}",
            f"  Regions:             {len(config['regions'])}",
            f"  Target rows:         {config['target_rows']:,}",
            f"  Actual rows (raw):   {raw_rows:,}",
            "",
            "Files:",
            f"  raw_cur_data.csv     {raw_rows:>10,} rows  ({raw_size:.1f} MB)",
            f"  daily_aggregated.csv {agg_rows:>10,} rows  ({agg_size:.1f} MB)",
            "",
            f"Total unblended cost:  ${total_cost:,.2f}",
            "",
            "Top 5 services by cost:",
        ]

        for svc_name, cost in top_services.items():
            pct = cost / total_cost * 100 if total_cost else 0
            lines.append(f"  {svc_name:<25s} ${cost:>12,.2f}  ({pct:5.1f}%)")

        lines.append("")
        lines.append("Top 5 accounts by cost:")

        for aid, cost in top_accounts.items():
            aname = acct_map.get(str(aid), str(aid))
            pct = cost / total_cost * 100 if total_cost else 0
            lines.append(f"  {aname:<25s} ${cost:>12,.2f}  ({pct:5.1f}%)")

        lines.extend([
            "",
            "Anomalies injected:",
            f"  Spikes:              {num_spikes}",
            f"  Cascades:            {num_cascades}",
            f"  Drifts:              {num_drifts}",
            f"  Labelled rows:       {anom_rows:,}",
            "",
            "=" * 60,
        ])

        return lines
