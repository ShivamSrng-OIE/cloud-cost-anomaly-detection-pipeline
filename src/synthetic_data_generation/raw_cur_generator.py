import datetime
import uuid
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utilities.general_utils import console_and_logger
from src.utilities.consts_handler import (
    SERVICE_CONFIG,
    REGION_PREFIX_MAP,
    TEAMS,
    ENVIRONMENTS,
    TAG_MISSING_RATE_TEAM,
    TAG_MISSING_RATE_ENVIRONMENT,
    LINE_ITEM_TYPES,
    SAVINGS_PLAN_DISCOUNT,
    CREDIT_FRACTION,
    TAX_FRACTION,
    MONTHLY_GROWTH_RATE,
    DAILY_NOISE_STD,
    MONTH_START_SPIKE_DAYS,
    MONTH_START_SPIKE_MULTIPLIER,
    MONTH_END_SPIKE_DAYS,
    MONTH_END_SPIKE_MULTIPLIER,
    MONTH_START_SPIKE_SERVICES,
    MONTH_END_SPIKE_SERVICES,
    BLENDED_RATE_FACTOR,
    RAW_CUR_COLUMNS,
)


class RawCurGenerator:

    def __init__(
            self,
            logger,
            generation_config: dict,
        ) -> None:
        """
        Stores the logger and the merged generation config that controls
        services, regions, seed, months, seasonal strength, and other
        parameters used throughout CUR line-item generation.

        logger: Pipeline logger for progress messages.
        generation_config (dict): Flat dict produced by Engine.__merge_config.
        """
        self.__logger = logger
        self.__config = generation_config

    def generate(
            self,
            output_path: str,
            accounts: List[dict],
            resource_pool: dict,
            anomaly_lookup: dict,
        ) -> None:
        """
        Produces the full raw CUR CSV by iterating over each billing month
        and writing one DataFrame chunk per month to keep memory bounded.
        For every day within each month, it walks the full
        (account, service, region, resource) space and builds one line-item
        row per resource, applying growth, seasonal patterns, weekday/weekend
        traffic differences, boundary spikes, and anomaly injection.

        output_path (str): Filesystem path for the output CSV.
        accounts (List[dict]): Account metadata from AccountBuilder.
        resource_pool (dict): Nested ARN pool
            pool[account_id][service][region] -> list of ARN strings.
        anomaly_lookup (dict): Lookup from AnomalyPlanner mapping
            (date, account_id, service) tuples to anomaly info dicts.
        """
        rng = np.random.RandomState(self.__config["seed"])
        services = self.__config["services"]
        regions = self.__config["regions"]
        num_months = self.__config["num_months"]

        today = datetime.date.today()
        month_starts = self.__compute_month_starts(today, num_months)
        seasonal_strength = self.__config["seasonal_strength"]

        total_rows = 0
        header_written = False

        for month_idx, m_start in enumerate(tqdm(
            month_starts,
            desc="Generating months",
            unit="month",
        )):
            m_end = (m_start + datetime.timedelta(days=32)).replace(day=1)
            days_in_month = (m_end - m_start).days
            growth = self.__growth_multiplier(month_idx, rng)

            rows: List[dict] = []

            for day_offset in range(days_in_month):
                current_date = m_start + datetime.timedelta(days=day_offset)
                dow = current_date.weekday()
                seasonal = self.__seasonal_multiplier(
                    current_date, seasonal_strength,
                )

                for acct in accounts:
                    aid = acct["account_id"]
                    cost_mult = acct["cost_multiplier"]

                    for svc in services:
                        svc_cfg = SERVICE_CONFIG.get(svc)
                        if svc_cfg is None:
                            continue

                        # Weekday / weekend traffic
                        wmin, wmax = svc_cfg["weekday_multiplier"]
                        weekday_mult = (
                            rng.uniform(wmin, wmax) if dow < 5
                            else rng.uniform(0.3, 0.6)
                        )

                        for region in regions:
                            arns = resource_pool.get(aid, {}).get(
                                svc, {},
                            ).get(region, [])

                            for arn in arns:
                                row = self.__build_row(
                                    rng=rng,
                                    current_date=current_date,
                                    m_start=m_start,
                                    m_end=m_end,
                                    day_offset=day_offset,
                                    days_in_month=days_in_month,
                                    acct=acct,
                                    svc=svc,
                                    svc_cfg=svc_cfg,
                                    region=region,
                                    arn=arn,
                                    cost_mult=cost_mult,
                                    growth=growth,
                                    seasonal=seasonal,
                                    weekday_mult=weekday_mult,
                                    anomaly_lookup=anomaly_lookup,
                                )
                                rows.append(row)

            df = pd.DataFrame(rows, columns=RAW_CUR_COLUMNS)
            df.to_csv(
                output_path,
                mode="a" if header_written else "w",
                header=not header_written,
                index=False,
            )
            header_written = True
            total_rows += len(df)

        console_and_logger(
            self.__logger,
            f"Raw CUR saved: {output_path} ({total_rows:,} rows)",
        )

    def __compute_month_starts(
            self,
            today: datetime.date,
            num_months: int,
        ) -> List[datetime.date]:
        """
        Walks backwards from the current month and collects the first day
        of each preceding billing month, then sorts them chronologically.
        This gives the generator a deterministic, ordered sequence of
        months to iterate over.

        today (datetime.date): Reference date (usually today).
        num_months (int): How many months of history to generate.

        Returns a sorted list of month-start dates.
        """
        current = today.replace(day=1)
        starts: List[datetime.date] = []

        for _ in range(num_months):
            current = (current - datetime.timedelta(days=1)).replace(day=1)
            starts.append(current)

        starts.sort()
        return starts

    def __growth_multiplier(
            self,
            month_idx: int,
            rng: np.random.RandomState,
        ) -> float:
        """
        Computes a compounding month-over-month growth factor by sampling
        a rate from MONTHLY_GROWTH_RATE and raising (1 + rate) to the
        power of month_idx. Earlier months get a multiplier near 1.0;
        later months compound upward.

        month_idx (int): Zero-based index of the current billing month.
        rng (np.random.RandomState): Seeded RNG.

        Returns the growth multiplier as a float >= 1.0.
        """
        rate = rng.uniform(*MONTHLY_GROWTH_RATE)
        return (1.0 + rate) ** month_idx

    def __seasonal_multiplier(
            self,
            current_date: datetime.date,
            strength: int,
        ) -> float:
        """
        Applies a sinusoidal seasonal curve to cost data based on the day
        of the year. The curve peaks around December and troughs near June,
        mimicking real-world cloud spend patterns where year-end activity
        drives higher costs. The amplitude scales with the strength
        parameter (1 = subtle, 5 = dramatic).

        current_date (datetime.date): The date to compute the factor for.
        strength (int): Seasonal effect intensity from 1 to 5.

        Returns a float centred around 1.0 representing the seasonal
        cost adjustment.
        """
        import math
        doy = current_date.timetuple().tm_yday
        raw = math.sin(2 * math.pi * (doy - 80) / 365)
        amplitude = 0.05 * strength
        return 1.0 + amplitude * raw

    def __pick_line_item_type(
            self,
            rng: np.random.RandomState,
        ) -> str:
        """
        Randomly selects a CUR lineItem/LineItemType (e.g. "Usage",
        "Tax", "SavingsPlanCoveredUsage", "Credit") according to the
        probability weights defined in LINE_ITEM_TYPES.

        rng (np.random.RandomState): Seeded RNG.

        Returns the selected type as a string.
        """
        types = list(LINE_ITEM_TYPES.keys())
        probs = list(LINE_ITEM_TYPES.values())
        return types[rng.choice(len(types), p=probs)]

    def __assign_tag(
            self,
            rng: np.random.RandomState,
            values: list,
            missing_rate: float,
        ) -> str:
        """
        Picks a random tag value from the provided list, but returns an
        empty string with probability equal to missing_rate. This simulates
        the real-world problem of untagged or partially tagged resources
        in AWS environments.

        rng (np.random.RandomState): Seeded RNG.
        values (list): Pool of possible tag strings (e.g. team names).
        missing_rate (float): Probability in [0, 1] of returning "".

        Returns a tag string or empty string.
        """
        if rng.random() < missing_rate:
            return ""
        return values[rng.randint(len(values))]

    def __build_row(
            self,
            rng: np.random.RandomState,
            current_date: datetime.date,
            m_start: datetime.date,
            m_end: datetime.date,
            day_offset: int,
            days_in_month: int,
            acct: dict,
            svc: str,
            svc_cfg: dict,
            region: str,
            arn: str,
            cost_mult: float,
            growth: float,
            seasonal: float,
            weekday_mult: float,
            anomaly_lookup: dict,
        ) -> dict:
        """
        Assembles a single CUR line-item dict representing one resource's
        usage for one day. The cost calculation layers several multipliers:

            base_usage * noise * weekday * account_cost_mult
            * growth * seasonal * boundary_spike * anomaly_injection

        After computing unblended cost from (usage * rate), it derives
        a blended rate/cost using a random factor, then adjusts both
        costs according to the line-item type (SavingsPlan discount,
        Credit negative fraction, Tax fraction). Tags are assigned with
        configurable missing rates.

        rng (np.random.RandomState): Seeded RNG.
        current_date (datetime.date): Usage date for this line item.
        m_start / m_end (datetime.date): Billing period boundaries.
        day_offset (int): Zero-based day within the month.
        days_in_month (int): Total days in this billing month.
        acct (dict): Account metadata including account_id.
        svc (str): Service code like "AmazonEC2".
        svc_cfg (dict): Service config from SERVICE_CONFIG.
        region (str): AWS region string.
        arn (str): Resource ARN from the pre-built pool.
        cost_mult (float): Per-account cost scaling factor.
        growth (float): Compounding monthly growth multiplier.
        seasonal (float): Seasonal cost adjustment for this date.
        weekday_mult (float): Weekday vs weekend traffic multiplier.
        anomaly_lookup (dict): Full anomaly lookup for injection.

        Returns a dict whose keys match RAW_CUR_COLUMNS.
        """
        aid = acct["account_id"]

        # Usage type and rate
        usage_types = list(svc_cfg["usage_types"].keys())
        ut_key = usage_types[rng.randint(len(usage_types))]
        ut_info = svc_cfg["usage_types"][ut_key]
        base_rate = ut_info["rate"]
        instance_type = ut_info.get("instance_type", "")

        region_prefix = REGION_PREFIX_MAP.get(region, "")
        full_usage_type = f"{region_prefix}-{ut_key}" if region_prefix else ut_key

        # Base usage and cost
        noise_std = rng.uniform(*DAILY_NOISE_STD)
        noise = max(0.3, rng.normal(1.0, noise_std))
        base_usage = rng.uniform(1, 24) * noise * weekday_mult * cost_mult

        # Month boundary spikes
        boundary_mult = 1.0
        if svc in MONTH_START_SPIKE_SERVICES and day_offset < MONTH_START_SPIKE_DAYS:
            boundary_mult = rng.uniform(*MONTH_START_SPIKE_MULTIPLIER)
        elif svc in MONTH_END_SPIKE_SERVICES and day_offset >= days_in_month - MONTH_END_SPIKE_DAYS:
            boundary_mult = rng.uniform(*MONTH_END_SPIKE_MULTIPLIER)

        usage_amount = base_usage * growth * seasonal * boundary_mult

        # Anomaly injection
        anomaly_key = (current_date, aid, svc)
        anomaly_info = anomaly_lookup.get(anomaly_key)
        if anomaly_info:
            atype = anomaly_info["type"]
            if atype == "spike":
                usage_amount *= anomaly_info["magnitude"]
            elif atype == "cascade":
                m_range = anomaly_info["magnitude_range"]
                usage_amount *= rng.uniform(*m_range)
            elif atype == "drift":
                usage_amount *= anomaly_info["compound"] ** anomaly_info["day_num"]

        unblended_cost = round(usage_amount * base_rate, 6)
        blended_rate = round(
            base_rate * rng.uniform(*BLENDED_RATE_FACTOR), 6,
        )
        blended_cost = round(usage_amount * blended_rate, 6)

        # Line-item type adjustments
        lit = self.__pick_line_item_type(rng)
        if lit == "SavingsPlanCoveredUsage":
            unblended_cost = round(unblended_cost * SAVINGS_PLAN_DISCOUNT, 6)
            blended_cost = round(blended_cost * SAVINGS_PLAN_DISCOUNT, 6)
        elif lit == "Credit":
            unblended_cost = round(unblended_cost * CREDIT_FRACTION, 6)
            blended_cost = round(blended_cost * CREDIT_FRACTION, 6)
        elif lit == "Tax":
            unblended_cost = round(abs(unblended_cost) * TAX_FRACTION, 6)
            blended_cost = round(abs(blended_cost) * TAX_FRACTION, 6)

        # Description
        desc = svc_cfg["description_template"].replace(
            "${rate}", f"{base_rate:.6f}",
        ).replace("{instance_type}", instance_type)

        operation = svc_cfg["operations"][rng.randint(len(svc_cfg["operations"]))]

        usage_end = current_date + datetime.timedelta(days=1)

        return {
            "identity/LineItemId": uuid.uuid4().hex,
            "bill/BillingPeriodStartDate": m_start.isoformat(),
            "bill/BillingPeriodEndDate": m_end.isoformat(),
            "bill/PayerAccountId": aid,
            "lineItem/UsageAccountId": aid,
            "lineItem/LineItemType": lit,
            "lineItem/UsageStartDate": current_date.isoformat(),
            "lineItem/UsageEndDate": usage_end.isoformat(),
            "lineItem/ProductCode": svc,
            "lineItem/UsageType": full_usage_type,
            "lineItem/Operation": operation,
            "lineItem/ResourceId": arn,
            "lineItem/UsageAmount": round(usage_amount, 6),
            "lineItem/UnblendedRate": round(base_rate, 6),
            "lineItem/UnblendedCost": unblended_cost,
            "lineItem/BlendedRate": blended_rate,
            "lineItem/BlendedCost": blended_cost,
            "lineItem/LineItemDescription": desc,
            "lineItem/CurrencyCode": "USD",
            "product/region": region,
            "product/instanceType": instance_type,
            "product/productFamily": svc_cfg["product_family"],
            "product/productName": svc_cfg["product_name"],
            "pricing/unit": svc_cfg["pricing_unit"],
            "resourceTags/user:team": self.__assign_tag(
                rng, TEAMS, TAG_MISSING_RATE_TEAM,
            ),
            "resourceTags/user:environment": self.__assign_tag(
                rng, ENVIRONMENTS, TAG_MISSING_RATE_ENVIRONMENT,
            ),
        }
