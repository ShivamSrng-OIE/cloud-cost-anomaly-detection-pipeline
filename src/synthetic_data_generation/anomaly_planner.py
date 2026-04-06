import datetime
import uuid
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utilities.general_utils import console_and_logger
from src.utilities.consts_handler import (
    SERVICE_CONFIG,
    ANOMALY_LOG_COLUMNS,
    CASCADE_SERVICES,
    DRIFT_DURATION_RANGE,
)


class AnomalyPlanner:

    def __init__(
            self,
            logger,
            generation_config: dict,
        ) -> None:
        """
        Stores the logger and generation config needed to determine how
        many anomalies of each type to create, which date window to
        scatter them across, and the seed for reproducibility.

        logger: Pipeline logger for progress messages.
        generation_config (dict): Flat dict produced by Engine.__merge_config.
        """
        self.__logger = logger
        self.__config = generation_config

    def plan_anomalies(
            self,
            accounts: List[dict],
        ) -> dict:
        """
        Generates the full anomaly schedule — spikes, cascades, and drifts —
        scattered randomly across the configured date window. Each anomaly
        type is planned independently, then all three are merged into a
        single O(1) lookup dict keyed by (date, account_id, service).

        accounts (List[dict]): Account metadata dicts from AccountBuilder.

        Returns a dict with four keys:
            spikes (list[dict]): Single-day cost spike definitions.
            cascades (list[dict]): Multi-service cascade definitions.
            drifts (list[dict]): Multi-day upward drift definitions.
            lookup (dict): {(date, account_id, service): anomaly_info}
                used by RawCurGenerator for O(1) injection during row building.
        """
        rng = np.random.RandomState(self.__config["seed"] + 100)
        today = datetime.date.today()
        start_date = today.replace(day=1) - datetime.timedelta(
            days=self.__config["num_months"] * 30,
        )
        end_date = today
        all_dates = pd.date_range(start_date, end_date).date.tolist()

        services = self.__config["services"]

        spikes = self.__plan_spikes(
            rng=rng,
            accounts=accounts,
            services=services,
            all_dates=all_dates,
        )
        cascades = self.__plan_cascades(
            rng=rng,
            accounts=accounts,
            all_dates=all_dates,
        )
        drifts = self.__plan_drifts(
            rng=rng,
            accounts=accounts,
            services=services,
            all_dates=all_dates,
        )

        lookup = self.__build_lookup(
            spikes=spikes,
            cascades=cascades,
            drifts=drifts,
        )

        console_and_logger(
            self.__logger,
            f"Planned {len(spikes)} spikes, {len(cascades)} cascades, "
            f"{len(drifts)} drifts",
        )
        console_and_logger(
            self.__logger,
            f"Anomaly lookup entries: {len(lookup):,}",
        )

        return {
            "spikes": spikes,
            "cascades": cascades,
            "drifts": drifts,
            "lookup": lookup,
        }

    def write_anomaly_log(
            self,
            anomaly_plan: dict,
            output_path: str,
        ) -> None:
        """
        Flattens every spike, cascade, and drift from the anomaly plan
        into a uniform row format and writes them to a CSV file. This
        log serves as ground truth for model evaluation.

        anomaly_plan (dict): Full plan dict returned by plan_anomalies().
        output_path (str): Destination path for the anomaly_log.csv file.
        """
        rows: List[dict] = []

        for sp in anomaly_plan["spikes"]:
            rows.append({
                "anomaly_id": sp["anomaly_id"],
                "anomaly_type": "spike",
                "date_start": sp["date"],
                "date_end": sp["date"],
                "account_id": sp["account_id"],
                "account_name": sp["account_name"],
                "services_affected": sp["service"],
                "regions_affected": "",
                "magnitude": sp["magnitude"],
                "cascade_id": "",
            })

        for cs in anomaly_plan["cascades"]:
            rows.append({
                "anomaly_id": cs["anomaly_id"],
                "anomaly_type": "cascade",
                "date_start": cs["date"],
                "date_end": cs["date"],
                "account_id": cs["account_id"],
                "account_name": cs["account_name"],
                "services_affected": ";".join(cs["services"]),
                "regions_affected": "",
                "magnitude": "",
                "cascade_id": cs["cascade_id"],
            })

        for dr in anomaly_plan["drifts"]:
            rows.append({
                "anomaly_id": dr["anomaly_id"],
                "anomaly_type": "drift",
                "date_start": dr["start_date"],
                "date_end": dr["end_date"],
                "account_id": dr["account_id"],
                "account_name": dr["account_name"],
                "services_affected": dr["service"],
                "regions_affected": "",
                "magnitude": "",
                "cascade_id": "",
            })

        df = pd.DataFrame(rows, columns=ANOMALY_LOG_COLUMNS)
        df.to_csv(output_path, index=False)

        console_and_logger(
            self.__logger,
            f"Anomaly log saved: {output_path} ({len(df)} entries)",
        )

    def __plan_spikes(
            self,
            rng: np.random.RandomState,
            accounts: List[dict],
            services: List[str],
            all_dates: list,
        ) -> List[dict]:
        """
        Creates single-day cost spike anomalies. Each spike picks a random
        account, service, and date, then applies a magnitude multiplier
        with jitter (0.8x to 1.2x of the configured spike_magnitude).

        rng (np.random.RandomState): Seeded RNG for reproducibility.
        accounts (List[dict]): Available account metadata.
        services (List[str]): Available service codes.
        all_dates (list): Full range of candidate dates.

        Returns a list of spike definition dicts.
        """
        spikes: List[dict] = []
        magnitude = self.__config["spike_magnitude"]

        for _ in range(self.__config["num_spike_anomalies"]):
            acct = accounts[rng.randint(len(accounts))]
            svc = services[rng.randint(len(services))]
            day = all_dates[rng.randint(len(all_dates))]
            jitter = rng.uniform(0.8, 1.2) * magnitude

            spikes.append({
                "anomaly_id": uuid.uuid4().hex[:8],
                "date": day,
                "account_id": acct["account_id"],
                "account_name": acct["account_name"],
                "service": svc,
                "magnitude": round(jitter, 2),
            })

        return spikes

    def __plan_cascades(
            self,
            rng: np.random.RandomState,
            accounts: List[dict],
            all_dates: list,
        ) -> List[dict]:
        """
        Creates cascade anomalies where multiple related services spike
        on the same day for one account, simulating a real-world event
        where e.g. a sudden EC2 scale-up drives correlated S3 and
        DataTransfer cost increases. The affected services come from
        CASCADE_SERVICES in consts_handler.

        rng (np.random.RandomState): Seeded RNG.
        accounts (List[dict]): Available account metadata.
        all_dates (list): Full range of candidate dates.

        Returns a list of cascade definition dicts, each carrying a
        shared cascade_id that links the affected services together.
        """
        cascades: List[dict] = []

        for _ in range(self.__config["num_cascade_anomalies"]):
            acct = accounts[rng.randint(len(accounts))]
            day = all_dates[rng.randint(len(all_dates))]
            cascade_id = uuid.uuid4().hex[:8]

            cascades.append({
                "anomaly_id": uuid.uuid4().hex[:8],
                "cascade_id": cascade_id,
                "date": day,
                "account_id": acct["account_id"],
                "account_name": acct["account_name"],
                "services": list(CASCADE_SERVICES),
            })

        return cascades

    def __plan_drifts(
            self,
            rng: np.random.RandomState,
            accounts: List[dict],
            services: List[str],
            all_dates: list,
        ) -> List[dict]:
        """
        Creates multi-day cost drift anomalies where spending on a single
        service for one account creeps upward day over day. The drift
        duration is sampled from DRIFT_DURATION_RANGE and compounding is
        applied per-day during row generation.

        rng (np.random.RandomState): Seeded RNG.
        accounts (List[dict]): Available account metadata.
        services (List[str]): Available service codes.
        all_dates (list): Full range of candidate dates.

        Returns a list of drift definition dicts with start_date, end_date,
        and duration fields.
        """
        drifts: List[dict] = []

        for _ in range(self.__config["num_drift_anomalies"]):
            acct = accounts[rng.randint(len(accounts))]
            svc = services[rng.randint(len(services))]
            duration = rng.randint(
                DRIFT_DURATION_RANGE[0],
                DRIFT_DURATION_RANGE[1] + 1,
            )
            idx = rng.randint(max(1, len(all_dates) - duration))
            start = all_dates[idx]
            end = all_dates[min(idx + duration - 1, len(all_dates) - 1)]

            drifts.append({
                "anomaly_id": uuid.uuid4().hex[:8],
                "start_date": start,
                "end_date": end,
                "duration": duration,
                "account_id": acct["account_id"],
                "account_name": acct["account_name"],
                "service": svc,
            })

        return drifts

    def __build_lookup(
            self,
            spikes: List[dict],
            cascades: List[dict],
            drifts: List[dict],
        ) -> Dict[Tuple, dict]:
        """
        Converts the three anomaly lists into a single flat dict keyed
        by (date, account_id, service) so the row-level generator can
        check for anomalies in O(1) time.

        Spike entries store the raw magnitude multiplier. Cascade entries
        store a per-service magnitude_range (e.g. EC2 gets 2.0-3.0x,
        S3 gets 1.5-2.0x). Drift entries are expanded day-by-day from
        start_date to end_date, each carrying a day_num and the daily
        compound factor so the generator can compute
        compound ** day_num at row-build time.

        spikes (List[dict]): Spike definitions from __plan_spikes.
        cascades (List[dict]): Cascade definitions from __plan_cascades.
        drifts (List[dict]): Drift definitions from __plan_drifts.

        Returns the lookup dict mapping (date, account_id, service)
        tuples to anomaly metadata.
        """
        from src.utilities.consts_handler import (
            CASCADE_EC2_MULTIPLIER,
            CASCADE_S3_MULTIPLIER,
            CASCADE_DT_MULTIPLIER,
            DRIFT_DAILY_COMPOUND,
        )

        lookup: Dict[Tuple, dict] = {}

        # Spikes
        for sp in spikes:
            key = (sp["date"], sp["account_id"], sp["service"])
            lookup[key] = {
                "type": "spike",
                "magnitude": sp["magnitude"],
            }

        # Cascades
        cascade_mults = {
            "AmazonEC2": CASCADE_EC2_MULTIPLIER,
            "AmazonS3": CASCADE_S3_MULTIPLIER,
            "AWSDataTransfer": CASCADE_DT_MULTIPLIER,
        }
        for cs in cascades:
            for svc in cs["services"]:
                key = (cs["date"], cs["account_id"], svc)
                m_range = cascade_mults.get(svc, (1.5, 2.0))
                lookup[key] = {
                    "type": "cascade",
                    "cascade_id": cs["cascade_id"],
                    "magnitude_range": m_range,
                }

        # Drifts — expand each day
        for dr in drifts:
            d = dr["start_date"]
            day_num = 0
            while d <= dr["end_date"]:
                key = (d, dr["account_id"], dr["service"])
                if key not in lookup:
                    lookup[key] = {
                        "type": "drift",
                        "day_num": day_num,
                        "compound": DRIFT_DAILY_COMPOUND,
                    }
                d = d + datetime.timedelta(days=1)
                day_num += 1

        return lookup
