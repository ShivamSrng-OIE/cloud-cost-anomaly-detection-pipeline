import uuid
from typing import Dict, List

import numpy as np

from src.utilities.general_utils import console_and_logger
from src.utilities.consts_handler import (
    ACCOUNT_PROFILES,
    SERVICE_CONFIG,
)


class AccountBuilder:

    def __init__(
            self,
            logger,
            generation_config: dict,
        ) -> None:
        """
        Stores the logger and the merged generation config dict that
        drives account count, service list, region list, target rows,
        and seed values throughout the build process.

        logger: Pipeline logger for progress messages.
        generation_config (dict): Flat dict produced by Engine.__merge_config.
        """
        self.__logger = logger
        self.__config = generation_config

    def build_accounts(self) -> List[dict]:
        """
        Creates a list of synthetic AWS account records by pairing each
        profile from ACCOUNT_PROFILES with a randomly generated 12-digit
        account ID. The ID is built from two 6-digit random ints
        concatenated as strings to avoid numpy int32 overflow on
        12-digit numbers.

        Each returned dict contains:
            account_id (str): 12-digit numeric string.
            account_name (str): Human-readable label from the profile.
            cost_multiplier (float): Relative spend weight that scales
                all costs for this account.

        Returns a list of account metadata dicts, one per profile.
        """
        num = self.__config["num_accounts"]
        profiles = ACCOUNT_PROFILES[:num]
        rng = np.random.RandomState(self.__config["seed"])

        accounts: List[dict] = []
        for prof in profiles:
            part_a = str(rng.randint(100_000, 999_999))
            part_b = str(rng.randint(100_000, 999_999))
            accounts.append({
                "account_id": part_a + part_b,
                "account_name": prof["name"],
                "cost_multiplier": prof["cost_multiplier"],
            })

        console_and_logger(
            self.__logger,
            f"Built {len(accounts)} accounts: "
            + ", ".join(a["account_name"] for a in accounts),
        )
        return accounts

    def build_resource_pool(
            self,
            accounts: List[dict],
        ) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """
        Pre-generates a fixed pool of ARN-style resource identifiers for
        every (account, service, region) combination. The same pool is
        reused across all billing months so that resource IDs stay
        consistent in the final dataset.

        The returned structure is a nested dict:
            pool[account_id][service][region] -> list of ARN strings.

        accounts (List[dict]): Account metadata dicts from build_accounts().

        Returns the nested resource ARN pool dict.
        """
        services = self.__config["services"]
        regions = self.__config["regions"]
        resources_per_combo = self.__compute_resources_per_combo(
            num_accounts=len(accounts),
        )

        pool: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

        for acct in accounts:
            aid = acct["account_id"]
            pool[aid] = {}

            for svc in services:
                svc_cfg = SERVICE_CONFIG.get(svc)
                if svc_cfg is None:
                    continue

                pool[aid][svc] = {}
                res_type = svc_cfg["resource_type"]

                for region in regions:
                    arns: List[str] = []
                    for _ in range(resources_per_combo):
                        rid = uuid.uuid4().hex[:12]
                        arn = f"arn:aws:{svc.lower()}:{region}:{aid}:{res_type}/{rid}"
                        arns.append(arn)
                    pool[aid][svc][region] = arns

        total_arns = sum(
            len(v)
            for a in pool.values()
            for s in a.values()
            for v in s.values()
        )
        console_and_logger(
            self.__logger,
            f"Resources per (account, service, region): {resources_per_combo}",
        )
        console_and_logger(
            self.__logger,
            f"Total resource ARNs generated: {total_arns:,}",
        )
        return pool

    def __compute_resources_per_combo(
            self,
            num_accounts: int,
        ) -> int:
        """
        Works backwards from the target row count to figure out how many
        resource ARNs each (account, service, region) combination needs.
        The raw CUR emits one row per resource per day, so:
            resources = target_rows / (accounts * services * regions * total_days)
        Result is clamped to a minimum of 1.

        num_accounts (int): Number of accounts in this run.

        Returns the integer resource count per combination.
        """
        n_services = len(self.__config["services"])
        n_regions = len(self.__config["regions"])
        n_months = self.__config["num_months"]
        target = self.__config["target_rows"]

        avg_days_per_month = 30
        combos = num_accounts * n_services * n_regions
        total_days = avg_days_per_month * n_months

        return max(1, round(target / (combos * total_days)))
