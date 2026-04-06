import os
import yaml
from src.utilities.log_handler import LogHandler
from src.utilities.general_utils import console_and_logger


REGION_PREFIX_MAP = {
    "us-east-1": "USE1",
    "us-west-2": "USW2",
    "eu-west-1": "EUW1",
    "ap-southeast-1": "APS1",
    "us-east-2": "USE2",
    "eu-central-1": "EUC1",
    "ap-northeast-1": "APN1",
}

SERVICE_CONFIG = {
    "AmazonEC2": {
        "product_family": "Compute Instance",
        "product_name": "Amazon Elastic Compute Cloud",
        "pricing_unit": "Hrs",
        "usage_types": {
            "BoxUsage:m5.xlarge":  {"rate": 0.192, "instance_type": "m5.xlarge"},
            "BoxUsage:c5.2xlarge": {"rate": 0.34,  "instance_type": "c5.2xlarge"},
            "BoxUsage:r5.large":   {"rate": 0.126, "instance_type": "r5.large"},
        },
        "operations": ["RunInstances", "RunInstances:0002"],
        "resource_type": "instance",
        "description_template": "${rate} per On Demand Linux {instance_type} Instance Hour",
        "cost_share": (0.40, 0.50),
        "weekday_multiplier": (1.5, 2.5),
    },
    "AmazonS3": {
        "product_family": "Storage",
        "product_name": "Amazon Simple Storage Service",
        "pricing_unit": "GB-Mo",
        "usage_types": {
            "TimedStorage-ByteHrs":   {"rate": 0.023,  "instance_type": ""},
            "Requests-Tier1":         {"rate": 0.0004, "instance_type": ""},
            "DataTransfer-Out-Bytes": {"rate": 0.09,   "instance_type": ""},
        },
        "operations": ["GetObject", "PutObject", "ListBucket"],
        "resource_type": "bucket",
        "description_template": "${rate} per GB - first 50 TB / month of storage used",
        "cost_share": (0.10, 0.15),
        "weekday_multiplier": (1.0, 1.1),
    },
    "AmazonRDS": {
        "product_family": "Database Instance",
        "product_name": "Amazon Relational Database Service",
        "pricing_unit": "Hrs",
        "usage_types": {
            "InstanceUsage:db.r5.large":  {"rate": 0.24,  "instance_type": "db.r5.large"},
            "InstanceUsage:db.m5.xlarge": {"rate": 0.342, "instance_type": "db.m5.xlarge"},
        },
        "operations": ["CreateDBInstance"],
        "resource_type": "db",
        "description_template": "${rate} per RDS {instance_type} Multi-AZ instance hour",
        "cost_share": (0.15, 0.20),
        "weekday_multiplier": (1.5, 2.5),
    },
    "AWSLambda": {
        "product_family": "Serverless",
        "product_name": "AWS Lambda",
        "pricing_unit": "GB-Second",
        "usage_types": {
            "Lambda-GB-Second": {"rate": 0.0000166667, "instance_type": ""},
            "Request":          {"rate": 0.0000002,    "instance_type": ""},
        },
        "operations": ["Invoke"],
        "resource_type": "function",
        "description_template": "${rate} per GB-Second for Lambda Duration",
        "cost_share": (0.03, 0.05),
        "weekday_multiplier": (1.5, 2.5),
    },
    "AmazonDynamoDB": {
        "product_family": "Database",
        "product_name": "Amazon DynamoDB",
        "pricing_unit": "Hrs",
        "usage_types": {
            "ReadCapacityUnit-Hrs":  {"rate": 0.00065, "instance_type": ""},
            "WriteCapacityUnit-Hrs": {"rate": 0.00065, "instance_type": ""},
        },
        "operations": ["GetItem", "PutItem", "Query"],
        "resource_type": "table",
        "description_template": "${rate} per hour for DynamoDB Provisioned Capacity",
        "cost_share": (0.02, 0.04),
        "weekday_multiplier": (1.0, 1.1),
    },
    "AmazonSageMaker": {
        "product_family": "Machine Learning",
        "product_name": "Amazon SageMaker",
        "pricing_unit": "Hrs",
        "usage_types": {
            "ML.m5.xlarge":  {"rate": 0.23,  "instance_type": "ml.m5.xlarge"},
            "ML.p3.2xlarge": {"rate": 3.825, "instance_type": "ml.p3.2xlarge"},
        },
        "operations": ["CreateTrainingJob", "CreateEndpoint"],
        "resource_type": "notebook-instance",
        "description_template": "${rate} per {instance_type} SageMaker instance hour",
        "cost_share": (0.05, 0.10),
        "weekday_multiplier": (1.5, 2.5),
    },
    "AmazonElastiCache": {
        "product_family": "In-Memory Cache",
        "product_name": "Amazon ElastiCache",
        "pricing_unit": "Hrs",
        "usage_types": {
            "NodeUsage:cache.r5.large": {"rate": 0.166, "instance_type": "cache.r5.large"},
        },
        "operations": ["CreateCacheCluster"],
        "resource_type": "cluster",
        "description_template": "${rate} per ElastiCache {instance_type} node hour",
        "cost_share": (0.02, 0.04),
        "weekday_multiplier": (1.5, 2.5),
    },
    "AWSDataTransfer": {
        "product_family": "Data Transfer",
        "product_name": "AWS Data Transfer",
        "pricing_unit": "GB",
        "usage_types": {
            "DataTransfer-Out-Bytes":      {"rate": 0.09, "instance_type": ""},
            "DataTransfer-Regional-Bytes": {"rate": 0.01, "instance_type": ""},
        },
        "operations": ["RunInstances"],
        "resource_type": "transfer",
        "description_template": "${rate} per GB - data transfer out",
        "cost_share": (0.08, 0.12),
        "weekday_multiplier": (1.0, 1.1),
    },
}

ACCOUNT_PROFILES = [
    {"name": "prod-main",   "cost_multiplier": 10.0},
    {"name": "prod-ml",     "cost_multiplier": 6.0},
    {"name": "staging",     "cost_multiplier": 3.0},
    {"name": "dev",         "cost_multiplier": 1.5},
    {"name": "sandbox",     "cost_multiplier": 1.0},
    {"name": "analytics",   "cost_multiplier": 4.0},
    {"name": "data-lake",   "cost_multiplier": 5.0},
    {"name": "security",    "cost_multiplier": 2.0},
    {"name": "shared-svcs", "cost_multiplier": 3.5},
    {"name": "dr-backup",   "cost_multiplier": 2.5},
]

TEAMS = [
    "ml-platform", "backend", "data-engineering", "frontend",
    "devops", "analytics", "infrastructure", "mobile",
]

ENVIRONMENTS = ["production", "staging", "development", "testing"]

TAG_MISSING_RATE_TEAM = 0.35
TAG_MISSING_RATE_ENVIRONMENT = 0.25

LINE_ITEM_TYPES = {
    "Usage": 0.85,
    "Tax": 0.05,
    "SavingsPlanCoveredUsage": 0.08,
    "Credit": 0.02,
}

SAVINGS_PLAN_DISCOUNT = 0.70
CREDIT_FRACTION = -0.02
TAX_FRACTION = 0.08

MONTHLY_GROWTH_RATE = (0.02, 0.04)
DAILY_NOISE_STD = (0.05, 0.10)

MONTH_START_SPIKE_DAYS = 3
MONTH_START_SPIKE_MULTIPLIER = (1.3, 1.5)
MONTH_END_SPIKE_DAYS = 2
MONTH_END_SPIKE_MULTIPLIER = (1.2, 1.4)
MONTH_START_SPIKE_SERVICES = {"AmazonS3", "AWSDataTransfer"}
MONTH_END_SPIKE_SERVICES = {"AmazonSageMaker"}

CASCADE_EC2_MULTIPLIER = (2.0, 3.0)
CASCADE_S3_MULTIPLIER = (1.5, 2.0)
CASCADE_DT_MULTIPLIER = (2.0, 3.0)
CASCADE_SERVICES = ["AmazonEC2", "AmazonS3", "AWSDataTransfer"]

DRIFT_DAILY_COMPOUND = 1.05
DRIFT_DURATION_RANGE = (5, 14)
BLENDED_RATE_FACTOR = (0.92, 1.0)

RAW_CUR_COLUMNS = [
    "identity/LineItemId",
    "bill/BillingPeriodStartDate",
    "bill/BillingPeriodEndDate",
    "bill/PayerAccountId",
    "lineItem/UsageAccountId",
    "lineItem/LineItemType",
    "lineItem/UsageStartDate",
    "lineItem/UsageEndDate",
    "lineItem/ProductCode",
    "lineItem/UsageType",
    "lineItem/Operation",
    "lineItem/ResourceId",
    "lineItem/UsageAmount",
    "lineItem/UnblendedRate",
    "lineItem/UnblendedCost",
    "lineItem/BlendedRate",
    "lineItem/BlendedCost",
    "lineItem/LineItemDescription",
    "lineItem/CurrencyCode",
    "product/region",
    "product/instanceType",
    "product/productFamily",
    "product/productName",
    "pricing/unit",
    "resourceTags/user:team",
    "resourceTags/user:environment",
]

AGG_COLUMNS = [
    "date", "account_id", "account_name", "service", "region",
    "daily_cost", "daily_usage", "num_resources",
    "day_of_week", "day_name", "is_weekend", "day_of_month",
    "week_of_year", "month",
    "cost_7d_rolling_avg", "cost_pct_change_vs_7d_avg",
    "cost_28d_rolling_avg", "cost_pct_change_vs_28d_avg",
    "is_anomaly", "anomaly_type", "cascade_id",
]

ANOMALY_LOG_COLUMNS = [
    "anomaly_id", "anomaly_type", "date_start", "date_end",
    "account_id", "account_name", "services_affected", "regions_affected",
    "magnitude", "cascade_id",
]


def _load_yaml_section(logger, section_name: str) -> dict:
    """
    Opens config.yaml from the project root, parses it, and returns
    the dict for the requested top-level section. Raises FileNotFoundError
    if the file is missing, or ValueError if the section does not exist.

    logger: Active logger instance for error reporting.
    section_name (str): The YAML key to extract, e.g. "generation".

    Returns the parsed dict for that section.
    """
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        console_and_logger(
            logger,
            f"Configuration file not found at {config_path}",
            level="error",
        )
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config or section_name not in config:
        raise ValueError(
            f"Invalid config file: missing '{section_name}' section"
        )

    return config[section_name]


class GenerationConsts:
    """Loads and exposes the 'generation' section from config.yaml."""

    def __init__(self, logger: LogHandler) -> None:
        """
        Reads the 'generation' section from config.yaml on construction.
        The parsed dict is cached internally and served via get_config().

        logger (LogHandler): Logger used to report load progress.
        """
        self.__logger = logger
        console_and_logger(self.__logger, "Initializing GenerationConsts")
        self.__config = _load_yaml_section(logger, "generation")
        console_and_logger(
            self.__logger,
            "Generation configuration loaded successfully.",
        )

    def get_config(self) -> dict:
        return self.__config


class AnomalyConsts:
    """Loads and exposes the 'anomalies' section from config.yaml."""

    def __init__(self, logger: LogHandler) -> None:
        """
        Reads the 'anomalies' section from config.yaml on construction.
        The parsed dict is cached internally and served via get_config().

        logger (LogHandler): Logger used to report load progress.
        """
        self.__logger = logger
        console_and_logger(self.__logger, "Initializing AnomalyConsts")
        self.__config = _load_yaml_section(logger, "anomalies")
        console_and_logger(
            self.__logger,
            "Anomaly configuration loaded successfully.",
        )

    def get_config(self) -> dict:
        return self.__config


class OutputConsts:
    """Loads and exposes the 'output' section from config.yaml."""

    def __init__(self, logger: LogHandler) -> None:
        """
        Reads the 'output' section from config.yaml on construction.
        The parsed dict is cached internally and served via get_config().

        logger (LogHandler): Logger used to report load progress.
        """
        self.__logger = logger
        console_and_logger(self.__logger, "Initializing OutputConsts")
        self.__config = _load_yaml_section(logger, "output")
        console_and_logger(
            self.__logger,
            "Output configuration loaded successfully.",
        )

    def get_config(self) -> dict:
        return self.__config
