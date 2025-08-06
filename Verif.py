import logging
import os
import json
import yaml
import pandas as pd
import re
import ast
from typing import Union, Dict
from pyspark.sql import SparkSession

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Spark session
spark = SparkSession.builder.appName("ModelProcessor").getOrCreate()

# Constants
VALID_MODEL_TYPES = ["xgboost", "random_forest", "linear_regression"]
LIST_COLUMNS = [
    "time_series_vars", "categorical_vars", "scenario_var",
    "optional_test", "optional_performance_tests",
    "omit_default_diagnostic_tests", "omit_performance_tests"
]
OVERRIDABLE_COLUMNS = [
    "training_data", "training_data_type",
    "validation_data", "validation_data_type",
    "time_var"
] + LIST_COLUMNS


def load_model_config(config_input: Union[str, Dict]) -> Dict:
    """
    Load the base model configuration.

    Args:
        config_input: Path to JSON/YAML file or a dictionary.

    Returns:
        Parsed model config as dictionary.

    Raises:
        FileNotFoundError, ValueError
    """
    log
  
