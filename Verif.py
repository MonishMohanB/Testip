import os
import json
import yaml
import pandas as pd
import re
import ast
import logging
from typing import Union, Dict

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

VALID_MODEL_TYPES = ["xgboost", "random_forest", "linear_regression"]
LIST_COLUMNS = [
    "time_series_vars", "categorical_vars", "scenario_var",
    "optional_test", "optional_performance_tests",
    "omit_default_diagnostic_tests", "omit_performance_tests"
]
FIXED_COLUMNS = [
    "model_type", "sub_model_name", "training_data", "validation_data",
    "input_data_format", "time_var"
] + LIST_COLUMNS

def load_model_config(config_input: Union[str, Dict]) -> Dict:
    """
    Load the base model config from dict, YAML or JSON file.
    
    Args:
        config_input (Union[str, Dict]): Path to config file or dict.

    Returns:
        Dict: Parsed model configuration.

    Raises:
        FileNotFoundError: If config path doesn't exist.
        ValueError: If file format is not supported.
    """
    logger.info("Loading model configuration...")
    if isinstance(config_input, dict):
        return config_input
    if not os.path.exists(config_input):
        raise FileNotFoundError(f"Config file not found: {config_input}")
    with open(config_input, "r") as f:
        if config_input.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        elif config_input.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format")

def validate_model_config(config: Dict):
    """
    Validate essential keys in the base model config.

    Args:
        config (Dict): Model config dictionary.

    Raises:
        KeyError, ValueError, FileNotFoundError
    """
    logger.info("Validating model configuration...")
    required_keys = ["model_id", "submission_id", "model_name", "base_path"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key: {key}")
    if not str(config["model_id"]).isdigit():
        raise ValueError("model_id must be numeric")
    if not str(config["submission_id"]).isdigit():
        raise ValueError("submission_id must be numeric")
    if not re.match(r"^[a-zA-Z0-9_]+$", config["model_name"]):
        raise ValueError("model_name must contain only letters, numbers, and underscores")
    if not os.path.exists(config["base_path"]):
        raise FileNotFoundError(f"base_path does not exist: {config['base_path']}")

def load_tabular_input(path: str) -> pd.DataFrame:
    """
    Load tabular input file.

    Args:
        path (str): Path to input file.

    Returns:
        pd.DataFrame: Parsed DataFrame.

    Raises:
        FileNotFoundError, ValueError
    """
    logger.info(f"Loading tabular input from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json"):
        return pd.read_json(path)
    elif path.endswith((".yaml", ".yml")):
        with open(path, "r") as f:
            return pd.DataFrame(yaml.safe_load(f))
    else:
        raise ValueError("Unsupported tabular input format")

def validate_row(row: pd.Series, index: int):
    """
    Validate row-level fields from tabular input.

    Args:
        row (pd.Series): Input row.
        index (int): Row index.

    Raises:
        ValueError: If any field is invalid.
    """
    logger.debug(f"Validating row {index}")
    name = row.get("sub_model_name", "")
    if not re.match(r"^[a-zA-Z0-9_]+$", str(name)):
        raise ValueError(f"Invalid sub_model_name at row {index}")
    model_type = row.get("model_type")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model_type '{model_type}' at row {index}")

def parse_list_column(value, index, colname):
    """
    Parse a column value into a list if applicable.

    Args:
        value: Value to parse.
        index (int): Row index.
        colname (str): Column name.

    Returns:
        list: Parsed list.

    Raises:
        ValueError
    """
    if pd.isna(value) or value == '':
        return []
    if isinstance(value, list):
        return value
    try:
        value = str(value).strip()
        if value.startswith("[") and value.endswith("]"):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        elif "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]
        else:
            return [value]
    except Exception:
        raise ValueError(f"Invalid list format in column '{colname}' at row {index}: {value}")

def load_data(data_format: str, path: str, source: str, repartition: int = 200) -> pd.DataFrame:
    """
    Load training or validation data.

    Args:
        data_format (str): Format (path/hdfs/hive).
        path (str): Data path.
        source (str): training/validation
        repartition (int): Optional Spark partitioning.

    Returns:
        pd.DataFrame
    """
    logger.debug(f"Loading {source} data with format '{data_format}' from {path}")
    if data_format == "path":
        return pd.read_csv(path)
    elif data_format == "hdfs":
        logger.debug(f"Simulating spark.read.parquet().repartition({repartition}) for HDFS")
        return pd.DataFrame()  # Placeholder
    elif data_format == "hive":
        logger.debug("Simulating spark.read.table() for Hive")
        return pd.DataFrame()  # Placeholder
    else:
        raise ValueError(f"Unsupported input_data_format: {data_format}")

def validate_columns_exist(row, train_df, valid_df, index):
    """
    Validate that time-related and list-based columns exist in dataframes.

    Args:
        row (pd.Series)
        train_df (pd.DataFrame)
        valid_df (pd.DataFrame)
        index (int)

    Returns:
        Dict: Validated values to be added to model_setting.
    """
    logger.debug(f"Validating column presence for row {index}")
    results = {}

    time_var = row.get("time_var")
    if pd.notna(time_var) and time_var != '':
        time_var = str(time_var).strip()
        if time_var not in train_df.columns:
            raise ValueError(f"time_var '{time_var}' missing in training_data at row {index}")
        if time_var not in valid_df.columns:
            raise ValueError(f"time_var '{time_var}' missing in validation_data at row {index}")
        results["time_var"] = time_var

    for col in LIST_COLUMNS:
        val = row.get(col)
        parsed = parse_list_column(val, index, col)
        for v in parsed:
            if v not in train_df.columns:
                raise ValueError(f"{col} value '{v}' missing in training_data at row {index}")
            if v not in valid_df.columns:
                raise ValueError(f"{col} value '{v}' missing in validation_data at row {index}")
        if parsed:
            results[col] = parsed

    return results

def process_model(config: Dict):
    """
    Placeholder for user-defined model processing logic.

    Args:
        config (Dict): Fully merged config.
    """
    logger.info(f"Model '{config.get('sub_model_name')}' processed successfully.")

def main(model_config_input: Union[str, Dict], tabular_input_path: str):
    """
    Main controller: merges tabular rows with model config and processes them.

    Args:
        model_config_input (str|Dict): Model base config.
        tabular_input_path (str): Path to tabular input (CSV, JSON, YAML).
    """
    try:
        base_config = load_model_config(model_config_input)
        validate_model_config(base_config)
        df = load_tabular_input(tabular_input_path)

        for idx, row in df.iterrows():
            try:
                logger.info(f"\n--- Processing row {idx} ---")
                validate_row(row, idx)
                row_data = row.dropna().to_dict()

                input_data_format = row_data.get("input_data_format") or base_config.get("input_data_format")
                if not input_data_format:
                    raise ValueError(f"Missing input_data_format for row {idx}")

                # Load datasets
                training_df = load_data(input_data_format, row_data.get("training_data", ""), "training", row_data.get("repartition", 200))
                validation_df = load_data(input_data_format, row_data.get("validation_data", ""), "validation", row_data.get("repartition", 200))

                validated_model_settings = validate_columns_exist(row, training_df, validation_df, idx)

                merged_config = base_config.copy()
                model_setting = validated_model_settings.copy()

                # Set required fields explicitly
                for key in ["model_type", "sub_model_name"]:
                    if key in row_data:
                        merged_config[key] = row_data[key]

                # Dynamically add all other custom keys to model_setting
                for key, val in row_data.items():
                    if key not in FIXED_COLUMNS and key not in LIST_COLUMNS:
                        if val not in [None, "", [], {}]:
                            model_setting[key] = val

                if model_setting:
                    merged_config["model_setting"] = model_setting

                merged_config["training_data_df"] = training_df
                merged_config["validation_data_df"] = validation_df

                process_model(merged_config)

            except Exception as row_err:
                logger.error(f"Failed to process row {idx}: {row_err}", exc_info=True)

    except Exception as err:
        logger.critical(f"Fatal error during execution: {err}", exc_info=True)

