import logging
import os
import json
import yaml
import pandas as pd
import re
import ast
from typing import Union, Dict

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

VALID_MODEL_TYPES = ["xgboost", "random_forest", "linear_regression"]
LIST_COLUMNS = [
    "time_series_vars", "categorical_vars", "scenario_var",
    "optional_test", "optional_performance_tests",
    "omit_default_diagnostic_tests", "omit_performance_tests"
]


def load_model_config(config_input: Union[str, Dict]) -> Dict:
    """Load the model configuration from a file or dictionary."""
    logger.info("Loading model configuration...")
    if isinstance(config_input, dict):
        logger.debug("Config input is a dictionary")
        return config_input
    if not os.path.exists(config_input):
        raise FileNotFoundError(f"Config file not found: {config_input}")
    with open(config_input, "r") as f:
        if config_input.endswith(".yaml") or config_input.endswith(".yml"):
            logger.debug("Parsing YAML config")
            return yaml.safe_load(f)
        elif config_input.endswith(".json"):
            logger.debug("Parsing JSON config")
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format")


def validate_model_config(config: Dict):
    """Validate required fields in the base model config."""
    logger.info("Validating model configuration...")
    for key in ["model_id", "submission_id", "model_name", "base_path"]:
        if key not in config:
            raise KeyError(f"Missing required key: {key}")

    if not str(config["model_id"]).isdigit():
        raise ValueError("model_id should be numeric")
    if not str(config["submission_id"]).isdigit():
        raise ValueError("submission_id should be numeric")
    if not re.match(r"^[a-zA-Z0-9_]+$", config["model_name"]):
        raise ValueError("model_name must contain only letters, numbers, and underscores")
    if not os.path.exists(config["base_path"]):
        raise FileNotFoundError(f"base_path does not exist: {config['base_path']}")


def validate_row(row: pd.Series, index: int):
    """Validate individual row inputs."""
    logger.debug(f"Validating row {index}...")
    name = row.get("sub_model_name", "")
    if not re.match(r"^[a-zA-Z0-9_]+$", name):
        raise ValueError(f"Invalid sub_model_name at row {index}")

    model_type = row.get("model_type")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model_type '{model_type}' at row {index}. Must be one of {VALID_MODEL_TYPES}")


def parse_list_column(value, index, colname):
    """Safely parse string or list into list."""
    try:
        if pd.isna(value) or value == '':
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                parsed = ast.literal_eval(value)
                if not isinstance(parsed, list):
                    raise ValueError
                return parsed
            if "," in value:
                return [v.strip() for v in value.split(",") if v.strip()]
            return [value]
        raise ValueError
    except Exception:
        raise ValueError(f"Invalid list format in column '{colname}' at row {index}: {value}")


def validate_columns_exist(row, train_df, valid_df, index):
    """Validate presence of time/list columns in training/validation data."""
    logger.debug(f"Validating column presence for row {index}...")
    time_var = row.get("time_var")
    if pd.isna(time_var) or time_var == '':
        time_var = None
    else:
        time_var = str(time_var).strip()

    result = {}

    for colname in LIST_COLUMNS:
        result[colname] = parse_list_column(row.get(colname), index, colname)

    if time_var:
        if time_var not in train_df.columns:
            raise ValueError(f"Time var '{time_var}' not found in training data at row {index}")
        if time_var not in valid_df.columns:
            raise ValueError(f"Time var '{time_var}' not found in validation data at row {index}")

    for colname in ["time_series_vars", "categorical_vars", "scenario_var"]:
        for col in result[colname]:
            if col not in train_df.columns:
                raise ValueError(f"Column '{col}' not found in training_data at row {index}")
            if col not in valid_df.columns:
                raise ValueError(f"Column '{col}' not found in validation_data at row {index}")

    if time_var:
        result["time_var"] = time_var

    return result


def load_tabular_input(path: str) -> pd.DataFrame:
    """Load tabular input from CSV, JSON, Excel, or YAML."""
    logger.info(f"Loading tabular input from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tabular input file not found: {path}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    elif path.endswith(".json"):
        return pd.read_json(path)
    elif path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            return pd.DataFrame(yaml.safe_load(f))
    else:
        raise ValueError("Unsupported tabular input format")


def load_data(data_type: str, path: str) -> pd.DataFrame:
    """Load training/validation data from path, HDFS, or Hive."""
    logger.debug(f"Loading data from type: {data_type}, path: {path}")
    if data_type == "path":
        return pd.read_csv(path)
    elif data_type == "hdfs":
        return pd.DataFrame()  # simulate spark.read for HDFS
    elif data_type == "hive":
        return pd.DataFrame()  # simulate spark.read for Hive
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def process_model(config: Dict):
    """Stub for model processing logic."""
    logger.info(f"Processing model: {config.get('sub_model_name')}")


def main(model_config_input: Union[str, Dict], tabular_input_path: str):
    """Main controller function for validating and merging configs and invoking processing."""
    try:
        config = load_model_config(model_config_input)
        validate_model_config(config)
        df = load_tabular_input(tabular_input_path)

        for idx, row in df.iterrows():
            try:
                logger.info(f"\n--- Processing row {idx} ---")
                validate_row(row, idx)
                row_data = row.dropna().to_dict()

                training_df = load_data(row_data['training_data_type'], row_data['training_data'])
                validation_df = load_data(row_data['validation_data_type'], row_data['validation_data'])

                validated_optional = validate_columns_exist(row, training_df, validation_df, idx)

                merged_config = config.copy()

                for key, value in row_data.items():
                    if value is not None and pd.notna(value) and key not in [
                        'training_data_type', 'training_data', 'validation_data_type', 'validation_data',
                        'time_var', 'time_series_vars', 'categorical_vars', 'scenario_var'
                    ] + LIST_COLUMNS:
                        merged_config[key] = value

                # Build model_setting and clean empty list fields
                model_setting = {
                    k: v for k, v in validated_optional.items()
                    if v is not None and (not isinstance(v, list) or len(v) > 0)
                }

                if model_setting:
                    merged_config['model_setting'] = model_setting

                merged_config['training_data_df'] = training_df
                merged_config['validation_data_df'] = validation_df

                logger.debug(f"Merged config ready for model: {merged_config.get('sub_model_name')}")
                process_model(merged_config)

            except Exception as row_err:
                logger.error(f"Row {idx} processing failed: {row_err}", exc_info=True)

    except Exception as e:
        logger.critical(f"Fatal error during processing: {e}", exc_info=True)



CSV: input.csv

JSON: input.json

YAML: input.yaml
📄 Download test_model_processor.py
