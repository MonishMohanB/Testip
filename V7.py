import os
import json
import yaml
import pandas as pd
import logging
import re
import ast
from typing import Union, Dict

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]

VALID_MODEL_TYPES = ["xgboost", "random_forest", "linear_regression"]

def load_model_config(config_input: Union[str, Dict]) -> Dict:
    """
    Load model configuration from a dictionary or a JSON/YAML file.

    Args:
        config_input (Union[str, Dict]): File path or dictionary containing model config.

    Returns:
        Dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    logger.debug("Loading model configuration...")
    if isinstance(config_input, dict):
        logger.debug("Config provided as dictionary.")
        return config_input
    if not os.path.exists(config_input):
        raise FileNotFoundError(f"Config file not found: {config_input}")
    with open(config_input, "r") as f:
        if config_input.endswith(".yaml") or config_input.endswith(".yml"):
            logger.debug("Parsing YAML config file.")
            return yaml.safe_load(f)
        elif config_input.endswith(".json"):
            logger.debug("Parsing JSON config file.")
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format")

def validate_model_config(config: Dict):
    """
    Validates required keys and formats in the model config.

    Args:
        config (Dict): Model configuration.

    Raises:
        KeyError: If required keys are missing.
        ValueError: If key values are in invalid formats.
        FileNotFoundError: If the base_path does not exist.
    """
    logger.debug("Validating model configuration...")
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
    """
    Validates sub_model row for naming and model_type.

    Args:
        row (pd.Series): Row of the tabular input.
        index (int): Row index.

    Raises:
        ValueError: If values are not in valid format.
    """
    logger.debug(f"Validating row {index} for naming and model_type...")
    name = row.get("sub_model_name", "")
    if not re.match(r"^[a-zA-Z0-9_]+$", name):
        raise ValueError(f"Invalid sub_model_name at row {index}")
    model_type = row.get("model_type")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model_type '{model_type}' at row {index}. Must be one of {VALID_MODEL_TYPES}")

def parse_list_column(value, index, colname):
    """
    Parses a string or list value into a list.

    Args:
        value: The value to parse.
        index (int): Row index.
        colname (str): Column name.

    Returns:
        List: Parsed list.

    Raises:
        ValueError: If value cannot be parsed as list.
    """
    logger.debug(f"Parsing column '{colname}' at row {index}...")
    try:
        if pd.isna(value) or value == '':
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return ast.literal_eval(value)
        raise ValueError
    except Exception:
        raise ValueError(f"Invalid list format in column '{colname}' at row {index}: {value}")

def validate_columns_exist(row, train_df, valid_df, index):
    """
    Validates the presence of required columns in train and validation DataFrames.

    Args:
        row (pd.Series): Input row.
        train_df (pd.DataFrame): Training data.
        valid_df (pd.DataFrame): Validation data.
        index (int): Row index.

    Returns:
        Dict: Dictionary of validated optional fields.
    """
    logger.debug(f"Validating required columns in training and validation data for row {index}...")
    time_var = row.get("time_var")
    config_opts = {
        "time_var": time_var,
        "time_series_vars": parse_list_column(row.get("time_series_vars"), index, "time_series_vars"),
        "categorical_vars": parse_list_column(row.get("categorical_vars"), index, "categorical_vars"),
        "scenario_var": parse_list_column(row.get("scenario_var"), index, "scenario_var"),
        "optional_test": parse_list_column(row.get("optional_test"), index, "optional_test"),
        "optional_performance_tests": parse_list_column(row.get("optional_performance_tests"), index, "optional_performance_tests"),
        "omit_default_diagnostic_tests": parse_list_column(row.get("omit_default_diagnostic_tests"), index, "omit_default_diagnostic_tests"),
        "omit_performance_tests": parse_list_column(row.get("omit_performance_tests"), index, "omit_performance_tests"),
    }

    for col_list, df, df_name in [
        (config_opts["time_series_vars"], train_df, "training_data"),
        (config_opts["categorical_vars"], train_df, "training_data"),
        (config_opts["scenario_var"], train_df, "training_data"),
        (config_opts["time_series_vars"], valid_df, "validation_data"),
        (config_opts["categorical_vars"], valid_df, "validation_data"),
        (config_opts["scenario_var"], valid_df, "validation_data")
    ]:
        for col in col_list:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {df_name} at row {index}")

    if pd.notna(time_var):
        if time_var not in train_df.columns:
            raise ValueError(f"Time var '{time_var}' not found in training data at row {index}")
        if time_var not in valid_df.columns:
            raise ValueError(f"Time var '{time_var}' not found in validation data at row {index}")

    # Filter out empty lists or None
    model_setting = {k: v for k, v in config_opts.items() if v and (not isinstance(v, list) or len(v) > 0)}
    return model_setting

def load_tabular_input(path: str) -> pd.DataFrame:
    """
    Load tabular input file into DataFrame.

    Args:
        path (str): Path to input file.

    Returns:
        pd.DataFrame: Parsed tabular data.

    Raises:
        FileNotFoundError: If file not found.
        ValueError: If unsupported file type.
    """
    logger.debug(f"Loading tabular input from: {path}")
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
    """
    Load training/validation data from a given source.

    Args:
        data_type (str): One of 'path', 'hdfs', or 'hive'.
        path (str): Source path or identifier.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        ValueError: For unsupported data type.
    """
    logger.debug(f"Loading data from type: {data_type} at path: {path}")
    if data_type == "path":
        return pd.read_csv(path)
    elif data_type in ["hdfs", "hive"]:
        logger.debug(f"Simulating {data_type} read...")
        return pd.DataFrame()  # Placeholder
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def process_model(config: Dict):
    """
    Process the merged model configuration.

    Args:
        config (Dict): Fully prepared config.
    """
    logger.info(f"Processing model: {config.get('sub_model_name')}")

def main(model_config_input: Union[str, Dict], tabular_input_path: str):
    """
    Main processing function to merge configs and process models.

    Args:
        model_config_input (Union[str, Dict]): File path or dict for model config.
        tabular_input_path (str): Path to tabular input file.
    """
    try:
        logger.info("Starting main processing...")
        config = load_model_config(model_config_input)
        validate_model_config(config)
        df = load_tabular_input(tabular_input_path)

        for idx, row in df.iterrows():
            try:
                logger.info(f"Processing row {idx}...")
                validate_row(row, idx)
                row_data = row.dropna().to_dict()

                training_df = load_data(row_data['training_data_type'], row_data['training_data'])
                validation_df = load_data(row_data['validation_data_type'], row_data['validation_data'])

                validated_optional = validate_columns_exist(row, training_df, validation_df, idx)

                merged_config = config.copy()

                # Add row data except excluded keys
                for key, value in row_data.items():
                    if value is not None and pd.notna(value) and key not in [
                        "training_data_type", "training_data",
                        "validation_data_type", "validation_data",
                        "time_var", "time_series_vars", "categorical_vars", "scenario_var",
                        "optional_test", "optional_performance_tests",
                        "omit_default_diagnostic_tests", "omit_performance_tests"
                    ]:
                        merged_config[key] = value

                # Add model settings
                if validated_optional:
                    merged_config['model_setting'] = validated_optional

                # Attach training and validation DataFrames
                merged_config['training_data_df'] = training_df
                merged_config['validation_data_df'] = validation_df

                logger.debug(f"Merged config for row {idx}: {merged_config}")
                process_model(merged_config)

            except Exception as row_err:
                logger.error(f"Row {idx} processing failed: {row_err}", exc_info=True)

    except Exception as e:
        logger.critical(f"Fatal error during processing: {e}", exc_info=True)
