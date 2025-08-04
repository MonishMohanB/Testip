import os
import re
import json
import yaml
import ast
import logging
import pandas as pd
from typing import Union, Dict, Any

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

VALID_MODEL_TYPES = ["xgboost", "random_forest", "linear_regression"]  # example list


def load_model_config(config_input: Union[str, Dict]) -> Dict:
    """
    Load model config from a dict or a YAML/JSON file.

    Args:
        config_input (str or dict): Path to YAML/JSON config file or dict itself.

    Returns:
        dict: Loaded model config.
    """
    if isinstance(config_input, dict):
        logger.debug("Using provided dict as model_config.")
        return config_input

    if not os.path.exists(config_input):
        raise FileNotFoundError(f"Config file not found: {config_input}")

    with open(config_input, "r") as f:
        if config_input.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        elif config_input.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format. Use JSON or YAML.")


def load_tabular_input(path: str) -> pd.DataFrame:
    """
    Load tabular input file into a pandas DataFrame.
    Supports CSV, Excel, JSON, YAML.

    Args:
        path (str): File path.

    Returns:
        pd.DataFrame: Loaded data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tabular input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, quotechar='"')
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    elif ext in [".yaml", ".yml"]:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported tabular input format. Use CSV, Excel, JSON, or YAML.")


def validate_model_config(config: Dict):
    """
    Validate required fields in model_config dict.

    Raises:
        KeyError, ValueError
    """
    for key in ["model_id", "submission_id", "model_name", "base_path"]:
        if key not in config:
            raise KeyError(f"Missing required key: {key}")

    if not str(config["model_id"]).isdigit():
        raise ValueError("model_id should be numeric")

    if not str(config["submission_id"]).isdigit():
        raise ValueError("submission_id should be numeric")

    if not re.match(r"^[a-zA-Z0-9_]+$", config["model_name"]):
        raise ValueError("model_name should contain only alphanumeric and underscores")

    if not os.path.exists(config["base_path"]):
        raise ValueError(f"base_path does not exist: {config['base_path']}")


def parse_list_column(value: Any, index: int, colname: str) -> list:
    """
    Parses a list from a string or returns list as is.
    Handles NaN, empty, None as empty list.

    Args:
        value: The value to parse.
        index: Row index for error messages.
        colname: Column name for error messages.

    Returns:
        list: Parsed list.

    Raises:
        ValueError: If value cannot be parsed to list.
    """
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == '':
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, str):
            value = value.strip()
            result = ast.literal_eval(value)
            if not isinstance(result, list):
                raise ValueError
            return result

        raise ValueError
    except Exception:
        raise ValueError(
            f"Invalid list format in column '{colname}' at row {index}: {value} (type: {type(value).__name__})"
        )


def validate_row(row: pd.Series, index: int):
    """
    Validate a row of tabular input.

    Raises:
        ValueError
    """
    # Validate sub_model_name
    if "sub_model_name" not in row or not isinstance(row["sub_model_name"], str) or not re.match(r"^[a-zA-Z0-9_]+$", row["sub_model_name"]):
        raise ValueError(f"Invalid sub_model_name at row {index}")

    # Validate model_type if present
    if "model_type" in row and pd.notna(row["model_type"]):
        if row["model_type"] not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type at row {index}, got: {row['model_type']}")

    # Validate training_data_type and validation_data_type
    for col in ["training_data_type", "validation_data_type"]:
        if col in row and pd.notna(row[col]):
            if row[col] not in ["path", "hdfs", "hive"]:
                raise ValueError(f"Invalid {col} at row {index}: {row[col]}")

    # Validate list columns and time_var existence in dataframes will be done later after loading data


def load_data(source: str, data_type: str) -> pd.DataFrame:
    """
    Load data based on type.

    Args:
        source (str): Path or string indicating hdfs/hive query.
        data_type (str): One of "path", "hdfs", "hive".

    Returns:
        pd.DataFrame
    """
    if data_type == "path":
        if not os.path.exists(source):
            raise FileNotFoundError(f"Training/validation data path not found: {source}")
        return pd.read_csv(source)

    elif data_type == "hdfs":
        # Placeholder for spark reading from HDFS
        logger.debug(f"Loading HDFS data from: {source}")
        # spark = get_spark_session()  # Your spark session getter
        # return spark.read.csv(source).toPandas()
        raise NotImplementedError("HDFS data loading not implemented.")

    elif data_type == "hive":
        # Placeholder for spark reading from Hive
        logger.debug(f"Loading Hive data from: {source}")
        # spark = get_spark_session()
        # return spark.sql(source).toPandas()
        raise NotImplementedError("Hive data loading not implemented.")

    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def verify_columns_exist(df_train: pd.DataFrame, df_valid: pd.DataFrame, col_list: list, index: int, colname: str):
    """
    Verify that all columns in col_list exist in both train and validation dataframes.

    Args:
        df_train: Training DataFrame
        df_valid: Validation DataFrame
        col_list: List of columns to verify
        index: Row index for error messages
        colname: Name of the input column being checked
    """
    missing_train = [c for c in col_list if c not in df_train.columns]
    missing_valid = [c for c in col_list if c not in df_valid.columns]
    if missing_train:
        raise ValueError(f"Missing columns in training data at row {index} for '{colname}': {missing_train}")
    if missing_valid:
        raise ValueError(f"Missing columns in validation data at row {index} for '{colname}': {missing_valid}")


def process_model(model_config: Dict):
    """
    Placeholder for downstream processing function.

    Args:
        model_config (Dict): Combined config for model processing.

    Returns:
        None
    """
    logger.info(f"Processing model with sub_model_name: {model_config.get('sub_model_name')}")


def main_process(model_config_input: Union[str, Dict], tabular_input_path: str):
    """
    Main processing function.

    Args:
        model_config_input: dict or path to model_config yaml/json
        tabular_input_path: path to tabular input file (csv/json/yaml)

    Returns:
        None
    """
    model_config = load_model_config(model_config_input)
    validate_model_config(model_config)

    df = load_tabular_input(tabular_input_path)

    required_cols = [
        "sub_model_name", "training_data", "training_data_type",
        "validation_data", "validation_data_type", "model_type",
        "time_var", "time_series_vars", "categorical_vars", "scenario_var",
        "optional_test", "optional_performance_tests",
        "omit_default_diagnostic_tests", "omit_performance_tests"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None  # Fill missing optional columns with None

    for index, row in df.iterrows():
        try:
            validate_row(row, index)

            # Load training and validation dataframes
            df_train = load_data(row["training_data"], row["training_data_type"])
            df_valid = load_data(row["validation_data"], row["validation_data_type"])

            # Parse list columns
            time_series_vars = parse_list_column(row.get("time_series_vars"), index, "time_series_vars")
            categorical_vars = parse_list_column(row.get("categorical_vars"), index, "categorical_vars")
            scenario_var = parse_list_column(row.get("scenario_var"), index, "scenario_var")
            optional_performance_tests = parse_list_column(row.get("optional_performance_tests"), index, "optional_performance_tests")
            omit_default_diagnostic_tests = parse_list_column(row.get("omit_default_diagnostic_tests"), index, "omit_default_diagnostic_tests")
            omit_performance_tests = parse_list_column(row.get("omit_performance_tests"), index, "omit_performance_tests")
            optional_test = row.get("optional_test")  # string or None

            # time_var is a string, verify if present
            time_var = row.get("time_var")
            if time_var is not None and pd.notna(time_var) and time_var != '':
                # Verify time_var column exists in train and valid
                verify_columns_exist(df_train, df_valid, [time_var], index, "time_var")
            else:
                time_var = None

            # Verify list columns exist in dataframes
            for col_name, col_list in [
                ("time_series_vars", time_series_vars),
                ("categorical_vars", categorical_vars),
                ("scenario_var", scenario_var)
            ]:
                if col_list:
                    verify_columns_exist(df_train, df_valid, col_list, index, col_name)

            # Merge row into a copy of model_config
            merged_config = dict(model_config)  # shallow copy
            # Only add keys if they are not None or empty
            if pd.notna(row.get("sub_model_name")):  # Required, but still safe
                merged_config["sub_model_name"] = row["sub_model_name"]

            if pd.notna(row.get("model_type")):
                merged_config["model_type"] = row["model_type"]

            merged_config["training_data"] = df_train
            merged_config["validation_data"] = df_valid

            # Add the new vars
            merged_config["time_var"] = time_var
            merged_config["time_series_vars"] = time_series_vars
            merged_config["categorical_vars"] = categorical_vars
            merged_config["scenario_var"] = scenario_var
            merged_config["optional_test"] = optional_test
            merged_config["optional_performance_tests"] = optional_performance_tests
            merged_config["omit_default_diagnostic_tests"] = omit_default_diagnostic_tests
            merged_config["omit_performance_tests"] = omit_performance_tests

            # Call downstream processing function
            process_model(merged_config)

        except Exception as e:
            logger.error(f"Processing failed at row {index}: {e}")
            # Depending on requirements, you might want to raise or continue
            # raise


if __name__ == "__main__":
    # Example usage:
    # main_process("model_config.yaml", "tabular_input.csv")
    pass




sub_model_name,training_data,training_data_type,validation_data,validation_data_type,model_type,time_var,time_series_vars,categorical_vars,scenario_var,optional_test,optional_performance_tests,omit_default_diagnostic_tests,omit_performance_tests
model_1,train.csv,path,valid.csv,path,xgboost,date,"['age', 'freq']","['gender', 'location']","['scenario1']",test_1,"['perf_test1']","['diag_test1']","['perf_test2']"


[
  {
    "sub_model_name": "model_1",
    "training_data": "train.csv",
    "training_data_type": "path",
    "validation_data": "valid.csv",
    "validation_data_type": "path",
    "model_type": "xgboost",
    "time_var": "date",
    "time_series_vars": ["age", "freq"],
    "categorical_vars": ["gender", "location"],
    "scenario_var": ["scenario1"],
    "optional_test": "test_1",
    "optional_performance_tests": ["perf_test1"],
    "omit_default_diagnostic_tests": ["diag_test1"],
    "omit_performance_tests": ["perf_test2"]
  }
]


- sub_model_name: model_1
  training_data: train.csv
  training_data_type: path
  validation_data: valid.csv
  validation_data_type: path
  model_type: xgboost
  time_var: date
  time_series_vars:
    - age
    - freq
  categorical_vars:
    - gender
    - location
  scenario_var:
    - scenario1
  optional_test: test_1
  optional_performance_tests:
    - perf_test1
  omit_default_diagnostic_tests:
    - diag_test1
  omit_performance_tests:
    - perf_test2
