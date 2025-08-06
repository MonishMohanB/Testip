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
if not logger.hasHandlers():
    logger.addHandler(handler)

VALID_MODEL_TYPES = ["xgboost", "random_forest", "linear_regression"]
LIST_COLUMNS = [
    "time_series_vars", "categorical_vars", "scenario_var",
    "optional_test", "optional_performance_tests",
    "omit_default_diagnostic_tests", "omit_performance_tests"
]


def load_model_config(config_input: Union[str, Dict]) -> Dict:
    """
    Load model configuration from a dictionary or JSON/YAML file.

    Args:
        config_input: Path to a JSON/YAML file or a dict containing the base config.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the given file path does not exist.
        ValueError: If the file is not supported (not JSON/YAML).
    """
    logger.info("Loading model configuration...")
    if isinstance(config_input, dict):
        logger.debug("Model config provided as dict.")
        return config_input
    if not os.path.exists(config_input):
        logger.error("Model config file not found: %s", config_input)
        raise FileNotFoundError(f"Config file not found: {config_input}")
    with open(config_input, "r") as f:
        if config_input.endswith((".yaml", ".yml")):
            logger.debug("Parsing YAML model config.")
            return yaml.safe_load(f)
        elif config_input.endswith(".json"):
            logger.debug("Parsing JSON model config.")
            return json.load(f)
        else:
            logger.error("Unsupported model config format: %s", config_input)
            raise ValueError("Unsupported config file format")


def validate_model_config(config: Dict):
    """
    Validate the main model configuration.

    Args:
        config: Configuration dictionary.

    Raises:
        KeyError: Missing required keys.
        ValueError: Invalid formats for values.
        FileNotFoundError: base_path does not exist.
    """
    logger.info("Validating model configuration...")
    for key in ["model_id", "submission_id", "model_name", "base_path"]:
        if key not in config:
            logger.error("Missing required key in model_config: %s", key)
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
    Validate required values in a row of the tabular input.

    Args:
        row: Pandas Series of the row.
        index: Index number of the row for logging.

    Raises:
        ValueError: If sub_model_name or model_type is invalid.
    """
    logger.debug("Validating row %s for sub_model_name and model_type...", index)
    name = row.get("sub_model_name", "")
    if not isinstance(name, str) or not re.match(r"^[a-zA-Z0-9_]+$", name):
        raise ValueError(f"Invalid sub_model_name at row {index}: {name}")
    model_type = row.get("model_type")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model_type '{model_type}' at row {index}. Must be one of {VALID_MODEL_TYPES}")


def parse_list_column(value, index, colname):
    """
    Parse flexible list-like inputs into a Python list.

    Accepts:
      - Python-style list string: "['a','b']"
      - Comma-separated string: "a, b"
      - Single value string: "a"
      - Actual list object

    Args:
        value: Raw value from tabular input.
        index: Row index for error context.
        colname: Column name being parsed.

    Returns:
        list: Parsed list (empty if no value).

    Raises:
        ValueError: If the value cannot be parsed into a list.
    """
    logger.debug("Parsing list column '%s' at row %s with raw value: %s", colname, index, value)
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            s = value.strip()
            # Try Python literal like ['a','b']
            if s.startswith("[") and s.endswith("]"):
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return parsed
            # Comma separated
            if "," in s:
                return [v.strip() for v in s.split(",") if v.strip()]
            # Single string
            return [s]
        # fallback
        raise ValueError
    except Exception:
        logger.error("Failed to parse list column '%s' at row %s: %s", colname, index, value)
        raise ValueError(f"Invalid list format in column '{colname}' at row {index}: {value}")


def validate_columns_exist(row, train_df, valid_df, index):
    """
    Validate that optional/time/list columns exist in both training and validation dataframes
    if they were provided in the row.

    Args:
        row: Pandas Series row from tabular input.
        train_df: DataFrame loaded from training data.
        valid_df: DataFrame loaded from validation data.
        index: Row index.

    Returns:
        dict: Keys and parsed values for present optional settings (e.g., model_setting and tests).

    Raises:
        ValueError: If a referenced column does not exist in the input dataframes.
    """
    logger.debug("Validating column existence for row %s...", index)
    result = {}

    # time_var is optional
    time_var = row.get("time_var") if "time_var" in row.index else None
    if pd.notna(time_var) and time_var != "":
        time_var = str(time_var).strip()
        if time_var not in train_df.columns:
            raise ValueError(f"Time var '{time_var}' not found in training data at row {index}")
        if time_var not in valid_df.columns:
            raise ValueError(f"Time var '{time_var}' not found in validation data at row {index}")
        result["time_var"] = time_var
        logger.debug("Validated time_var '%s' for row %s", time_var, index)

    # List/group columns
    model_setting = {}
    for colname in ["time_series_vars", "categorical_vars", "scenario_var"]:
        if colname in row.index:
            parsed = parse_list_column(row.get(colname), index, colname)
            if parsed:
                for col in parsed:
                    if col not in train_df.columns:
                        raise ValueError(f"Column '{col}' from '{colname}' not found in training_data at row {index}")
                    if col not in valid_df.columns:
                        raise ValueError(f"Column '{col}' from '{colname}' not found in validation_data at row {index}")
                model_setting[colname] = parsed
                logger.debug("Validated %s: %s for row %s", colname, parsed, index)

    if model_setting:
        result.update(model_setting)

    # Other optional lists (tests and omissions)
    for colname in [
        "optional_test", "optional_performance_tests",
        "omit_default_diagnostic_tests", "omit_performance_tests"
    ]:
        if colname in row.index:
            parsed = parse_list_column(row.get(colname), index, colname)
            if parsed:
                result[colname] = parsed
                logger.debug("Validated %s: %s for row %s", colname, parsed, index)

    return result


def load_tabular_input(path: str) -> pd.DataFrame:
    """
    Load tabular input file into a DataFrame.

    Supports CSV, Excel, JSON, YAML.

    Args:
        path: Path to the tabular input file.

    Returns:
        DataFrame with tabular rows.

    Raises:
        FileNotFoundError: If file missing.
        ValueError: If unsupported extension.
    """
    logger.info("Loading tabular input from %s", path)
    if not os.path.exists(path):
        logger.error("Tabular input file not found: %s", path)
        raise FileNotFoundError(f"Tabular input file not found: {path}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    elif path.endswith(".json"):
        return pd.read_json(path)
    elif path.endswith((".yaml", ".yml")):
        with open(path, "r") as f:
            return pd.DataFrame(yaml.safe_load(f))
    else:
        logger.error("Unsupported tabular input format: %s", path)
        raise ValueError("Unsupported tabular input format")


def load_data(data_type: str, path: str) -> pd.DataFrame:
    """
    Load training or validation data based on the type indicator.

    Args:
        data_type: One of 'path', 'hdfs', 'hive'.
        path: Data source string (e.g., local path or HDFS/Hive identifier).

    Returns:
        DataFrame loaded.

    Raises:
        ValueError: If the data_type is unsupported.
    """
    logger.debug("Loading data (type=%s) from '%s'", data_type, path)
    if data_type == "path":
        if not os.path.exists(path):
            logger.error("Training/validation data file not found: %s", path)
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_csv(path)
    elif data_type in ("hdfs", "hive"):
        logger.debug("Simulated read for %s at %s", data_type, path)
        return pd.DataFrame()  # Placeholder for real spark read
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def process_model(config: Dict):
    """
    Placeholder for downstream model processing.

    Args:
        config: Fully merged and validated configuration for a sub-model.
    """
    logger.info("Processing model with sub_model_name=%s", config.get("sub_model_name"))
    # Actual model logic would go here.


def main(model_config_input: Union[str, Dict], tabular_input_path: str):
    """
    Orchestrator: load configs, iterate rows, validate, merge, and call processing.

    Args:
        model_config_input: Path to base model config file or dict.
        tabular_input_path: Path to tabular input file (CSV/JSON/YAML).
    """
    try:
        logger.info("Starting main processing pipeline...")
        base_config = load_model_config(model_config_input)
        validate_model_config(base_config)
        df = load_tabular_input(tabular_input_path)

        for idx, row in df.iterrows():
            try:
                logger.info("---- Processing row %s ----", idx)
                validate_row(row, idx)
                row_data = row.dropna().to_dict()

                # Load the training and validation data
                training_df = load_data(row_data["training_data_type"], row_data["training_data"])
                validation_df = load_data(row_data["validation_data_type"], row_data["validation_data"])

                # Validate optional columns and existence
                validated_optional = validate_columns_exist(row, training_df, validation_df, idx)

                # Build merged config
                merged_config = base_config.copy()

                # Add non-optional row-level fields (excluding raw data pointers and list fields)
                excluded = {
                    "training_data_type", "training_data",
                    "validation_data_type", "validation_data"
                } | set(["time_var", "time_series_vars", "categorical_vars", "scenario_var",
                         "optional_test", "optional_performance_tests",
                         "omit_default_diagnostic_tests", "omit_performance_tests"])
                for key, value in row_data.items():
                    if key not in excluded and pd.notna(value):
                        merged_config[key] = value

                # Insert model_setting and other optional validated fields
                if validated_optional:
                    # model_setting comprises only time_var, time_series_vars, categorical_vars, scenario_var if present
                    model_setting = {}
                    for mkey in ["time_var", "time_series_vars", "categorical_vars", "scenario_var"]:
                        if mkey in validated_optional:
                            model_setting[mkey] = validated_optional.pop(mkey)
                    if model_setting:
                        merged_config["model_setting"] = model_setting
                    # Remaining optional test/omit fields
                    for remaining_key, remaining_val in validated_optional.items():
                        merged_config[remaining_key] = remaining_val

                # Attach loaded data
                merged_config["training_data_df"] = training_df
                merged_config["validation_data_df"] = validation_df

                logger.debug("Final merged config for row %s: %s", idx, merged_config)
                process_model(merged_config)

            except Exception as row_err:
                logger.error("Row %s processing failed: %s", idx, row_err, exc_info=True)

    except Exception as e:
        logger.critical("Fatal error during processing: %s", e, exc_info=True)






import pytest
import pandas as pd
import tempfile
import os
import json
import yaml
from your_script_filename import (  # Replace with the actual filename
    load_model_config, validate_model_config, validate_row,
    parse_list_column, validate_columns_exist, load_tabular_input
)

# ---------- Fixtures ----------

@pytest.fixture
def base_config_dict():
    return {
        "model_id": 1,
        "submission_id": 10,
        "model_name": "my_model",
        "base_path": tempfile.gettempdir()
    }

@pytest.fixture
def sample_row():
    return pd.Series({
        "sub_model_name": "model_abc",
        "model_type": "xgboost",
        "training_data": "train.csv",
        "training_data_type": "path",
        "validation_data": "val.csv",
        "validation_data_type": "path",
        "time_var": "date",
        "categorical_vars": "cat1,cat2",
        "optional_test": "foo, bar",
    })

@pytest.fixture
def dummy_data(tmp_path):
    df = pd.DataFrame({
        "date": ["2021-01-01", "2021-01-02"],
        "cat1": ["A", "B"],
        "cat2": ["X", "Y"],
        "target": [1, 0]
    })
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    df.to_csv(train_path, index=False)
    df.to_csv(val_path, index=False)
    return str(train_path), str(val_path), df

# ---------- Unit Tests ----------

def test_load_model_config_from_dict(base_config_dict):
    config = load_model_config(base_config_dict)
    assert config["model_name"] == "my_model"

def test_load_model_config_from_yaml(tmp_path, base_config_dict):
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(base_config_dict, f)
    config = load_model_config(str(file_path))
    assert config["model_id"] == 1

def test_validate_model_config_valid(base_config_dict):
    validate_model_config(base_config_dict)  # Should not raise

def test_validate_row_valid(sample_row):
    validate_row(sample_row, index=0)

@pytest.mark.parametrize("value,expected", [
    ("a,b,c", ["a", "b", "c"]),
    ("['a', 'b']", ["a", "b"]),
    ("x", ["x"]),
    (["x", "y"], ["x", "y"]),
    ("", []),
    (None, []),
])
def test_parse_list_column_variants(value, expected):
    result = parse_list_column(value, index=0, colname="test_col")
    assert result == expected

def test_validate_columns_exist_success(sample_row, dummy_data):
    train_path, val_path, df = dummy_data
    sample_row["training_data"] = train_path
    sample_row["validation_data"] = val_path
    result = validate_columns_exist(sample_row, df, df, index=0)
    assert "categorical_vars" in result
    assert result["time_var"] == "date"

def test_load_tabular_input_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    file_path = tmp_path / "input.csv"
    df.to_csv(file_path, index=False)
    loaded = load_tabular_input(str(file_path))
    assert loaded.shape == (2, 1)
