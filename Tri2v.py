import os
import re
import json
import yaml
import pandas as pd
import logging
from typing import Union, Dict, Any
from pyspark.sql import SparkSession

# -------------------------------
# Logger Configuration
# -------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(ch)

# -------------------------------
# Spark Initialization
# -------------------------------
spark = SparkSession.builder.appName("ModelProcessor").getOrCreate()

# -------------------------------
# Core Functions
# -------------------------------

def load_model_config(model_config_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Loads a model configuration from a dictionary or file.

    Args:
        model_config_input (str | dict): Path to JSON/YAML file or a dictionary.

    Returns:
        dict: Parsed model configuration.

    Raises:
        FileNotFoundError, ValueError, TypeError
    """
    if isinstance(model_config_input, dict):
        return model_config_input
    elif isinstance(model_config_input, str):
        if not os.path.isfile(model_config_input):
            raise FileNotFoundError(f"Config file not found: {model_config_input}")
        ext = os.path.splitext(model_config_input)[-1].lower()
        with open(model_config_input, 'r') as f:
            if ext == '.json':
                return json.load(f)
            elif ext in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported config file format. Use .json or .yaml")
    else:
        raise TypeError("model_config must be a dict or a path to a JSON/YAML file.")

def validate_model_config(config: Dict[str, Any]) -> None:
    """
    Validates required keys and formats in the model config.

    Args:
        config (dict): Model configuration.

    Raises:
        KeyError, ValueError
    """
    required_keys = ['model_id', 'submission_id', 'model_name', 'base_path']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    if not str(config['model_id']).isdigit():
        raise ValueError("model_id must be numeric.")

    if not str(config['submission_id']).isdigit():
        raise ValueError("submission_id must be numeric.")

    if not re.match(r'^[A-Za-z0-9]+(_[A-Za-z0-9]+)*$', config['model_name']):
        raise ValueError("model_name must use only alphanumerics and underscores.")

    if not os.path.isdir(config['base_path']):
        raise ValueError(f"base_path '{config['base_path']}' does not exist or is not a directory.")

def load_tabular_input(tabular_input_path: str) -> pd.DataFrame:
    """
    Loads a tabular input file into a pandas DataFrame.

    Args:
        tabular_input_path (str): Path to the input file (CSV, Excel, JSON, YAML).

    Returns:
        pd.DataFrame: Parsed tabular input.

    Raises:
        FileNotFoundError, ValueError
    """
    if not os.path.isfile(tabular_input_path):
        raise FileNotFoundError(f"{tabular_input_path} not found.")

    ext = os.path.splitext(tabular_input_path)[-1].lower()
    try:
        if ext == ".csv":
            return pd.read_csv(tabular_input_path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(tabular_input_path)
        elif ext == ".json":
            return pd.read_json(tabular_input_path)
        elif ext in [".yaml", ".yml"]:
            with open(tabular_input_path, "r") as f:
                data = yaml.safe_load(f)
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        raise ValueError(f"Error reading tabular input: {e}")

def validate_row(row: pd.Series, idx: int) -> None:
    """
    Validates an individual row from the tabular input.

    Args:
        row (pd.Series): Row to validate.
        idx (int): Row index (used for error reporting).

    Raises:
        ValueError
    """
    required_columns = {
        'sub_model_name', 'training_data', 'validation_data',
        'training_data_type', 'validation_data_type', 'model_type'
    }
    missing = required_columns - set(row.index)
    if missing:
        raise ValueError(f"Row {idx} is missing required columns: {missing}")

    if not isinstance(row['sub_model_name'], str) or not re.match(r'^[A-Za-z0-9]+(_[A-Za-z0-9]+)*$', row['sub_model_name']):
        raise ValueError(f"Row {idx}: Invalid sub_model_name: {row['sub_model_name']}")

def load_data(source_type: str, source_value: str):
    """
    Loads data based on the source type and location.

    Args:
        source_type (str): One of "path", "hdfs", or "hive".
        source_value (str): Path or table name.

    Returns:
        pd.DataFrame or pyspark.sql.DataFrame: Loaded data.

    Raises:
        ValueError, RuntimeError
    """
    try:
        if source_type == 'path':
            return pd.read_csv(source_value)
        elif source_type == 'hdfs':
            return spark.read.csv(source_value, header=True, inferSchema=True)
        elif source_type == 'hive':
            return spark.read.table(source_value)
        else:
            raise ValueError(f"Unsupported data type: {source_type}")
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {source_type}: {e}")

def process_model(config: Dict[str, Any]) -> None:
    """
    Placeholder for model processing logic.

    Args:
        config (dict): Merged config containing sub-model info and data.

    Returns:
        None
    """
    logger.info(f"Processing sub_model: {config.get('sub_model_name')}")

def main(model_config_input: Union[str, Dict[str, Any]], tabular_input_path: str) -> None:
    """
    Main entry point to load config and tabular input, validate, process rows.

    Args:
        model_config_input (str | dict): Path to YAML/JSON or a dict object.
        tabular_input_path (str): Path to tabular input file.

    Returns:
        None
    """
    try:
        config = load_model_config(model_config_input)
        validate_model_config(config)
        df = load_tabular_input(tabular_input_path)

        for idx, row in df.iterrows():
            try:
                validate_row(row, idx)
                row_data = row.to_dict()

                training_df = load_data(row_data['training_data_type'], row_data['training_data'])
                validation_df = load_data(row_data['validation_data_type'], row_data['validation_data'])

                merged_config = config.copy()
                merged_config.update(row_data)
                merged_config['training_data_df'] = training_df
                merged_config['validation_data_df'] = validation_df

                process_model(merged_config)

            except Exception as row_err:
                logger.error(f"Row {idx} processing failed: {row_err}")

    except Exception as e:
        logger.critical(f"Fatal error during processing: {e}")





def validate_row(row: pd.Series, idx: int) -> None:
    """
    Validates required columns and formatting for a row.

    Args:
        row (pd.Series): Row to validate.
        idx (int): Row index.

    Raises:
        ValueError: On missing or malformed values.
    """
    required_columns = {
        'sub_model_name', 'training_data', 'validation_data',
        'training_data_type', 'validation_data_type', 'model_type'
    }
    missing = required_columns - set(row.index)
    if missing:
        raise ValueError(f"Row {idx} is missing required columns: {missing}")

    if not isinstance(row['sub_model_name'], str) or not re.match(r'^[A-Za-z0-9]+(_[A-Za-z0-9]+)*$', row['sub_model_name']):
        raise ValueError(f"Row {idx}: Invalid sub_model_name: {row['sub_model_name']}")



def validate_columns_exist(
    row: Dict[str, Any],
    training_df,
    validation_df,
    idx: int
) -> Dict[str, Any]:
    """
    Validates that selected columns (if provided) exist in the datasets.

    Args:
        row (dict): The row dictionary from tabular input.
        training_df: The loaded training DataFrame.
        validation_df: The loaded validation DataFrame.
        idx (int): Row index.

    Returns:
        dict: Dict containing cleaned versions of valid columns to inject into config.

    Raises:
        ValueError: If a column is missing in either dataset.
    """
    optional_fields = [
        "time_var", "time_series_vars", "categorical_vars",
        "scenario_var", "optional_test", "optional_performance_tests",
        "omit_default_diagnostic_tests", "omit_performance_tests"
    ]

    validated_fields = {}

    def _check_list(name, value):
        if value is None or pd.isna(value):
            return []
        if isinstance(value, str):
            value = json.loads(value)  # if provided as a stringified list
        if not isinstance(value, list):
            raise ValueError(f"Row {idx}: Field '{name}' must be a list.")
        for col in value:
            if col not in training_df.columns or col not in validation_df.columns:
                raise ValueError(f"Row {idx}: Column '{col}' from '{name}' not found in datasets.")
        return value

    for field in optional_fields:
        value = row.get(field, None)
        if field == "time_var":
            if value and not pd.isna(value):
                if value not in training_df.columns or value not in validation_df.columns:
                    raise ValueError(f"Row {idx}: time_var '{value}' not found in datasets.")
                validated_fields[field] = value
        elif field in ["time_series_vars", "categorical_vars", "scenario_var"]:
            if value and not pd.isna(value):
                validated_fields[field] = _check_list(field, value)
        else:
            # For test-related fields: just parse if list or string
            if value and not pd.isna(value):
                if isinstance(value, str):
                    value = json.loads(value)
                validated_fields[field] = value

    return validated_fields

def main(model_config_input: Union[str, Dict[str, Any]], tabular_input_path: str) -> None:
    """
    Main entry point to load config and tabular input, validate, and process.

    Args:
        model_config_input (str | dict): Path to config or dict.
        tabular_input_path (str): Path to input table.

    Returns:
        None
    """
    try:
        config = load_model_config(model_config_input)
        validate_model_config(config)
        df = load_tabular_input(tabular_input_path)

        for idx, row in df.iterrows():
            try:
                validate_row(row, idx)
                row_data = row.to_dict()

                training_df = load_data(row_data['training_data_type'], row_data['training_data'])
                validation_df = load_data(row_data['validation_data_type'], row_data['validation_data'])

                validated_optional = validate_columns_exist(row_data, training_df, validation_df, idx)

                merged_config = config.copy()
                merged_config.update(row_data)
                merged_config.update(validated_optional)
                merged_config['training_data_df'] = training_df
                merged_config['validation_data_df'] = validation_df

                process_model(merged_config)

            except Exception as row_err:
                logger.error(f"Row {idx} processing failed: {row_err}")

    except Exception as e:
        logger.critical(f"Fatal error during processing: {e}")


- sub_model_name: sub_model_1
  training_data: "/data/train.csv"
  training_data_type: path
  validation_data: "/data/val.csv"
  validation_data_type: path
  model_type: xgboost
  time_var: "date"
  time_series_vars: ["feature1", "feature2"]
  categorical_vars: ["cat1", "cat2"]
  scenario_var: ["scn"]
  optional_test: ["custom_test_1"]
  optional_performance_tests: []
  omit_default_diagnostic_tests: []
  omit_performance_tests: []


import pytest
import pandas as pd
import tempfile
import json
import yaml
import os
from model_processor import (
    load_model_config, validate_model_config,
    load_tabular_input, validate_row,
    validate_columns_exist
)

# --- Fixtures for Testing ---

@pytest.fixture
def valid_model_config_dict():
    return {
        "model_id": "101",
        "submission_id": "202",
        "model_name": "valid_model_name",
        "base_path": tempfile.gettempdir()
    }

@pytest.fixture
def valid_training_data():
    return pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "feature1": [1, 2],
        "feature2": [3, 4],
        "cat1": ["a", "b"],
        "cat2": ["x", "y"],
        "scn": ["A", "B"]
    })

@pytest.fixture
def valid_row_dict():
    return {
        "sub_model_name": "sub_model_1",
        "training_data": "/path/train.csv",
        "validation_data": "/path/val.csv",
        "training_data_type": "path",
        "validation_data_type": "path",
        "model_type": "xgboost",
        "time_var": "date",
        "time_series_vars": json.dumps(["feature1", "feature2"]),
        "categorical_vars": json.dumps(["cat1", "cat2"]),
        "scenario_var": json.dumps(["scn"]),
        "optional_test": json.dumps(["test1"]),
        "optional_performance_tests": json.dumps([]),
        "omit_default_diagnostic_tests": json.dumps([]),
        "omit_performance_tests": json.dumps([])
    }

# --- Tests ---

def test_load_model_config_from_dict(valid_model_config_dict):
    config = load_model_config(valid_model_config_dict)
    assert isinstance(config, dict)
    assert config['model_id'] == "101"

def test_load_model_config_from_json(tmp_path, valid_model_config_dict):
    json_path = tmp_path / "config.json"
    with open(json_path, "w") as f:
        json.dump(valid_model_config_dict, f)

    config = load_model_config(str(json_path))
    assert config['model_name'] == "valid_model_name"

def test_load_model_config_from_yaml(tmp_path, valid_model_config_dict):
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(valid_model_config_dict, f)

    config = load_model_config(str(yaml_path))
    assert config['model_id'] == "101"

def test_validate_model_config_success(valid_model_config_dict):
    validate_model_config(valid_model_config_dict)  # Should not raise

def test_validate_model_config_missing_key():
    with pytest.raises(KeyError):
        validate_model_config({"model_id": "123"})

def test_validate_model_config_invalid_model_name(valid_model_config_dict):
    valid_model_config_dict["model_name"] = "bad name!"
    with pytest.raises(ValueError):
        validate_model_config(valid_model_config_dict)

def test_validate_row_valid():
    row = pd.Series({
        "sub_model_name": "sub_model_1",
        "training_data": "/x.csv",
        "validation_data": "/y.csv",
        "training_data_type": "path",
        "validation_data_type": "path",
        "model_type": "xgboost"
    })
    validate_row(row, 0)  # Should not raise

def test_validate_row_invalid_name():
    row = pd.Series({
        "sub_model_name": "bad model",
        "training_data": "/x.csv",
        "validation_data": "/y.csv",
        "training_data_type": "path",
        "validation_data_type": "path",
        "model_type": "xgboost"
    })
    with pytest.raises(ValueError):
        validate_row(row, 0)

def test_validate_columns_exist(valid_row_dict, valid_training_data):
    row = valid_row_dict.copy()
    row["time_series_vars"] = json.dumps(["feature1"])
    row["categorical_vars"] = json.dumps(["cat1"])
    row["scenario_var"] = json.dumps(["scn"])

    validated = validate_columns_exist(row, valid_training_data, valid_training_data, 0)
    assert validated["time_var"] == "date"
    assert "categorical_vars" in validated
    assert validated["categorical_vars"] == ["cat1"]

def test_validate_columns_exist_missing_feature(valid_row_dict, valid_training_data):
    row = valid_row_dict.copy()
    row["time_series_vars"] = json.dumps(["missing_feature"])
    with pytest.raises(ValueError):
        validate_columns_exist(row, valid_training_data, valid_training_data, 0)


