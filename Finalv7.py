import os
import json
import yaml
import pandas as pd
import logging
import re
import ast
from typing import Union, Dict

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Constants
VALID_MODEL_TYPES = ["xgboost", "random_forest", "linear_regression"]
FALLBACK_FIELDS = [
    "training_data", "training_data_type",
    "validation_data", "validation_data_type",
    "time_var", "time_series_vars", "categorical_vars", "scenario_var",
    "optional_test", "optional_performance_tests",
    "omit_default_diagnostic_tests", "omit_performance_tests"
]


def load_model_config(config_input: Union[str, Dict]) -> Dict:
    """
    Load model configuration from a dict or a file (YAML/JSON).

    Args:
        config_input: Either a dictionary or a path to a .yaml/.yml/.json file.

    Returns:
        A dictionary representing the loaded model configuration.

    Raises:
        FileNotFoundError: If the path is provided and does not exist.
        ValueError: If the file format is unsupported.
    """
    logger.info("Loading model configuration from '%s'", config_input)
    if isinstance(config_input, dict):
        logger.debug("Model config provided directly as dict.")
        return config_input

    if not os.path.exists(config_input):
        logger.error("Model config file does not exist: %s", config_input)
        raise FileNotFoundError(f"Config file not found: {config_input}")

    with open(config_input, "r") as f:
        if config_input.endswith((".yaml", ".yml")):
            logger.debug("Parsing YAML model config.")
            return yaml.safe_load(f)
        elif config_input.endswith(".json"):
            logger.debug("Parsing JSON model config.")
            return json.load(f)
        else:
            logger.error("Unsupported model config file format: %s", config_input)
            raise ValueError("Unsupported config file format; must be .yaml/.yml/.json")


def validate_model_config(config: Dict):
    """
    Validate the required keys and their formats in the main model configuration.

    Args:
        config: The base model configuration dictionary.

    Raises:
        KeyError: If any required key is missing.
        ValueError: If format constraints are violated.
        FileNotFoundError: If base_path does not exist.
    """
    logger.info("Validating base model configuration.")
    required_keys = ["model_id", "submission_id", "model_name", "base_path"]
    for key in required_keys:
        if key not in config:
            logger.error("Missing required key in model_config: %s", key)
            raise KeyError(f"Missing required key: {key}")

    if not str(config["model_id"]).isdigit():
        raise ValueError("model_id should be numeric")
    if not str(config["submission_id"]).isdigit():
        raise ValueError("submission_id should be numeric")
    if not re.match(r"^[a-zA-Z0-9_]+$", config["model_name"]):
        raise ValueError("model_name must contain only alphanumeric and underscores")
    if not os.path.exists(config["base_path"]):
        logger.error("base_path does not exist: %s", config["base_path"])
        raise FileNotFoundError(f"base_path does not exist: {config['base_path']}")


def validate_row(row: pd.Series, index: int):
    """
    Validate required identifiers in a tabular input row.

    Args:
        row: A pandas Series representing the row.
        index: Row index for logging context.

    Raises:
        ValueError: If sub_model_name or model_type is invalid.
    """
    logger.debug("Validating row %s: checking sub_model_name and model_type.", index)
    name = row.get("sub_model_name", "")
    if not isinstance(name, str) or not re.match(r"^[a-zA-Z0-9_]+$", name):
        raise ValueError(f"Invalid sub_model_name at row {index}: {name}")

    model_type = row.get("model_type")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model_type '{model_type}' at row {index}. Must be one of {VALID_MODEL_TYPES}")


def parse_list_column(value, index, colname):
    """
    Parse various user-friendly list representations into a Python list.

    Accepts:
      - Python list-like string: "['a','b']"
      - Comma-separated string: "a, b"
      - Single string: "a"
      - Actual list object

    Args:
        value: Raw input value.
        index: Row index for error context.
        colname: Name of the column being parsed.

    Returns:
        list: Parsed list (empty if input is blank or None).

    Raises:
        ValueError: If input cannot be interpreted as a list.
    """
    logger.debug("Parsing list field '%s' at row %s with value: %r", colname, index, value)
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            s = value.strip()
            # Python list literal
            if s.startswith("[") and s.endswith("]"):
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return parsed
            # Comma-separated
            if "," in s:
                return [v.strip() for v in s.split(",") if v.strip()]
            # Single value
            return [s]
        raise ValueError
    except Exception:
        logger.error("Failed to parse list column '%s' at row %s with value: %r", colname, index, value)
        raise ValueError(f"Invalid list format in column '{colname}' at row {index}: {value}")


def validate_columns_exist(row, train_df, valid_df, index):
    """
    Validate that any provided optional column references exist in the training and validation data.

    Fields considered: time_var, time_series_vars, categorical_vars, scenario_var,
    optional_test, optional_performance_tests, omit_default_diagnostic_tests, omit_performance_tests.

    Args:
        row: Row from tabular input (pandas Series).
        train_df: Loaded training DataFrame.
        valid_df: Loaded validation DataFrame.
        index: Index of the row for logging.

    Returns:
        dict: Parsed and validated optional settings (including model_setting keys and other lists).

    Raises:
        ValueError: If any referenced column is missing in data.
    """
    logger.debug("Validating column existence for row %s.", index)
    result = {}

    # time_var handling (optional)
    if "time_var" in row.index:
        time_var = row.get("time_var")
        if pd.notna(time_var) and time_var != "":
            time_var = str(time_var).strip()
            if time_var not in train_df.columns:
                raise ValueError(f"Time var '{time_var}' not found in training data at row {index}")
            if time_var not in valid_df.columns:
                raise ValueError(f"Time var '{time_var}' not found in validation data at row {index}")
            result["time_var"] = time_var
            logger.debug("Validated time_var: %s for row %s", time_var, index)

    # Group-like model_setting fields
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

    # Other optional lists
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

    Supports .csv, .xlsx/.xls, .json, .yaml/.yml input formats.

    Args:
        path: Path to the tabular input file.

    Returns:
        pd.DataFrame: Loaded tabular data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If format not supported.
    """
    logger.info("Loading tabular input from '%s'", path)
    if not os.path.exists(path):
        logger.error("Tabular input file not found: %s", path)
        raise FileNotFoundError(f"Tabular input file not found: {path}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith((".xlsx", ".xls")):
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
    Load dataset according to specified type.

    Args:
        data_type: One of 'path', 'hdfs', or 'hive'.
        path: Location or identifier for the data.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If path for 'path' type does not exist.
        ValueError: If unsupported data_type.
    """
    logger.debug("Loading data type='%s' from '%s'", data_type, path)
    if data_type == "path":
        if not os.path.exists(path):
            logger.error("Data file not found at path: %s", path)
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_csv(path)
    elif data_type in ("hdfs", "hive"):
        logger.debug("Simulated read for %s at %s", data_type, path)
        return pd.DataFrame()  # Placeholder for real Spark/Hive read
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def process_model(config: Dict):
    """
    Placeholder for downstream processing logic.

    Args:
        config: Fully merged configuration dictionary for a sub-model.
    """
    logger.info("Processing model with sub_model_name=%s", config.get("sub_model_name"))
    # Insert actual model execution logic here.


def main(model_config_input: Union[str, Dict], tabular_input_path: str):
    """
    Entry point: orchestrates loading, validation, merging, and processing.

    Logic:
      - Loads and validates base model config.
      - Loads tabular input.
      - For each row, applies overrides with precedence row > base config.
      - Validates column-level dependencies.
      - Builds merged_config, with optional 'model_setting'.
      - Calls process_model with the merged configuration.

    Args:
        model_config_input: Path or dict for base model config.
        tabular_input_path: Path to tabular input file.

    Raises:
        Exceptions propagate for critical failures; individual row errors are logged and skipped.
    """
    try:
        logger.info("Starting processing pipeline.")
        base_config = load_model_config(model_config_input)
        validate_model_config(base_config)
        df = load_tabular_input(tabular_input_path)

        for idx, row in df.iterrows():
            try:
                logger.info("---- Processing row %s ----", idx)
                validate_row(row, idx)

                # Helper: get field with precedence row > base_config
                def get_field(field_name):
                    if field_name in row.index and pd.notna(row.get(field_name)) and row.get(field_name) != "":
                        return row.get(field_name)
                    return base_config.get(field_name)

                training_data_type = get_field("training_data_type")
                training_data = get_field("training_data")
                validation_data_type = get_field("validation_data_type")
                validation_data = get_field("validation_data")

                if not training_data_type or not training_data:
                    raise ValueError(f"Missing training_data or training_data_type for row {idx}")
                if not validation_data_type or not validation_data:
                    raise ValueError(f"Missing validation_data or validation_data_type for row {idx}")

                # Load datasets
                training_df = load_data(training_data_type, training_data)
                validation_df = load_data(validation_data_type, validation_data)

                # Build synthetic row to supply fallbacks into validation logic
                synthetic_row = row.copy()
                for field in FALLBACK_FIELDS:
                    if (field not in synthetic_row.index) or pd.isna(synthetic_row.get(field)) or synthetic_row.get(field) == "":
                        if field in base_config:
                            synthetic_row[field] = base_config[field]

                validated_optional = validate_columns_exist(synthetic_row, training_df, validation_df, idx)

                # Build merged config
                merged_config = base_config.copy()

                # Add non-optional row-level entries (excluding raw pointers and list/time fields)
                excluded = {
                    "training_data_type", "training_data",
                    "validation_data_type", "validation_data"
                } | {
                    "time_var", "time_series_vars", "categorical_vars", "scenario_var",
                    "optional_test", "optional_performance_tests",
                    "omit_default_diagnostic_tests", "omit_performance_tests"
                }

                for key, value in row.dropna().to_dict().items():
                    if key not in excluded:
                        merged_config[key] = value

                # Attach dataset DataFrames explicitly
                merged_config["training_data_df"] = training_df
                merged_config["validation_data_df"] = validation_df

                # Build model_setting if applicable
                model_setting = {}
                for mkey in ["time_var", "time_series_vars", "categorical_vars", "scenario_var"]:
                    val = validated_optional.pop(mkey, None)
                    if val is not None and (not isinstance(val, list) or len(val) > 0):
                        model_setting[mkey] = val
                if model_setting:
                    merged_config["model_setting"] = model_setting
                else:
                    merged_config.pop("model_setting", None)

                # Remaining optional lists (tests/omit)
                for remaining_key, remaining_val in validated_optional.items():
                    if remaining_val is not None and (not isinstance(remaining_val, list) or len(remaining_val) > 0):
                        merged_config[remaining_key] = remaining_val

                logger.debug("Merged config for row %s: %s", idx, merged_config)
                process_model(merged_config)

            except Exception as row_err:
                logger.error("Row %s processing failed: %s", idx, row_err, exc_info=True)
    except Exception as e:
        logger.critical("Fatal error in pipeline: %s", e, exc_info=True)


import pytest
import pandas as pd
import os
import tempfile
import yaml
import json
from pathlib import Path

# Import functions from your module; adjust module name if different.
from model_processor import (
    load_model_config,
    validate_model_config,
    validate_row,
    parse_list_column,
    validate_columns_exist,
    load_tabular_input,
    load_data,
    main,
)


# ---------- Fixtures ----------

@pytest.fixture
def base_config_dict(tmp_path):
    # Provide fallback defaults for training/validation and optional fields
    return {
        "model_id": "100",
        "submission_id": "200",
        "model_name": "base_model",
        "base_path": str(tmp_path),
        "training_data_type": "path",
        "training_data": str(tmp_path / "train_base.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "valid_base.csv"),
        "time_var": "date_base",
        "time_series_vars": ["f1", "f2"],
        "categorical_vars": ["cat_base"],
        "scenario_var": ["scen_base"],
        "optional_test": ["ot_base"],
        "optional_performance_tests": ["opt_base"],
        "omit_default_diagnostic_tests": ["omit_base"],
        "omit_performance_tests": ["omit_perf_base"],
    }

@pytest.fixture
def write_base_data(tmp_path):
    # Create base training/validation CSVs used for fallback
    df = pd.DataFrame({
        "date_base": ["2020-01-01"],
        "f1": [1],
        "f2": [2],
        "cat_base": ["A"],
        "scen_base": ["S"]
    })
    train = tmp_path / "train_base.csv"
    valid = tmp_path / "valid_base.csv"
    df.to_csv(train, index=False)
    df.to_csv(valid, index=False)
    return str(train), str(valid), df

@pytest.fixture
def minimal_tabular(tmp_path):
    # Only required fields, no optional columns
    df = pd.DataFrame([{
        "sub_model_name": "model_min",
        "model_type": "xgboost",
        "training_data_type": "path",
        "training_data": str(tmp_path / "train.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "valid.csv"),
    }])
    # Create data files with columns expected from base config fallback
    df_data = pd.DataFrame({
        "date_base": ["2020-01-01"],
        "f1": [10],
        "f2": [20],
        "cat_base": ["B"],
        "scen_base": ["T"]
    })
    (tmp_path / "train.csv").write_text(df_data.to_csv(index=False))
    (tmp_path / "valid.csv").write_text(df_data.to_csv(index=False))
    return df, df_data

@pytest.fixture
def full_tabular(tmp_path):
    # All fields provided, list-type fields as comma-separated or list
    df = pd.DataFrame([{
        "sub_model_name": "model_full",
        "model_type": "random_forest",
        "training_data_type": "path",
        "training_data": str(tmp_path / "train_full.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "valid_full.csv"),
        "time_var": "date",
        "time_series_vars": "f1, f2",
        "categorical_vars": "cat1, cat2",
        "scenario_var": "scen1",
        "optional_test": "t1,t2",
        "optional_performance_tests": "pt1",
        "omit_default_diagnostic_tests": "od1",
        "omit_performance_tests": "op1"
    }])
    # create training/validation csv with all relevant columns
    df_data = pd.DataFrame({
        "date": ["2021-02-02"],
        "f1": [1],
        "f2": [2],
        "cat1": ["X"],
        "cat2": ["Y"],
        "scen1": ["Z"]
    })
    df_data.to_csv(tmp_path / "train_full.csv", index=False)
    df_data.to_csv(tmp_path / "valid_full.csv", index=False)
    return df, df_data

# ---------- Unit Tests ----------

def test_load_model_config_dict(base_config_dict):
    cfg = load_model_config(base_config_dict)
    assert isinstance(cfg, dict)
    assert cfg["model_name"] == "base_model"

def test_load_model_config_yaml(tmp_path, base_config_dict):
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(base_config_dict, f)
    cfg = load_model_config(str(p))
    assert cfg["submission_id"] == "200"

def test_validate_model_config_good(tmp_path, base_config_dict, write_base_data):
    # Should not raise
    validate_model_config(base_config_dict)

def test_validate_model_config_missing_key():
    with pytest.raises(KeyError):
        validate_model_config({"model_id": "1", "submission_id": "2", "model_name": "x"})  # missing base_path

def test_validate_row_good():
    row = pd.Series({"sub_model_name": "valid_name", "model_type": "xgboost"})
    validate_row(row, 0)  # no error

def test_validate_row_bad_name():
    row = pd.Series({"sub_model_name": "bad name", "model_type": "xgboost"})
    with pytest.raises(ValueError):
        validate_row(row, 0)

def test_validate_row_bad_model_type():
    row = pd.Series({"sub_model_name": "good_name", "model_type": "unknown"})
    with pytest.raises(ValueError):
        validate_row(row, 0)

@pytest.mark.parametrize("val,expected", [
    ("a,b,c", ["a", "b", "c"]),
    ("['a','b']", ["a", "b"]),
    ("single", ["single"]),
    (["x", "y"], ["x", "y"]),
    ("", []),
    (None, []),
])
def test_parse_list_column_variants(val, expected):
    result = parse_list_column(val, index=1, colname="test")
    assert result == expected

def test_validate_columns_exist_with_full_override(full_tabular, tmp_path):
    # Use full tabular input with its own columns
    df_tab, df_data = full_tabular
    row = df_tab.iloc[0]
    train_df = pd.read_csv(row["training_data"])
    valid_df = pd.read_csv(row["validation_data"])
    result = validate_columns_exist(row, train_df, valid_df, 0)
    assert "time_var" in result and result["time_var"] == "date"
    assert "time_series_vars" in result and result["time_series_vars"] == ["f1", "f2"]
    assert "categorical_vars" in result and result["categorical_vars"] == ["cat1", "cat2"]
    assert "scenario_var" in result and result["scenario_var"] == ["scen1"]
    assert result["optional_test"] == ["t1", "t2"]
    assert result["optional_performance_tests"] == ["pt1"]
    assert result["omit_default_diagnostic_tests"] == ["od1"]
    assert result["omit_performance_tests"] == ["op1"]

def test_validate_columns_exist_with_fallback(base_config_dict, write_base_data, minimal_tabular):
    df_tab, df_data = minimal_tabular
    row = df_tab.iloc[0]
    train_df = pd.read_csv(row["training_data"])
    valid_df = pd.read_csv(row["validation_data"])
    # Provide fallback in base config for time_series_vars etc.
    base_config = base_config_dict
    # Add base values for required fallback fields
    base_config["time_var"] = "date_base"
    base_config["time_series_vars"] = ["f1", "f2"]
    base_config["categorical_vars"] = ["cat_base"]
    base_config["scenario_var"] = ["scen_base"]
    result = validate_columns_exist(row, train_df, valid_df, 0)
    # Since row doesn't have those, result should be empty for model_setting (no error)
    assert "time_var" not in result  # none provided in row, and fallback isn't passed here (this tests absence)

# ---------- Integration Tests ----------

def test_main_with_full_override(tmp_path, full_tabular, base_config_dict):
    # Write base config to file
    config_path = tmp_path / "base.yaml"
    with open(config_path, "w") as f:
        yaml.dump(base_config_dict, f)

    df_tab, df_data = full_tabular
    input_path = tmp_path / "input_full.csv"
    df_tab.to_csv(input_path, index=False)

    # Create required training/validation CSVs
    # Already done in fixture

    # Run main
    # Should not raise
    main(str(config_path), str(input_path))

def test_main_row_missing_optionals_fallback_to_base(tmp_path, minimal_tabular, base_config_dict, write_base_data):
    # Write base config to file
    config_path = tmp_path / "base.yaml"
    with open(config_path, "w") as f:
        yaml.dump(base_config_dict, f)

    df_tab, df_data = minimal_tabular
    input_path = tmp_path / "input_min.csv"
    df_tab.to_csv(input_path, index=False)

    # Write base training/validation files are from write_base_data fixture
    # Run main - should not raise and should use fallback from base_config for missing fields
    main(str(config_path), str(input_path))

def test_main_invalid_model_type(tmp_path):
    # Build config and tabular input with bad model_type
    base_config = {
        "model_id": "1",
        "submission_id": "2",
        "model_name": "mod",
        "base_path": str(tmp_path),
        "training_data_type": "path",
        "training_data": str(tmp_path / "train.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "valid.csv"),
    }
    # Create dummy train/valid
    df = pd.DataFrame({"a":[1], "b":[2]})
    df.to_csv(tmp_path / "train.csv", index=False)
    df.to_csv(tmp_path / "valid.csv", index=False)

    df_tab = pd.DataFrame([{
        "sub_model_name": "goodname",
        "model_type": "BADTYPE",
        "training_data_type": "path",
        "training_data": str(tmp_path / "train.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "valid.csv"),
    }])
    input_file = tmp_path / "input_bad.csv"
    df_tab.to_csv(input_file, index=False)

    config_file = tmp_path / "base.yaml"
    with open(config_file, "w") as f:
        yaml.dump(base_config, f)

    with pytest.raises(ValueError):
        main(str(config_file), str(input_file))

def test_missing_training_data_error(tmp_path, base_config_dict):
    # Row missing training_data entirely and no fallback
    df_tab = pd.DataFrame([{
        "sub_model_name": "m",
        "model_type": "xgboost",
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "valid.csv"),
    }])
    input_path = tmp_path / "input_bad2.csv"
    df_tab.to_csv(input_path, index=False)

    config_file = tmp_path / "base.yaml"
    # Provide base config without training_data to force error
    minimal_base = {
        "model_id": "1",
        "submission_id": "2",
        "model_name": "mod",
        "base_path": str(tmp_path),
    }
    with open(config_file, "w") as f:
        yaml.dump(minimal_base, f)

    with pytest.raises(Exception):
        main(str(config_file), str(input_path))
        


sub_model_name,model_type,training_data,training_data_type,validation_data,validation_data_type,time_var,time_series_vars,categorical_vars,scenario_var,optional_test,optional_performance_tests,omit_default_diagnostic_tests,omit_performance_tests
model_a,xgboost,/data/train_a.csv,path,/data/valid_a.csv,path,date,f1|f2,cat1|cat2,scenario,test1|test2,perf1|perf2,diag1|diag2,perf_omit1
model_b,random_forest,/data/train_b.csv,path,/data/valid_b.csv,path,date_b,"['feat1','feat2']","['catA']",scenarioB,,,,
model_c,xgboost,,,,,,,,"",,,


[
  {
    "sub_model_name": "model_a",
    "model_type": "xgboost",
    "training_data": "/data/train_a.csv",
    "training_data_type": "path",
    "validation_data": "/data/valid_a.csv",
    "validation_data_type": "path",
    "time_var": "date",
    "time_series_vars": ["f1", "f2"],
    "categorical_vars": ["cat1", "cat2"],
    "scenario_var": ["scenario"],
    "optional_test": ["test1", "test2"],
    "optional_performance_tests": ["perf1", "perf2"],
    "omit_default_diagnostic_tests": ["diag1", "diag2"],
    "omit_performance_tests": ["perf_omit1"]
  },
  {
    "sub_model_name": "model_b",
    "model_type": "random_forest",
    "training_data": "/data/train_b.csv",
    "training_data_type": "path",
    "validation_data": "/data/valid_b.csv",
    "validation_data_type": "path",
    "time_var": "date_b",
    "time_series_vars": ["feat1", "feat2"],
    "categorical_vars": ["catA"],
    "scenario_var": ["scenarioB"]
  },
  {
    "sub_model_name": "model_c",
    "model_type": "xgboost"
  }
]

- sub_model_name: model_a
  model_type: xgboost
  training_data: /data/train_a.csv
  training_data_type: path
  validation_data: /data/valid_a.csv
  validation_data_type: path
  time_var: date
  time_series_vars: [f1, f2]
  categorical_vars: [cat1, cat2]
  scenario_var: [scenario]
  optional_test: [test1, test2]
  optional_performance_tests: [perf1, perf2]
  omit_default_diagnostic_tests: [diag1, diag2]
  omit_performance_tests: [perf_omit1]

- sub_model_name: model_b
  model_type: random_forest
  training_data: /data/train_b.csv
  training_data_type: path
  validation_data: /data/valid_b.csv
  validation_data_type: path
  time_var: date_b
  time_series_vars: [feat1, feat2]
  categorical_vars: [catA]
  scenario_var: [scenarioB]

- sub_model_name: model_c
  model_type: xgboost

