import os
import json
import yaml
import pandas as pd
import logging
import re
import ast
import argparse
from typing import Union, Dict

# Spark import (needs pyspark installed)
from pyspark.sql import SparkSession

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
    "omit_default_diagnostic_tests", "omit_performance_tests", "repartition"
]

# Initialize Spark session globally
def get_spark_session():
    """
    Get or create a SparkSession.
    """
    return SparkSession.builder.appName("ModelProcessor").getOrCreate()

# ------------------ Core Functions ------------------

def load_model_config(config_input: Union[str, Dict]) -> Dict:
    """
    Load model configuration from a dict or a YAML/JSON file.

    Args:
        config_input: Dictionary or path to .yaml/.yml/.json file.

    Returns:
        dict: Loaded model configuration.

    Raises:
        FileNotFoundError: If the file path doesn't exist.
        ValueError: If unsupported file format.
    """
    logger.info("Loading model configuration from '%s'", config_input)
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
            logger.error("Unsupported config file format: %s", config_input)
            raise ValueError("Unsupported config file format; must be .yaml/.yml/.json")


def validate_model_config(config: Dict):
    """
    Validate required keys and formats in the base model configuration.

    Args:
        config: Base model configuration.

    Raises:
        KeyError: If required key missing.
        ValueError: If format is invalid.
        FileNotFoundError: If base_path doesn't exist.
    """
    logger.info("Validating base model configuration.")
    required = ["model_id", "submission_id", "model_name", "base_path"]
    for key in required:
        if key not in config:
            logger.error("Missing required key in model_config: %s", key)
            raise KeyError(f"Missing required key: {key}")
    if not str(config["model_id"]).isdigit():
        raise ValueError("model_id should be numeric")
    if not str(config["submission_id"]).isdigit():
        raise ValueError("submission_id should be numeric")
    if not re.match(r"^[a-zA-Z0-9_]+$", config["model_name"]):
        raise ValueError("model_name must be alphanumeric or underscores only")
    if not os.path.exists(config["base_path"]):
        logger.error("base_path does not exist: %s", config["base_path"])
        raise FileNotFoundError(f"base_path does not exist: {config['base_path']}")


def validate_row(row: pd.Series, index: int):
    """
    Validate per-row required identifiers.

    Args:
        row: Row from tabular input.
        index: Row index.

    Raises:
        ValueError: If sub_model_name or model_type invalid.
    """
    logger.debug("Validating row %s for sub_model_name and model_type.", index)
    name = row.get("sub_model_name", "")
    if not isinstance(name, str) or not re.match(r"^[a-zA-Z0-9_]+$", name):
        raise ValueError(f"Invalid sub_model_name at row {index}: {name}")
    model_type = row.get("model_type")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model_type '{model_type}' at row {index}. Must be one of {VALID_MODEL_TYPES}")


def parse_list_column(value, index, colname):
    """
    Parse a potentially flexible list-like input into Python list.

    Args:
        value: Raw value (string, list, None).
        index: Row index for context.
        colname: Column name.

    Returns:
        list: Parsed list value (empty if absent).

    Raises:
        ValueError: If unable to parse to list.
    """
    logger.debug("Parsing list column '%s' at row %s with value: %r", colname, index, value)
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
            # delimiter could be comma or pipe
            if "," in s:
                return [v.strip() for v in s.split(",") if v.strip()]
            if "|" in s:
                return [v.strip() for v in s.split("|") if v.strip()]
            return [s]
        raise ValueError
    except Exception:
        logger.error("Failed to parse list column '%s' at row %s: %s", colname, index, value)
        raise ValueError(f"Invalid list format in column '{colname}' at row {index}: {value}")


def validate_columns_exist(row, train_df, valid_df, index):
    """
    Validate existence of optional/time/list columns in provided dataframes.

    Args:
        row: Row of tabular input.
        train_df: Training DataFrame.
        valid_df: Validation DataFrame.
        index: Row index.

    Returns:
        dict: Validated optional settings including model_setting and test lists.

    Raises:
        ValueError: If any referenced column is missing.
    """
    logger.debug("Validating column existence for row %s", index)
    result = {}

    # time_var
    if "time_var" in row.index:
        time_var = row.get("time_var")
        if pd.notna(time_var) and time_var != "":
            time_var = str(time_var).strip()
            if time_var not in train_df.columns:
                raise ValueError(f"Time var '{time_var}' not found in training data at row {index}")
            if time_var not in valid_df.columns:
                raise ValueError(f"Time var '{time_var}' not found in validation data at row {index}")
            result["time_var"] = time_var
            logger.debug("Validated time_var '%s' for row %s", time_var, index)

    # model_setting components
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

    # Optional test/omit lists
    for colname in ["optional_test", "optional_performance_tests", "omit_default_diagnostic_tests", "omit_performance_tests"]:
        if colname in row.index:
            parsed = parse_list_column(row.get(colname), index, colname)
            if parsed:
                result[colname] = parsed
                logger.debug("Validated %s: %s for row %s", colname, parsed, index)

    # repartition (if provided) passed through as-is
    if "repartition" in row.index:
        val = row.get("repartition")
        if pd.notna(val) and str(val).strip() != "":
            try:
                result["repartition"] = int(val)
                logger.debug("Using repartition=%s for row %s", result["repartition"], index)
            except Exception:
                logger.warning("Invalid repartition value '%s' at row %s, defaulting will apply", val, index)

    return result


def load_tabular_input(path: str) -> pd.DataFrame:
    """
    Load tabular input from supported formats into a DataFrame.

    Args:
        path: Path to CSV/JSON/YAML/XLSX file.

    Returns:
        DataFrame: Loaded tabular input.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If format is unsupported.
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


def load_data(data_type: str, path: str, repartition: int = 200):
    """
    Load training/validation data based on type: 'path', 'hdfs', or 'hive'.

    Args:
        data_type: Source type.
        path: Location from which to load.
        repartition: Number of partitions when using Spark (hdfs).

    Returns:
        DataFrame: Pandas DataFrame for 'path', Spark DataFrame for 'hdfs'/'hive'.

    Raises:
        FileNotFoundError: If a local path doesn't exist.
        ValueError: If unsupported data_type.
    """
    logger.debug("Loading data of type '%s' from '%s' with repartition=%s", data_type, path, repartition)
    if data_type == "path":
        if not os.path.exists(path):
            logger.error("Data file not found: %s", path)
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_csv(path)
    elif data_type == "hdfs":
        spark = get_spark_session()
        df_spark = spark.read.parquet(path)
        df_spark = df_spark.repartition(repartition)
        logger.debug("Loaded HDFS parquet and repartitioned to %s partitions", repartition)
        return df_spark
    elif data_type == "hive":
        spark = get_spark_session()
        df_spark = spark.read.table(path)
        logger.debug("Loaded Hive table %s", path)
        return df_spark
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def process_model(config: Dict):
    """
    Stub for actual model processing.

    Args:
        config: Merged configuration for a sub-model.
    """
    logger.info("Processing model: %s", config.get("sub_model_name"))
    # Placeholder: actual logic to train/evaluate model would go here.


def main(model_config_input: Union[str, Dict], tabular_input_path: str):
    """
    Orchestrate the pipeline: load base config, iterate tabular rows, apply overrides,
    validate, merge, and process each sub-model.

    Args:
        model_config_input: Path to or dict of base model config.
        tabular_input_path: Path to tabular input file.
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

                # Helper to resolve field precedence: row > base_config
                def get_field(field_name):
                    if field_name in row.index and pd.notna(row.get(field_name)) and row.get(field_name) != "":
                        return row.get(field_name)
                    return base_config.get(field_name)

                training_data_type = get_field("training_data_type")
                training_data = get_field("training_data")
                validation_data_type = get_field("validation_data_type")
                validation_data = get_field("validation_data")
                repartition_val = 200
                if "repartition" in row.index and pd.notna(row.get("repartition")):
                    try:
                        repartition_val = int(row.get("repartition"))
                    except Exception:
                        logger.warning("Invalid repartition value at row %s, defaulting to 200", idx)

                if not training_data_type or not training_data:
                    raise ValueError(f"Missing training_data or training_data_type for row {idx}")
                if not validation_data_type or not validation_data:
                    raise ValueError(f"Missing validation_data or validation_data_type for row {idx}")

                # Load data with repartition if applicable
                training_df = load_data(training_data_type, training_data, repartition=repartition_val)
                validation_df = load_data(validation_data_type, validation_data, repartition=repartition_val)

                # Construct synthetic row to include fallbacks
                synthetic_row = row.copy()
                for field in FALLBACK_FIELDS:
                    if (field not in synthetic_row.index) or pd.isna(synthetic_row.get(field)) or synthetic_row.get(field) == "":
                        if field in base_config:
                            synthetic_row[field] = base_config[field]

                validated_optional = validate_columns_exist(synthetic_row, training_df, validation_df, idx)

                # Build merged config
                merged_config = base_config.copy()

                # Add direct row overrides except excluded ones
                excluded = {
                    "training_data_type", "training_data",
                    "validation_data_type", "validation_data",
                    "time_var", "time_series_vars", "categorical_vars", "scenario_var",
                    "optional_test", "optional_performance_tests",
                    "omit_default_diagnostic_tests", "omit_performance_tests", "repartition"
                }
                for key, value in row.dropna().to_dict().items():
                    if key not in excluded:
                        merged_config[key] = value

                # Attach dataframes
                merged_config["training_data_df"] = training_df
                merged_config["validation_data_df"] = validation_df

                # Build model_setting and optional groups
                model_setting = {}
                for mkey in ["time_var", "time_series_vars", "categorical_vars", "scenario_var"]:
                    val = validated_optional.pop(mkey, None)
                    if val is not None and (not isinstance(val, list) or len(val) > 0):
                        model_setting[mkey] = val
                if model_setting:
                    merged_config["model_setting"] = model_setting
                else:
                    merged_config.pop("model_setting", None)

                for remaining_key, remaining_val in validated_optional.items():
                    if remaining_val is not None and (not isinstance(remaining_val, list) or len(remaining_val) > 0):
                        merged_config[remaining_key] = remaining_val

                logger.debug("Final merged config for row %s: %s", idx, merged_config)
                process_model(merged_config)

            except Exception as row_err:
                logger.error("Row %s processing failed: %s", idx, row_err, exc_info=True)
    except Exception as e:
        logger.critical("Fatal error during pipeline execution: %s", e, exc_info=True)


# ------------------ CLI ------------------

def cli():
    """
    Command-line entry point.
    """
    parser = argparse.ArgumentParser(description="Run the model processing pipeline.")
    parser.add_argument("--model_config", required=True, help="Path to model config file (YAML/JSON) or JSON string.")
    parser.add_argument("--tabular_input", required=True, help="Path to tabular input file (CSV/JSON/YAML).")
    args = parser.parse_args()

    # Determine model_config input
    model_config_input = args.model_config
    if os.path.exists(model_config_input):
        pass  # file path, will be consumed directly
    else:
        try:
            model_config_input = json.loads(model_config_input)
        except json.JSONDecodeError:
            logger.error("model_config is neither a valid file nor JSON string.")
            raise

    main(model_config_input, args.tabular_input)


if __name__ == "__main__":
    cli()



import pytest
import pandas as pd
import os
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import from your module
import model_processor  # assume the script is named model_processor.py

# Aliases for clarity
load_model_config = model_processor.load_model_config
validate_model_config = model_processor.validate_model_config
validate_row = model_processor.validate_row
parse_list_column = model_processor.parse_list_column
validate_columns_exist = model_processor.validate_columns_exist
load_tabular_input = model_processor.load_tabular_input
load_data = model_processor.load_data
main = model_processor.main


# ---------- Fixtures ----------

@pytest.fixture
def base_config(tmp_path):
    base = {
        "model_id": "100",
        "submission_id": "200",
        "model_name": "base_model",
        "base_path": str(tmp_path),
        # fallback values
        "training_data_type": "path",
        "training_data": str(tmp_path / "train_base.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "valid_base.csv"),
        "time_var": "d",
        "time_series_vars": ["f1", "f2"],
        "categorical_vars": ["cat"],
        "scenario_var": ["scen"],
        "optional_test": ["ot"],
        "optional_performance_tests": ["opt"],
        "omit_default_diagnostic_tests": ["omit"],
        "omit_performance_tests": ["omit_perf"],
    }
    # create fallback train/valid
    df = pd.DataFrame({
        "d": ["2020-01-01"],
        "f1": [1], "f2": [2],
        "cat": ["A"],
        "scen": ["S"]
    })
    df.to_csv(tmp_path / "train_base.csv", index=False)
    df.to_csv(tmp_path / "valid_base.csv", index=False)
    return base

@pytest.fixture
def minimal_tabular(tmp_path):
    # only required fields; no optional fields
    df_tab = pd.DataFrame([{
        "sub_model_name": "m1",
        "model_type": "xgboost",
        "training_data_type": "path",
        "training_data": str(tmp_path / "train.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "valid.csv"),
    }])
    df = pd.DataFrame({
        "d": ["2021-02-02"],
        "f1": [5], "f2": [6],
        "cat": ["B"],
        "scen": ["T"]
    })
    df.to_csv(tmp_path / "train.csv", index=False)
    df.to_csv(tmp_path / "valid.csv", index=False)
    return df_tab

@pytest.fixture
def full_tabular(tmp_path):
    df_tab = pd.DataFrame([{
        "sub_model_name": "m_full",
        "model_type": "random_forest",
        "training_data_type": "path",
        "training_data": str(tmp_path / "train_full.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "valid_full.csv"),
        "time_var": "time",
        "time_series_vars": "a,b",
        "categorical_vars": "cat1|cat2",
        "scenario_var": "scen1",
        "optional_test": "[t1, t2]",
        "optional_performance_tests": "pf1",
        "omit_default_diagnostic_tests": "od1",
        "omit_performance_tests": "op1",
        "repartition": 10
    }])
    df = pd.DataFrame({
        "time": ["2022-03-03"],
        "a": [1], "b": [2],
        "cat1": ["X"], "cat2": ["Y"],
        "scen1": ["Z"]
    })
    df.to_csv(tmp_path / "train_full.csv", index=False)
    df.to_csv(tmp_path / "valid_full.csv", index=False)
    return df_tab

# ---------- Tests ----------

def test_load_model_config_from_dict(base_config):
    cfg = load_model_config(base_config)
    assert isinstance(cfg, dict)
    assert cfg["model_id"] == "100"

def test_load_model_config_from_yaml(tmp_path, base_config):
    p = tmp_path / "conf.yaml"
    with open(p, "w") as f:
        yaml.dump(base_config, f)
    cfg = load_model_config(str(p))
    assert cfg["submission_id"] == "200"

def test_validate_model_config_success(base_config):
    # should not raise
    validate_model_config(base_config)

def test_validate_model_config_missing_key():
    with pytest.raises(KeyError):
        validate_model_config({"model_id": "1", "submission_id": "2", "model_name": "x"})  # missing base_path

def test_validate_row_good():
    row = pd.Series({"sub_model_name": "good", "model_type": "xgboost"})
    validate_row(row, 0)

def test_validate_row_bad_name():
    row = pd.Series({"sub_model_name": "bad name", "model_type": "xgboost"})
    with pytest.raises(ValueError):
        validate_row(row, 1)

def test_validate_row_bad_type():
    row = pd.Series({"sub_model_name": "good", "model_type": "unknown"})
    with pytest.raises(ValueError):
        validate_row(row, 1)

@pytest.mark.parametrize("input_value,expected", [
    ("a,b,c", ["a", "b", "c"]),
    ("['a','b']", ["a", "b"]),
    ("x", ["x"]),
    (["x", "y"], ["x", "y"]),
    ("", []),
    (None, []),
    ("a|b", ["a", "b"]),
])
def test_parse_list_column(input_value, expected):
    result = parse_list_column(input_value, index=2, colname="test")
    assert result == expected

def test_validate_columns_exist_full_override(full_tabular, tmp_path):
    row = full_tabular.iloc[0]
    train_df = pd.read_csv(row["training_data"])
    valid_df = pd.read_csv(row["validation_data"])
    result = validate_columns_exist(row, train_df, valid_df, 0)
    assert result["time_var"] == "time"
    assert result["time_series_vars"] == ["a", "b"]
    assert result["categorical_vars"] == ["cat1", "cat2"]
    assert result["scenario_var"] == ["scen1"]
    assert result["optional_test"] == ["t1", "t2"]
    assert result["optional_performance_tests"] == ["pf1"]
    assert result["omit_default_diagnostic_tests"] == ["od1"]
    assert result["omit_performance_tests"] == ["op1"]
    assert result["repartition"] == 10

def test_main_minimal_with_fallback(tmp_path, minimal_tabular, base_config):
    # Write base config
    config_path = tmp_path / "base.yaml"
    with open(config_path, "w") as f:
        yaml.dump(base_config, f)
    # Tabular input
    input_path = tmp_path / "min.csv"
    minimal_tabular.to_csv(input_path, index=False)
    # Should run without exception (uses fallback for optional/model_setting)
    main(str(config_path), str(input_path))

def test_main_full_override(tmp_path, full_tabular, base_config):
    # Write base config
    config_path = tmp_path / "base.yaml"
    with open(config_path, "w") as f:
        yaml.dump(base_config, f)
    input_path = tmp_path / "full.csv"
    full_tabular.to_csv(input_path, index=False)
    # Should run without exception
    main(str(config_path), str(input_path))

def test_invalid_model_type_integration(tmp_path):
    # Create minimal config
    base = {
        "model_id": "1",
        "submission_id": "2",
        "model_name": "mod",
        "base_path": str(tmp_path),
        "training_data_type": "path",
        "training_data": str(tmp_path / "t.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "v.csv"),
    }
    # Create files
    pd.DataFrame({"a":[1]}).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"a":[2]}).to_csv(tmp_path / "v.csv", index=False)
    # Tabular with bad type
    df_tab = pd.DataFrame([{
        "sub_model_name": "m",
        "model_type": "BAD",
        "training_data_type": "path",
        "training_data": str(tmp_path / "t.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "v.csv"),
    }])
    input_path = tmp_path / "bad.csv"
    df_tab.to_csv(input_path, index=False)
    config_path = tmp_path / "base.yaml"
    with open(config_path, "w") as f:
        yaml.dump(base, f)
    with pytest.raises(ValueError):
        main(str(config_path), str(input_path))

# ---------- Spark/HDFS Mocking Tests ----------

@patch("model_processor.get_spark_session")
def test_load_data_hdfs_with_repartition(mock_spark_session, tmp_path):
    # Mock spark session and DataFrame
    fake_spark = MagicMock()
    mock_spark_session.return_value = fake_spark
    fake_df = MagicMock()
    fake_spark.read.parquet.return_value = fake_df
    fake_df.repartition.return_value = fake_df

    # Call load_data for hdfs
    df = load_data("hdfs", "some/path", repartition=5)
    fake_spark.read.parquet.assert_called_once_with("some/path")
    fake_df.repartition.assert_called_once_with(5)
    assert df == fake_df

@patch("model_processor.get_spark_session")
def test_load_data_hive(mock_spark_session):
    fake_spark = MagicMock()
    mock_spark_session.return_value = fake_spark
    fake_table = MagicMock()
    fake_spark.read.table.return_value = fake_table

    df = load_data("hive", "db.table")
    fake_spark.read.table.assert_called_once_with("db.table")
    assert df == fake_table
  
