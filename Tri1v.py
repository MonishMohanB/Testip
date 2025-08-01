import os
import re
import json
import yaml
import pandas as pd

def validate_model_config(model_config):
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a dictionary.")

    for key in ['model_id', 'submission_id']:
        if key not in model_config:
            raise KeyError(f"{key} is missing in model_config.")
        if not str(model_config[key]).isdigit():
            raise ValueError(f"{key} must be numeric. Got: {model_config[key]}")

    model_name = model_config.get('model_name')
    if not isinstance(model_name, str) or not re.match(r'^[A-Za-z0-9]+(_[A-Za-z0-9]+)*$', model_name):
        raise ValueError("model_name must be a string with only letters, numbers, and underscores between words.")

def load_tabular_input(tabular_input_path):
    if not os.path.isfile(tabular_input_path):
        raise FileNotFoundError(f"{tabular_input_path} not found.")

    _, ext = os.path.splitext(tabular_input_path)

    try:
        if ext.lower() in ['.csv']:
            return pd.read_csv(tabular_input_path)
        elif ext.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(tabular_input_path)
        elif ext.lower() in ['.json']:
            return pd.read_json(tabular_input_path)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(tabular_input_path, 'r') as f:
                data = yaml.safe_load(f)
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        raise ValueError(f"Error loading tabular input: {e}")

def process_model(config):
    """
    This function simulates processing of a model.
    Replace this with your actual processing logic.
    """
    print(f"Processing model with config: {config}")

def main(model_config, tabular_input_path):
    try:
        validate_model_config(model_config)
        df = load_tabular_input(tabular_input_path)

        required_columns = {'sub_model_name', 'training_data', 'validation_data', 'model_type'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns in tabular input: {missing}")

        for idx, row in df.iterrows():
            try:
                merged_config = model_config.copy()
                merged_config.update(row.to_dict())
                process_model(merged_config)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")

    except Exception as e:
        print(f"Fatal error: {e}")

# Example usage:
if __name__ == "__main__":
    model_config = {
        "model_id": "12345",
        "submission_id": "67890",
        "model_name": "example_model"
    }

    tabular_input_path = "models_data.csv"  # or .xlsx, .json, .yaml
    main(model_config, tabular_input_path)


import pytest
import pandas as pd
import tempfile
import json
import yaml
import os
from unittest.mock import patch
from your_module_name import (
    validate_model_config,
    load_tabular_input,
    main,
)

# ------------------------
# Fixtures and sample data
# ------------------------

valid_model_config = {
    "model_id": "123",
    "submission_id": "456",
    "model_name": "valid_model_name"
}

invalid_model_configs = [
    {"model_id": "abc", "submission_id": "456", "model_name": "valid_model_name"},  # model_id not numeric
    {"model_id": "123", "submission_id": "xyz", "model_name": "valid_model_name"},  # submission_id not numeric
    {"model_id": "123", "submission_id": "456", "model_name": "invalid name!"},     # invalid model_name
    {},  # missing all keys
]

sample_df = pd.DataFrame([
    {"sub_model_name": "sub1", "training_data": "train1.csv", "validation_data": "val1.csv", "model_type": "typeA"},
    {"sub_model_name": "sub2", "training_data": "train2.csv", "validation_data": "val2.csv", "model_type": "typeB"},
])

# ------------------------
# Unit tests
# ------------------------

def test_validate_model_config_valid():
    validate_model_config(valid_model_config)

@pytest.mark.parametrize("bad_config", invalid_model_configs)
def test_validate_model_config_invalid(bad_config):
    with pytest.raises((ValueError, KeyError)):
        validate_model_config(bad_config)

@pytest.mark.parametrize("ext,writer", [
    (".csv", lambda df, path: df.to_csv(path, index=False)),
    (".json", lambda df, path: df.to_json(path, orient="records")),
    (".yaml", lambda df, path: yaml.dump(df.to_dict(orient="records"), open(path, 'w'))),
    (".xlsx", lambda df, path: df.to_excel(path, index=False)),
])
def test_load_tabular_input_valid_formats(ext, writer):
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmpfile:
        path = tmpfile.name
    writer(sample_df, path)
    df = load_tabular_input(path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    os.remove(path)

def test_load_tabular_input_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_tabular_input("non_existing_file.csv")

def test_load_tabular_input_invalid_extension():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmpfile:
        tmpfile.write(b"dummy")
        path = tmpfile.name
    with pytest.raises(ValueError):
        load_tabular_input(path)
    os.remove(path)

# ------------------------
# Integration test with mocking
# ------------------------

def test_main_process_model_called(monkeypatch):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
        sample_df.to_csv(tmpfile.name, index=False)
        path = tmpfile.name

    calls = []

    def mock_process_model(config):
        calls.append(config)

    monkeypatch.setattr("your_module_name.process_model", mock_process_model)
    main(valid_model_config, path)
    assert len(calls) == 2  # Should process 2 rows
    os.remove(path)

def test_main_handles_row_error(monkeypatch):
    # Invalid row data: missing required columns
    broken_df = pd.DataFrame([
        {"not_required": "oops"}
    ])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
        broken_df.to_csv(tmpfile.name, index=False)
        path = tmpfile.name

    with patch("your_module_name.process_model") as mock_process:
        mock_process.side_effect = Exception("Shouldn't be called")
        with pytest.raises(ValueError, match="Missing required columns"):
            main(valid_model_config, path)
    os.remove(path)



import os
import re
import json
import yaml
import pandas as pd
import logging
from pyspark.sql import SparkSession

# -------------------------------
# Logger Setup
# -------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(ch)

# -------------------------------
# Spark Setup
# -------------------------------
spark = SparkSession.builder.appName("ModelProcessor").getOrCreate()

# -------------------------------
# Config Validation
# -------------------------------

def load_model_config(model_config_input):
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

def validate_model_config(config):
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

# -------------------------------
# Tabular Input Loader & Validator
# -------------------------------

def load_tabular_input(tabular_input_path):
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

def validate_row(row, idx):
    required_columns = {
        'sub_model_name', 'training_data', 'validation_data',
        'training_data_type', 'validation_data_type', 'model_type'
    }

    missing = required_columns - set(row.index)
    if missing:
        raise ValueError(f"Row {idx} is missing required columns: {missing}")

    if not isinstance(row['sub_model_name'], str) or not re.match(r'^[A-Za-z0-9]+(_[A-Za-z0-9]+)*$', row['sub_model_name']):
        raise ValueError(f"Row {idx}: Invalid sub_model_name: {row['sub_model_name']}")

# -------------------------------
# Data Loader (training/validation)
# -------------------------------

def load_data(source_type, source_value):
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

# -------------------------------
# Model Processor
# -------------------------------

def process_model(config):
    # Replace this with actual model training or inference logic
    logger.info(f"Processing sub_model: {config.get('sub_model_name')}")

# -------------------------------
# Main Workflow
# -------------------------------

def main(model_config_input, tabular_input_path):
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
        

