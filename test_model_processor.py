
import pytest
import pandas as pd
import os
from main import (
    load_model_config,
    validate_model_config,
    validate_row,
    load_tabular_input,
    parse_list_column,
    validate_columns_exist,
    process_model,
    main,
)

@pytest.fixture
def valid_model_config(tmp_path):
    config = {
        "model_id": "123",
        "submission_id": "456",
        "model_name": "valid_model_name",
        "base_path": str(tmp_path)
    }
    return config

@pytest.fixture
def valid_tabular_df(tmp_path):
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    df_data = pd.DataFrame({
        "timestamp": ["2021-01-01"],
        "feature1": [1],
        "feature2": [2],
        "category1": ["A"],
        "scenario1": ["X"],
        "scenario2": ["Y"]
    })
    df_data.to_csv(train_path, index=False)
    df_data.to_csv(valid_path, index=False)

    return pd.DataFrame([{
        "sub_model_name": "model_1",
        "model_type": "xgboost",
        "training_data_type": "path",
        "training_data": str(train_path),
        "validation_data_type": "path",
        "validation_data": str(valid_path),
        "time_var": "timestamp",
        "time_series_vars": "feature1, feature2",
        "categorical_vars": "category1",
        "scenario_var": "scenario1, scenario2",
        "optional_test": "test1, test2",
        "optional_performance_tests": "perf1",
        "omit_default_diagnostic_tests": "",
        "omit_performance_tests": "perf2"
    }])

def test_load_model_config_dict(valid_model_config):
    assert isinstance(load_model_config(valid_model_config), dict)

def test_validate_model_config_success(valid_model_config):
    validate_model_config(valid_model_config)  # should not raise

def test_validate_model_config_failure_missing_key():
    with pytest.raises(KeyError):
        validate_model_config({"model_id": "123"})

def test_validate_row_valid(valid_tabular_df):
    validate_row(valid_tabular_df.iloc[0], 0)  # should not raise

def test_parse_list_column():
    assert parse_list_column("a, b", 0, "test") == ["a", "b"]
    assert parse_list_column("['a', 'b']", 0, "test") == ["a", "b"]
    assert parse_list_column("", 0, "test") == []

def test_validate_columns_exist(valid_tabular_df):
    row = valid_tabular_df.iloc[0]
    df = pd.read_csv(row["training_data"])
    validate_columns_exist(row, df, df, 0)

def test_main_success(tmp_path):
    # Prepare model_config
    config_path = tmp_path / "model.yaml"
    config_data = {
        "model_id": "123",
        "submission_id": "456",
        "model_name": "valid_model_name",
        "base_path": str(tmp_path)
    }
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Prepare input CSV
    input_df = pd.DataFrame([{
        "sub_model_name": "model_1",
        "model_type": "xgboost",
        "training_data_type": "path",
        "training_data": str(tmp_path / "train.csv"),
        "validation_data_type": "path",
        "validation_data": str(tmp_path / "valid.csv"),
        "time_var": "timestamp",
        "time_series_vars": "feature1, feature2",
        "categorical_vars": "category1",
        "scenario_var": "scenario1, scenario2",
        "optional_test": "test1, test2",
        "optional_performance_tests": "perf1",
        "omit_default_diagnostic_tests": "",
        "omit_performance_tests": "perf2"
    }])
    input_path = tmp_path / "input.csv"
    input_df.to_csv(input_path, index=False)

    # Create dummy training/validation CSVs
    df = pd.DataFrame({
        "timestamp": ["2021-01-01"],
        "feature1": [1],
        "feature2": [2],
        "category1": ["A"],
        "scenario1": ["X"],
        "scenario2": ["Y"]
    })
    df.to_csv(tmp_path / "train.csv", index=False)
    df.to_csv(tmp_path / "valid.csv", index=False)

    # Run main
    main(str(config_path), str(input_path))
