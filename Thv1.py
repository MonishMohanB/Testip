import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import os

# Assuming the functions are in a module named 'data_utils'
# Replace 'data_utils' with the actual module name
# from data_utils import (
#     manage_directory,
#     merge_yaml_columns,
#     remove_missing,
#     apply_get_dummies,
#     apply_lambda,
#     load_csv,
# )

# Implementations of the functions for testing purposes (if not available)
def manage_directory(path):
    path = Path(path)
    if path.exists():
        for file in path.glob('*'):
            file.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)

def merge_yaml_columns(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    combined = []
    for column in data.get('columns', []):
        combined.extend(column)
    return combined

def remove_missing(df):
    return df.dropna()

def apply_get_dummies(df, columns):
    return pd.get_dummies(df, columns=columns)

def apply_lambda(df, column, func):
    df[column] = df[column].apply(func)
    return df

def load_csv(file_path):
    file_path = Path(file_path)
    if file_path.exists():
        return pd.read_csv(file_path)
    return None


# Tests
def test_manage_directory(tmp_path):
    # Test directory creation if it doesn't exist
    dir_path = tmp_path / "test_library"
    manage_directory(dir_path)
    assert dir_path.exists() and dir_path.is_dir()

    # Test directory cleanup if it exists
    test_file = dir_path / "test.txt"
    test_file.write_text("dummy content")
    manage_directory(dir_path)
    assert not list(dir_path.iterdir()), "Directory should be empty after cleanup"

def test_merge_yaml_columns(tmp_path):
    yaml_data = {
        'columns': [
            [1, 2, 3],
            [4, 5, 6]
        ]
    }
    yaml_file = tmp_path / "test.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_data, f)
    
    combined = merge_yaml_columns(yaml_file)
    assert combined == [1, 2, 3, 4, 5, 6], "Columns should be merged into a single list"

def test_remove_missing():
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, np.nan]
    })
    cleaned_df = remove_missing(df)
    assert cleaned_df.isna().sum().sum() == 0, "DataFrame should have no missing values"
    assert len(cleaned_df) == 1, "Only one row without missing values should remain"

def test_apply_get_dummies():
    df = pd.DataFrame({
        'Category': ['A', 'B', 'A', 'C']
    })
    dummied_df = apply_get_dummies(df, ['Category'])
    expected_columns = ['Category_A', 'Category_B', 'Category_C']
    assert all(col in dummied_df.columns for col in expected_columns), "All category dummies should be present"
    assert dummied_df.shape == (4, 3), "DataFrame should have 4 rows and 3 dummy columns"

def test_apply_lambda():
    df = pd.DataFrame({'Values': [1, 2, 3]})
    apply_lambda(df, 'Values', lambda x: x ** 2)
    assert df['Values'].tolist() == [1, 4, 9], "Lambda function should square the values"

def test_load_csv(tmp_path):
    # Test loading existing CSV
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({'X': [10, 20], 'Y': [30, 40]})
    df.to_csv(csv_path, index=False)
    loaded_df = load_csv(csv_path)
    pd.testing.assert_frame_equal(loaded_df, df), "Loaded DataFrame should match the original"

    # Test non-existent CSV
    non_existent_path = tmp_path / "nonexistent.csv"
    assert load_csv(non_existent_path) is None, "Non-existent file should return None"
