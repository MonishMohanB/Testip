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


def load_model_config(config_input: Union[str, Dict]) -> Dict:
    if isinstance(config_input, dict):
        return config_input
    if not os.path.exists(config_input):
        raise FileNotFoundError(f"Config file not found: {config_input}")
    with open(config_input, "r") as f:
        if config_input.endswith(".yaml") or config_input.endswith(".yml"):
            return yaml.safe_load(f)
        elif config_input.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format")


def validate_model_config(config: Dict):
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
    name = row.get("sub_model_name", "")
    if not re.match(r"^[a-zA-Z0-9_]+$", name):
        raise ValueError(f"Invalid sub_model_name at row {index}")

    model_type = row.get("model_type")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model_type '{model_type}' at row {index}. Must be one of {VALID_MODEL_TYPES}")


def parse_list_column(value, index, colname):
    try:
        if pd.notna(value):
            return ast.literal_eval(value) if isinstance(value, str) else value
        return []
    except Exception:
        raise ValueError(f"Invalid list format in column '{colname}' at row {index}: {value}")


def validate_columns_exist(row, train_df, valid_df, index):
    time_var = row.get("time_var")
    time_series_vars = parse_list_column(row.get("time_series_vars"), index, "time_series_vars")
    categorical_vars = parse_list_column(row.get("categorical_vars"), index, "categorical_vars")
    scenario_var = parse_list_column(row.get("scenario_var"), index, "scenario_var")
    optional_test = parse_list_column(row.get("optional_test"), index, "optional_test")
    optional_perf_tests = parse_list_column(row.get("optional_performance_tests"), index, "optional_performance_tests")
    omit_diag_tests = parse_list_column(row.get("omit_default_diagnostic_tests"), index, "omit_default_diagnostic_tests")
    omit_perf_tests = parse_list_column(row.get("omit_performance_tests"), index, "omit_performance_tests")

    for col_list, df, df_name in [
        (time_series_vars, train_df, "training_data"),
        (categorical_vars, train_df, "training_data"),
        (scenario_var, train_df, "training_data"),
        (time_series_vars, valid_df, "validation_data"),
        (categorical_vars, valid_df, "validation_data"),
        (scenario_var, valid_df, "validation_data")
    ]:
        for col in col_list:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {df_name} at row {index}")

    if pd.notna(time_var):
        if time_var not in train_df.columns:
            raise ValueError(f"Time var '{time_var}' not found in training data at row {index}")
        if time_var not in valid_df.columns:
            raise ValueError(f"Time var '{time_var}' not found in validation data at row {index}")

    return {
        "time_var": time_var,
        "time_series_vars": time_series_vars,
        "categorical_vars": categorical_vars,
        "scenario_var": scenario_var,
        "optional_test": optional_test,
        "optional_performance_tests": optional_perf_tests,
        "omit_default_diagnostic_tests": omit_diag_tests,
        "omit_performance_tests": omit_perf_tests,
    }


def load_tabular_input(path: str) -> pd.DataFrame:
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
    if data_type == "path":
        return pd.read_csv(path)
    elif data_type == "hdfs":
        # simulate spark read for testing
        return pd.DataFrame()  # placeholder
    elif data_type == "hive":
        # simulate spark read for testing
        return pd.DataFrame()  # placeholder
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def process_model(config: Dict):
    logger.info(f"Processing model: {config.get('sub_model_name')}")


def main(model_config_input: Union[str, Dict], tabular_input_path: str):
    try:
        config = load_model_config(model_config_input)
        validate_model_config(config)
        df = load_tabular_input(tabular_input_path)

        for idx, row in df.iterrows():
            try:
                validate_row(row, idx)
                row_data = row.dropna().to_dict()

                training_df = load_data(row_data['training_data_type'], row_data['training_data'])
                validation_df = load_data(row_data['validation_data_type'], row_data['validation_data'])

                validated_optional = validate_columns_exist(row, training_df, validation_df, idx)

                merged_config = config.copy()
                for key, value in row_data.items():
                    if value is not None and pd.notna(value):
                        merged_config[key] = value

                merged_config.update(validated_optional)
                merged_config['training_data_df'] = training_df
                merged_config['validation_data_df'] = validation_df

                process_model(merged_config)

            except Exception as row_err:
                logger.error(f"Row {idx} processing failed: {row_err}", exc_info=True)

    except Exception as e:
        logger.critical(f"Fatal error during processing: {e}", exc_info=True)
