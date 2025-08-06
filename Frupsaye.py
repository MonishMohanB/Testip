# Build model_setting and clean empty list fields
model_setting = {
    k: v for k, v in validated_optional.items()
    if v is not None and (not isinstance(v, list) or len(v) > 0)
}

# Add custom (unknown) keys from row_data to model_setting
known_keys = {
    'sub_model_name', 'model_type',
    'training_data', 'training_data_type',
    'validation_data', 'validation_data_type',
    'time_var', 'time_series_vars', 'categorical_vars', 'scenario_var',
    'optional_test', 'optional_performance_tests',
    'omit_default_diagnostic_tests', 'omit_performance_tests',
    'repartition'
}

for key, value in row_data.items():
    if key not in known_keys and value is not None and pd.notna(value):
        logger.info(f"Adding extra column '{key}' to model_setting from row {idx}")
        model_setting[key] = value

# Only add model_setting if it contains any values
if model_setting:
    merged_config['model_setting'] = model_setting
elif 'model_setting' in merged_config:
    del merged_config['model_setting']
